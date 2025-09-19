#include "include/AudioFingerprinting.h"
#include "include/ProductionFFT.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <chrono>
#include <sqlite3.h>

// Internal database context
struct fingerprint_db {
    fingerprint_config_t config;
    sqlite3* db_handle;
    std::unique_ptr<AudioFingerprinting::LSHManager> lsh_manager;
    std::unique_ptr<AudioFingerprinting::FingerprintDatabase> cpp_database;
    bool is_initialized;
    
    fingerprint_db() : db_handle(nullptr), is_initialized(false) {}
};

namespace {
    // Database schema
    const char* CREATE_FINGERPRINTS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_id INTEGER UNIQUE NOT NULL,
            source_filename TEXT,
            file_hash INTEGER,
            duration_seconds REAL,
            sample_rate REAL,
            creation_timestamp REAL,
            num_hashes INTEGER,
            num_peaks INTEGER
        )
    )";
    
    const char* CREATE_HASHES_TABLE = R"(
        CREATE TABLE IF NOT EXISTS hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_id INTEGER,
            hash_value INTEGER,
            hash_index INTEGER,
            FOREIGN KEY(fingerprint_id) REFERENCES fingerprints(fingerprint_id)
        )
    )";
    
    const char* CREATE_PEAKS_TABLE = R"(
        CREATE TABLE IF NOT EXISTS peaks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_id INTEGER,
            frequency_bin INTEGER,
            time_frame INTEGER,
            magnitude REAL,
            frequency_hz REAL,
            timestamp_seconds REAL,
            FOREIGN KEY(fingerprint_id) REFERENCES fingerprints(fingerprint_id)
        )
    )";
    
    const char* CREATE_INDICES = R"(
        CREATE INDEX IF NOT EXISTS idx_hashes_value ON hashes(hash_value);
        CREATE INDEX IF NOT EXISTS idx_hashes_fingerprint ON hashes(fingerprint_id);
        CREATE INDEX IF NOT EXISTS idx_peaks_fingerprint ON peaks(fingerprint_id);
        CREATE INDEX IF NOT EXISTS idx_fingerprints_file_hash ON fingerprints(file_hash);
    )";
    
    // Hash generation utilities
    uint64_t generate_file_hash(const char* filename) {
        std::hash<std::string> hasher;
        return hasher(std::string(filename));
    }
    
    double get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
    }
    
    // Simple hash function for combining two 32-bit values into 64-bit
    uint64_t combine_hash(uint32_t a, uint32_t b) {
        return (static_cast<uint64_t>(a) << 32) | b;
    }
}

// C Interface Implementation
int32_t fingerprint_db_create(const fingerprint_config_t* config, fingerprint_db_t** db) {
    if (!config || !db) {
        return FINGERPRINT_ERROR_INVALID_PARAMS;
    }
    
    auto* ctx = new fingerprint_db();
    if (!ctx) {
        return FINGERPRINT_ERROR_MEMORY_ALLOCATION;
    }
    
    try {
        ctx->config = *config;
        
        // Open SQLite database
        int result = sqlite3_open(config->database_path, &ctx->db_handle);
        if (result != SQLITE_OK) {
            delete ctx;
            return FINGERPRINT_ERROR_DATABASE_ERROR;
        }
        
        // Create tables
        char* error_msg = nullptr;
        
        result = sqlite3_exec(ctx->db_handle, CREATE_FINGERPRINTS_TABLE, nullptr, nullptr, &error_msg);
        if (result != SQLITE_OK) {
            sqlite3_free(error_msg);
            sqlite3_close(ctx->db_handle);
            delete ctx;
            return FINGERPRINT_ERROR_DATABASE_ERROR;
        }
        
        result = sqlite3_exec(ctx->db_handle, CREATE_HASHES_TABLE, nullptr, nullptr, &error_msg);
        if (result != SQLITE_OK) {
            sqlite3_free(error_msg);
            sqlite3_close(ctx->db_handle);
            delete ctx;
            return FINGERPRINT_ERROR_DATABASE_ERROR;
        }
        
        result = sqlite3_exec(ctx->db_handle, CREATE_PEAKS_TABLE, nullptr, nullptr, &error_msg);
        if (result != SQLITE_OK) {
            sqlite3_free(error_msg);
            sqlite3_close(ctx->db_handle);
            delete ctx;
            return FINGERPRINT_ERROR_DATABASE_ERROR;
        }
        
        result = sqlite3_exec(ctx->db_handle, CREATE_INDICES, nullptr, nullptr, &error_msg);
        if (result != SQLITE_OK) {
            sqlite3_free(error_msg);
            sqlite3_close(ctx->db_handle);
            delete ctx;
            return FINGERPRINT_ERROR_DATABASE_ERROR;
        }
        
        // Initialize LSH manager
        ctx->lsh_manager = std::make_unique<AudioFingerprinting::LSHManager>(*config);
        
        // Initialize C++ database wrapper
        ctx->cpp_database = std::make_unique<AudioFingerprinting::FingerprintDatabase>(*config);
        
        ctx->is_initialized = true;
        *db = ctx;
        
        return FINGERPRINT_SUCCESS;
        
    } catch (...) {
        if (ctx->db_handle) {
            sqlite3_close(ctx->db_handle);
        }
        delete ctx;
        return FINGERPRINT_ERROR_MEMORY_ALLOCATION;
    }
}

void fingerprint_db_destroy(fingerprint_db_t* db) {
    if (!db) return;
    
    if (db->db_handle) {
        sqlite3_close(db->db_handle);
    }
    
    delete db;
}

fingerprint_config_t fingerprint_get_default_config(void) {
    fingerprint_config_t config = {};
    
    // Spectral analysis parameters
    config.fft_size = 2048;
    config.hop_length = 512;
    config.num_mel_filters = 26;
    config.sample_rate = 48000.0;
    
    // Peak extraction parameters
    config.num_peaks_per_frame = 5;
    config.peak_threshold_db = -40.0;
    config.peak_neighborhood_size = 3;
    config.enable_peak_filtering = true;
    
    // Hash generation parameters
    config.hash_time_delta = 4;
    config.hash_freq_delta = 2;
    config.fingerprint_duration_frames = 100;
    config.enable_perceptual_hashing = true;
    
    // LSH parameters
    config.lsh_num_tables = 16;
    config.lsh_hash_length = 64;
    config.lsh_key_length = 8;
    config.lsh_similarity_threshold = 0.7;
    
    // Database parameters
    config.database_path = "fingerprints.db";
    config.cache_size = 1000;
    config.enable_database_compression = true;
    config.max_database_entries = 1000000;
    
    return config;
}

// C++ Implementation
namespace AudioFingerprinting {

// SpectralPeakExtractor implementation
SpectralPeakExtractor::SpectralPeakExtractor(const fingerprint_config_t& config) 
    : config_(config), fft_processor_(nullptr) {
    
    fft_config_t fft_config = fft_get_default_config(config_.fft_size);
    fft_config.window_type = WINDOW_HANN;
    
    // Create FFT processor and store as opaque pointer
    auto* processor = new ProductionFFT::FFTProcessor(fft_config);
    fft_processor_ = static_cast<void*>(processor);
}

SpectralPeakExtractor::~SpectralPeakExtractor() {
    if (fft_processor_) {
        auto* processor = static_cast<ProductionFFT::FFTProcessor*>(fft_processor_);
        delete processor;
        fft_processor_ = nullptr;
    }
}

std::vector<spectral_peak_t> SpectralPeakExtractor::extractPeaks(
    const std::vector<float>& audio,
    double sample_rate) {
    
    // Compute spectrogram
    auto spectrogram = computeSpectrogram(audio, sample_rate);
    
    // Extract peaks from spectrogram
    return extractPeaksFromSpectrogram(spectrogram, sample_rate, config_.hop_length);
}

std::vector<spectral_peak_t> SpectralPeakExtractor::extractPeaksFromSpectrogram(
    const std::vector<std::vector<float>>& spectrogram,
    double sample_rate,
    uint32_t hop_length) {
    
    auto peaks = findLocalMaxima(spectrogram, sample_rate, hop_length);
    
    if (config_.enable_peak_filtering) {
        filterPeaks(peaks);
    }
    
    return peaks;
}

std::vector<std::vector<float>> SpectralPeakExtractor::computeSpectrogram(
    const std::vector<float>& audio,
    double sample_rate) {
    
    std::vector<std::vector<float>> spectrogram;
    
    uint32_t num_frames = (audio.size() - config_.fft_size) / config_.hop_length + 1;
    spectrogram.reserve(num_frames);
    
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        uint32_t start_idx = frame * config_.hop_length;
        
        // Extract window
        std::vector<float> window(audio.begin() + start_idx, 
                                audio.begin() + start_idx + config_.fft_size);
        
        // Compute FFT
        auto* processor = static_cast<ProductionFFT::FFTProcessor*>(fft_processor_);
        auto spectrum = processor->computeFFT(window);
        
        // Convert to magnitude spectrum
        std::vector<float> magnitude(spectrum.size());
        for (size_t i = 0; i < spectrum.size(); i++) {
            magnitude[i] = std::abs(spectrum[i]);
        }
        
        spectrogram.push_back(std::move(magnitude));
    }
    
    return spectrogram;
}

std::vector<spectral_peak_t> SpectralPeakExtractor::findLocalMaxima(
    const std::vector<std::vector<float>>& spectrogram,
    double sample_rate,
    uint32_t hop_length) {
    
    std::vector<spectral_peak_t> peaks;
    
    if (spectrogram.empty()) return peaks;
    
    uint32_t num_frames = spectrogram.size();
    uint32_t num_bins = spectrogram[0].size();
    
    for (uint32_t frame = 1; frame < num_frames - 1; frame++) {
        std::vector<std::pair<float, uint32_t>> frame_peaks;
        
        for (uint32_t bin = 1; bin < num_bins - 1; bin++) {
            if (isPeak(spectrogram, frame, bin)) {
                float magnitude = spectrogram[frame][bin];
                
                // Check threshold
                float magnitude_db = 20.0f * log10f(std::max(magnitude, 1e-10f));
                if (magnitude_db > config_.peak_threshold_db) {
                    frame_peaks.emplace_back(magnitude, bin);
                }
            }
        }
        
        // Sort peaks by magnitude and take top N
        std::sort(frame_peaks.begin(), frame_peaks.end(), std::greater<>());
        
        uint32_t num_peaks_to_take = std::min(static_cast<uint32_t>(frame_peaks.size()), 
                                            config_.num_peaks_per_frame);
        
        for (uint32_t i = 0; i < num_peaks_to_take; i++) {
            spectral_peak_t peak;
            peak.frequency_bin = frame_peaks[i].second;
            peak.time_frame = frame;
            peak.magnitude = frame_peaks[i].first;
            peak.frequency_hz = (peak.frequency_bin * sample_rate) / (2.0 * num_bins);
            peak.timestamp_seconds = (frame * hop_length) / sample_rate;
            
            peaks.push_back(peak);
        }
    }
    
    return peaks;
}

bool SpectralPeakExtractor::isPeak(
    const std::vector<std::vector<float>>& spectrogram,
    uint32_t time_frame,
    uint32_t freq_bin) {
    
    float center_value = spectrogram[time_frame][freq_bin];
    uint32_t neighborhood = config_.peak_neighborhood_size;
    
    // Check neighborhood in time and frequency
    for (int32_t dt = -static_cast<int32_t>(neighborhood); 
         dt <= static_cast<int32_t>(neighborhood); dt++) {
        for (int32_t df = -static_cast<int32_t>(neighborhood); 
             df <= static_cast<int32_t>(neighborhood); df++) {
            
            if (dt == 0 && df == 0) continue; // Skip center point
            
            int32_t t = static_cast<int32_t>(time_frame) + dt;
            int32_t f = static_cast<int32_t>(freq_bin) + df;
            
            if (t >= 0 && t < static_cast<int32_t>(spectrogram.size()) &&
                f >= 0 && f < static_cast<int32_t>(spectrogram[0].size())) {
                
                if (spectrogram[t][f] >= center_value) {
                    return false; // Not a local maximum
                }
            }
        }
    }
    
    return true;
}

void SpectralPeakExtractor::filterPeaks(std::vector<spectral_peak_t>& peaks) {
    // Remove peaks that are too close in time and frequency
    std::sort(peaks.begin(), peaks.end(), [](const spectral_peak_t& a, const spectral_peak_t& b) {
        return a.magnitude > b.magnitude;
    });
    
    std::vector<spectral_peak_t> filtered_peaks;
    const double min_time_separation = 0.01; // 10ms
    const double min_freq_separation = 100.0; // 100Hz
    
    for (const auto& peak : peaks) {
        bool too_close = false;
        
        for (const auto& existing_peak : filtered_peaks) {
            double time_diff = std::abs(peak.timestamp_seconds - existing_peak.timestamp_seconds);
            double freq_diff = std::abs(peak.frequency_hz - existing_peak.frequency_hz);
            
            if (time_diff < min_time_separation && freq_diff < min_freq_separation) {
                too_close = true;
                break;
            }
        }
        
        if (!too_close) {
            filtered_peaks.push_back(peak);
        }
    }
    
    peaks = std::move(filtered_peaks);
}

// PerceptualHashGenerator implementation
PerceptualHashGenerator::PerceptualHashGenerator(const fingerprint_config_t& config) 
    : config_(config) {}

PerceptualHashGenerator::~PerceptualHashGenerator() = default;

std::vector<uint64_t> PerceptualHashGenerator::generateHashes(
    const std::vector<spectral_peak_t>& peaks) {
    
    return generateConstellationHashes(peaks);
}

std::vector<uint64_t> PerceptualHashGenerator::generateConstellationHashes(
    const std::vector<spectral_peak_t>& peaks) {
    
    std::vector<uint64_t> hashes;
    
    // Generate hashes from peak pairs
    for (size_t i = 0; i < peaks.size(); i++) {
        for (size_t j = i + 1; j < peaks.size(); j++) {
            const auto& peak1 = peaks[i];
            const auto& peak2 = peaks[j];
            
            // Check time and frequency deltas
            uint32_t time_delta = std::abs(static_cast<int32_t>(peak2.time_frame) - 
                                        static_cast<int32_t>(peak1.time_frame));
            uint32_t freq_delta = std::abs(static_cast<int32_t>(peak2.frequency_bin) - 
                                        static_cast<int32_t>(peak1.frequency_bin));
            
            if (time_delta <= config_.hash_time_delta && freq_delta <= config_.hash_freq_delta) {
                uint64_t hash = generatePairHash(peak1, peak2);
                hashes.push_back(hash);
            }
        }
    }
    
    return hashes;
}

uint64_t PerceptualHashGenerator::generatePairHash(
    const spectral_peak_t& peak1,
    const spectral_peak_t& peak2) {
    
    // Create hash from frequency bins and time delta
    uint32_t freq1 = peak1.frequency_bin;
    uint32_t freq2 = peak2.frequency_bin;
    uint32_t time_delta = std::abs(static_cast<int32_t>(peak2.time_frame) - 
                                 static_cast<int32_t>(peak1.time_frame));
    
    return combineFrequencyTime(freq1, freq2, time_delta);
}

uint64_t PerceptualHashGenerator::combineFrequencyTime(
    uint32_t freq1, uint32_t freq2,
    uint32_t time_delta) {
    
    // Combine frequencies and time delta into a single hash
    uint64_t hash = 0;
    
    hash |= (static_cast<uint64_t>(freq1 & 0xFFFF) << 48);
    hash |= (static_cast<uint64_t>(freq2 & 0xFFFF) << 32);
    hash |= (static_cast<uint64_t>(time_delta & 0xFFFF) << 16);
    
    return hash;
}

double PerceptualHashGenerator::computeHashSimilarity(
    const std::vector<uint64_t>& hash1,
    const std::vector<uint64_t>& hash2) {
    
    if (hash1.empty() || hash2.empty()) {
        return 0.0;
    }
    
    // Convert to sets for efficient intersection
    std::unordered_set<uint64_t> set1(hash1.begin(), hash1.end());
    std::unordered_set<uint64_t> set2(hash2.begin(), hash2.end());
    
    // Count intersection
    uint32_t intersection_count = 0;
    for (const auto& hash : set1) {
        if (set2.count(hash) > 0) {
            intersection_count++;
        }
    }
    
    // Jaccard similarity
    uint32_t union_count = set1.size() + set2.size() - intersection_count;
    return union_count > 0 ? static_cast<double>(intersection_count) / union_count : 0.0;
}

// LSHManager implementation
LSHManager::LSHManager(const fingerprint_config_t& config) 
    : config_(config) {
    
    initializeLSHTables();
}

LSHManager::~LSHManager() = default;

void LSHManager::initializeLSHTables() {
    lsh_tables_.resize(config_.lsh_num_tables);
    random_projections_.resize(config_.lsh_num_tables);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (uint32_t i = 0; i < config_.lsh_num_tables; i++) {
        random_projections_[i].resize(config_.lsh_hash_length);
        for (uint32_t j = 0; j < config_.lsh_hash_length; j++) {
            random_projections_[i][j] = dist(gen);
        }
    }
}

std::vector<uint32_t> LSHManager::computeLSHHashes(const std::vector<uint64_t>& fingerprint_hashes) {
    std::vector<uint32_t> lsh_hashes(config_.lsh_num_tables);
    
    for (uint32_t table = 0; table < config_.lsh_num_tables; table++) {
        lsh_hashes[table] = computeSingleLSHHash(fingerprint_hashes, table);
    }
    
    return lsh_hashes;
}

uint32_t LSHManager::computeSingleLSHHash(
    const std::vector<uint64_t>& hashes,
    uint32_t table_index) {
    
    uint32_t lsh_hash = 0;
    
    for (size_t i = 0; i < hashes.size() && i < config_.lsh_hash_length; i++) {
        uint64_t hash = hashes[i];
        uint32_t projection = random_projections_[table_index][i];
        
        // Simple bit mixing
        uint32_t mixed = static_cast<uint32_t>(hash ^ projection);
        lsh_hash ^= mixed;
    }
    
    return lsh_hash;
}

void LSHManager::addFingerprint(
    uint64_t fingerprint_id,
    const std::vector<uint64_t>& hashes) {
    
    auto lsh_hashes = computeLSHHashes(hashes);
    
    for (uint32_t table = 0; table < config_.lsh_num_tables; table++) {
        uint32_t lsh_hash = lsh_hashes[table];
        lsh_tables_[table][lsh_hash].push_back(fingerprint_id);
    }
}

std::vector<uint64_t> LSHManager::querySimilar(
    const std::vector<uint64_t>& query_hashes,
    double similarity_threshold) {
    
    std::unordered_map<uint64_t, uint32_t> candidate_counts;
    auto lsh_hashes = computeLSHHashes(query_hashes);
    
    // Query each LSH table
    for (uint32_t table = 0; table < config_.lsh_num_tables; table++) {
        uint32_t lsh_hash = lsh_hashes[table];
        
        if (lsh_tables_[table].count(lsh_hash) > 0) {
            for (uint64_t candidate_id : lsh_tables_[table][lsh_hash]) {
                candidate_counts[candidate_id]++;
            }
        }
    }
    
    // Filter candidates by minimum threshold
    std::vector<uint64_t> results;
    uint32_t min_votes = static_cast<uint32_t>(config_.lsh_num_tables * similarity_threshold);
    
    for (const auto& candidate : candidate_counts) {
        if (candidate.second >= min_votes) {
            results.push_back(candidate.first);
        }
    }
    
    return results;
}

void LSHManager::clearTables() {
    for (auto& table : lsh_tables_) {
        table.clear();
    }
}

size_t LSHManager::getTableSize() const {
    size_t total_size = 0;
    for (const auto& table : lsh_tables_) {
        total_size += table.size();
    }
    return total_size;
}

// FingerprintDatabase implementation
FingerprintDatabase::FingerprintDatabase(const fingerprint_config_t& config) 
    : config_(config), db_connection_(nullptr), cache_hits_(0), cache_misses_(0) {
    
    lsh_manager_ = std::make_unique<LSHManager>(config_);
}

FingerprintDatabase::~FingerprintDatabase() = default;

} // namespace AudioFingerprinting

void fingerprint_free(audio_fingerprint_t* fingerprint) {
    if (!fingerprint) return;
    
    if (fingerprint->hash_values) {
        free(fingerprint->hash_values);
        fingerprint->hash_values = nullptr;
    }
    
    if (fingerprint->peaks) {
        free(fingerprint->peaks);
        fingerprint->peaks = nullptr;
    }
    
    if (fingerprint->source_filename) {
        free(fingerprint->source_filename);
        fingerprint->source_filename = nullptr;
    }
    
    fingerprint->num_hashes = 0;
    fingerprint->num_peaks = 0;
}

const char* fingerprint_get_error_string(int32_t error_code) {
    switch (error_code) {
        case FINGERPRINT_SUCCESS: return "Success";
        case FINGERPRINT_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case FINGERPRINT_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case FINGERPRINT_ERROR_DATABASE_ERROR: return "Database error";
        case FINGERPRINT_ERROR_NO_PEAKS_FOUND: return "No spectral peaks found";
        case FINGERPRINT_ERROR_HASH_GENERATION_FAILED: return "Hash generation failed";
        case FINGERPRINT_ERROR_MATCH_NOT_FOUND: return "No match found";
        default: return "Unknown error";
    }
}