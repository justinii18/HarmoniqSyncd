#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

// Fingerprint configuration
typedef struct {
    // Spectral analysis parameters
    uint32_t fft_size;                      // FFT size for analysis
    uint32_t hop_length;                    // Hop length between frames
    uint32_t num_mel_filters;               // Number of mel filters
    double sample_rate;                     // Target sample rate
    
    // Peak extraction parameters
    uint32_t num_peaks_per_frame;           // Number of peaks to extract per frame
    double peak_threshold_db;               // Minimum peak threshold in dB
    uint32_t peak_neighborhood_size;        // Peak neighborhood for local maxima
    bool enable_peak_filtering;             // Enable spectral peak filtering
    
    // Hash generation parameters
    uint32_t hash_time_delta;               // Time delta for hash pairs (frames)
    uint32_t hash_freq_delta;               // Frequency delta for hash pairs (bins)
    uint32_t fingerprint_duration_frames;   // Duration of fingerprint in frames
    bool enable_perceptual_hashing;         // Enable perceptual hashing
    
    // LSH (Locality Sensitive Hashing) parameters
    uint32_t lsh_num_tables;                // Number of LSH hash tables
    uint32_t lsh_hash_length;               // Length of each LSH hash
    uint32_t lsh_key_length;                // Length of LSH keys
    double lsh_similarity_threshold;        // Similarity threshold for LSH matching
    
    // Database parameters
    const char* database_path;              // Path to SQLite database file
    uint32_t cache_size;                    // Size of fingerprint cache
    bool enable_database_compression;      // Enable fingerprint compression
    uint32_t max_database_entries;          // Maximum database entries
} fingerprint_config_t;

// Spectral peak structure
typedef struct {
    uint32_t frequency_bin;                 // Frequency bin index
    uint32_t time_frame;                    // Time frame index
    float magnitude;                        // Peak magnitude
    float frequency_hz;                     // Actual frequency in Hz
    double timestamp_seconds;               // Timestamp in seconds
} spectral_peak_t;

// Audio fingerprint structure
typedef struct {
    uint64_t* hash_values;                  // Array of hash values
    uint32_t num_hashes;                    // Number of hash values
    
    spectral_peak_t* peaks;                 // Array of spectral peaks
    uint32_t num_peaks;                     // Number of peaks
    
    double duration_seconds;                // Duration of fingerprinted audio
    double sample_rate;                     // Sample rate of source audio
    uint64_t fingerprint_id;                // Unique fingerprint ID
    
    // Metadata
    char* source_filename;                  // Source file name
    uint64_t file_hash;                     // Hash of source file
    double creation_timestamp;              // Creation timestamp
} audio_fingerprint_t;

// Fingerprint match result
typedef struct {
    uint64_t fingerprint_id;                // Matched fingerprint ID
    double similarity_score;                // Similarity score (0.0-1.0)
    double time_offset_seconds;             // Time offset of match
    uint32_t num_matching_hashes;           // Number of matching hashes
    
    // Match quality metrics
    double confidence_score;                // Match confidence
    bool is_reliable_match;                 // Whether match is reliable
    double snr_db;                          // Signal-to-noise ratio
    
    char* matched_filename;                 // Filename of matched fingerprint
    double matched_duration;                // Duration of matched audio
} fingerprint_match_t;

// Fingerprint database context
typedef struct fingerprint_db fingerprint_db_t;

// Error codes
#define FINGERPRINT_SUCCESS 0
#define FINGERPRINT_ERROR_INVALID_PARAMS -1
#define FINGERPRINT_ERROR_MEMORY_ALLOCATION -2
#define FINGERPRINT_ERROR_DATABASE_ERROR -3
#define FINGERPRINT_ERROR_NO_PEAKS_FOUND -4
#define FINGERPRINT_ERROR_HASH_GENERATION_FAILED -5
#define FINGERPRINT_ERROR_MATCH_NOT_FOUND -6

// Database management
int32_t fingerprint_db_create(const fingerprint_config_t* config, fingerprint_db_t** db);
void fingerprint_db_destroy(fingerprint_db_t* db);

// Fingerprint generation
int32_t fingerprint_generate(
    fingerprint_db_t* db,
    const float* audio_samples,
    uint32_t num_samples,
    double sample_rate,
    const char* source_filename,
    audio_fingerprint_t* fingerprint
);

// Fingerprint storage
int32_t fingerprint_store(
    fingerprint_db_t* db,
    const audio_fingerprint_t* fingerprint
);

// Fingerprint matching
int32_t fingerprint_match(
    fingerprint_db_t* db,
    const audio_fingerprint_t* query_fingerprint,
    fingerprint_match_t** matches,
    uint32_t* num_matches,
    uint32_t max_matches
);

// Fast lookup using LSH
int32_t fingerprint_fast_lookup(
    fingerprint_db_t* db,
    const uint64_t* query_hashes,
    uint32_t num_query_hashes,
    fingerprint_match_t** matches,
    uint32_t* num_matches
);

// Utility functions
fingerprint_config_t fingerprint_get_default_config(void);
void fingerprint_free(audio_fingerprint_t* fingerprint);
void fingerprint_free_matches(fingerprint_match_t* matches, uint32_t num_matches);
const char* fingerprint_get_error_string(int32_t error_code);

// Database statistics
typedef struct {
    uint32_t total_fingerprints;            // Total fingerprints in database
    uint32_t total_hashes;                  // Total hash values stored
    uint64_t database_size_bytes;           // Database size in bytes
    double avg_fingerprint_duration;       // Average fingerprint duration
    uint32_t cache_hits;                    // Cache hits
    uint32_t cache_misses;                  // Cache misses
    double cache_hit_ratio;                 // Cache hit ratio
} fingerprint_db_stats_t;

fingerprint_db_stats_t fingerprint_db_get_stats(fingerprint_db_t* db);
void fingerprint_db_optimize(fingerprint_db_t* db);

#ifdef __cplusplus
}

// C++ Interface for Audio Fingerprinting
namespace AudioFingerprinting {

// Spectral peak extractor
class SpectralPeakExtractor {
public:
    explicit SpectralPeakExtractor(const fingerprint_config_t& config);
    ~SpectralPeakExtractor();
    
    // Extract spectral peaks from audio
    std::vector<spectral_peak_t> extractPeaks(
        const std::vector<float>& audio,
        double sample_rate
    );
    
    // Extract peaks from spectrogram
    std::vector<spectral_peak_t> extractPeaksFromSpectrogram(
        const std::vector<std::vector<float>>& spectrogram,
        double sample_rate,
        uint32_t hop_length
    );
    
    // Configuration
    void updateConfig(const fingerprint_config_t& config) { config_ = config; }
    fingerprint_config_t getConfig() const { return config_; }
    
private:
    fingerprint_config_t config_;
    void* fft_processor_; // Opaque pointer to avoid forward declaration issues
    
    // Internal processing methods
    std::vector<std::vector<float>> computeSpectrogram(
        const std::vector<float>& audio,
        double sample_rate
    );
    
    std::vector<spectral_peak_t> findLocalMaxima(
        const std::vector<std::vector<float>>& spectrogram,
        double sample_rate,
        uint32_t hop_length
    );
    
    bool isPeak(
        const std::vector<std::vector<float>>& spectrogram,
        uint32_t time_frame,
        uint32_t freq_bin
    );
    
    void filterPeaks(std::vector<spectral_peak_t>& peaks);
};

// Perceptual hash generator
class PerceptualHashGenerator {
public:
    explicit PerceptualHashGenerator(const fingerprint_config_t& config);
    ~PerceptualHashGenerator();
    
    // Generate hashes from spectral peaks
    std::vector<uint64_t> generateHashes(
        const std::vector<spectral_peak_t>& peaks
    );
    
    // Generate perceptual hash from audio
    std::vector<uint64_t> generatePerceptualHash(
        const std::vector<float>& audio,
        double sample_rate
    );
    
    // Hash similarity calculation
    double computeHashSimilarity(
        const std::vector<uint64_t>& hash1,
        const std::vector<uint64_t>& hash2
    );
    
private:
    fingerprint_config_t config_;
    
    // Hash generation methods
    uint64_t generatePairHash(
        const spectral_peak_t& peak1,
        const spectral_peak_t& peak2
    );
    
    std::vector<uint64_t> generateConstellationHashes(
        const std::vector<spectral_peak_t>& peaks
    );
    
    uint64_t combineFrequencyTime(
        uint32_t freq1, uint32_t freq2,
        uint32_t time_delta
    );
};

// LSH (Locality Sensitive Hashing) manager
class LSHManager {
public:
    explicit LSHManager(const fingerprint_config_t& config);
    ~LSHManager();
    
    // LSH operations
    std::vector<uint32_t> computeLSHHashes(const std::vector<uint64_t>& fingerprint_hashes);
    
    // Add fingerprint to LSH tables
    void addFingerprint(
        uint64_t fingerprint_id,
        const std::vector<uint64_t>& hashes
    );
    
    // Query LSH tables for similar fingerprints
    std::vector<uint64_t> querySimilar(
        const std::vector<uint64_t>& query_hashes,
        double similarity_threshold = 0.7
    );
    
    // LSH table management
    void clearTables();
    size_t getTableSize() const;
    
private:
    fingerprint_config_t config_;
    
    // LSH hash tables
    std::vector<std::unordered_map<uint32_t, std::vector<uint64_t>>> lsh_tables_;
    std::vector<std::vector<uint32_t>> random_projections_;
    
    void initializeLSHTables();
    uint32_t computeSingleLSHHash(
        const std::vector<uint64_t>& hashes,
        uint32_t table_index
    );
};

// Fingerprint database manager
class FingerprintDatabase {
public:
    explicit FingerprintDatabase(const fingerprint_config_t& config);
    ~FingerprintDatabase();
    
    // Database operations
    bool initialize();
    bool store(const audio_fingerprint_t& fingerprint);
    std::vector<fingerprint_match_t> search(
        const audio_fingerprint_t& query,
        uint32_t max_results = 10
    );
    
    // Fast lookup operations
    std::vector<fingerprint_match_t> fastLookup(
        const std::vector<uint64_t>& query_hashes,
        uint32_t max_results = 10
    );
    
    // Database management
    bool optimize();
    bool vacuum();
    fingerprint_db_stats_t getStatistics();
    
    // Cache management
    void clearCache();
    double getCacheHitRatio() const;
    
private:
    fingerprint_config_t config_;
    void* db_connection_; // Opaque pointer to avoid forward declaration issues  
    std::unique_ptr<LSHManager> lsh_manager_;
    
    // Cache for frequently accessed fingerprints
    std::unordered_map<uint64_t, audio_fingerprint_t> fingerprint_cache_;
    mutable std::mutex cache_mutex_;
    uint32_t cache_hits_;
    uint32_t cache_misses_;
    
    // Database schema management
    bool createTables();
    bool createIndices();
    
    // Internal query methods
    std::vector<uint64_t> findCandidateFingerprints(
        const std::vector<uint64_t>& query_hashes
    );
    
    double computeFingerprintSimilarity(
        const audio_fingerprint_t& fp1,
        const audio_fingerprint_t& fp2
    );
    
    bool loadFingerprintFromCache(uint64_t fingerprint_id, audio_fingerprint_t& fingerprint);
    void storeFingerprintInCache(const audio_fingerprint_t& fingerprint);
};

// Complete fingerprinting system
class AudioFingerprintingSystem {
public:
    explicit AudioFingerprintingSystem(const fingerprint_config_t& config = fingerprint_get_default_config());
    ~AudioFingerprintingSystem();
    
    // High-level fingerprinting operations
    audio_fingerprint_t generateFingerprint(
        const std::vector<float>& audio,
        double sample_rate,
        const std::string& source_filename = ""
    );
    
    bool storeFingerprint(const audio_fingerprint_t& fingerprint);
    
    std::vector<fingerprint_match_t> findMatches(
        const std::vector<float>& query_audio,
        double sample_rate,
        uint32_t max_results = 10
    );
    
    // Batch operations
    std::vector<audio_fingerprint_t> generateFingerprintsFromFiles(
        const std::vector<std::string>& audio_files
    );
    
    bool buildDatabase(const std::vector<std::string>& audio_files);
    
    // Database management
    fingerprint_db_stats_t getDatabaseStatistics();
    bool optimizeDatabase();
    
    // Configuration
    void updateConfig(const fingerprint_config_t& config);
    fingerprint_config_t getConfig() const { return config_; }
    
private:
    fingerprint_config_t config_;
    std::unique_ptr<SpectralPeakExtractor> peak_extractor_;
    std::unique_ptr<PerceptualHashGenerator> hash_generator_;
    std::unique_ptr<FingerprintDatabase> database_;
    
    // Internal processing
    void validateConfig();
    audio_fingerprint_t createFingerprintFromPeaksAndHashes(
        const std::vector<spectral_peak_t>& peaks,
        const std::vector<uint64_t>& hashes,
        double duration,
        double sample_rate,
        const std::string& filename
    );
};

// Utility classes
class FingerprintMatcher {
public:
    // Match fingerprints with various algorithms
    static double computeJaccardSimilarity(
        const std::vector<uint64_t>& hashes1,
        const std::vector<uint64_t>& hashes2
    );
    
    static double computeHammingSimilarity(
        const std::vector<uint64_t>& hashes1,
        const std::vector<uint64_t>& hashes2
    );
    
    static double computeCosineSimilarity(
        const std::vector<uint64_t>& hashes1,
        const std::vector<uint64_t>& hashes2
    );
    
    // Time-offset detection
    static double findBestTimeOffset(
        const std::vector<spectral_peak_t>& peaks1,
        const std::vector<spectral_peak_t>& peaks2
    );
    
    // Match quality assessment
    static bool isReliableMatch(
        const fingerprint_match_t& match,
        double min_similarity = 0.7,
        uint32_t min_matching_hashes = 10
    );
};

} // namespace AudioFingerprinting

#endif