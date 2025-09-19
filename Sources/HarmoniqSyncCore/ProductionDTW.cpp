#include "include/ProductionDTW.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <numeric>
#include <limits>

// Internal context structure
struct dtw_context {
    dtw_config_t config;
    bool is_initialized;
    
    // Feature extraction components
    production_fft_context_t* fft_context;
    std::vector<std::vector<float>> mel_filter_bank;
    
    // Memory pools for efficiency
    std::vector<std::vector<float>> distance_matrix_pool;
    std::vector<float> feature_buffer;
    
    dtw_context() : fft_context(nullptr), is_initialized(false) {}
};

namespace {
    const double PI = 3.14159265358979323846;
    const float MIN_FLOAT = std::numeric_limits<float>::min();
    const float MAX_FLOAT = std::numeric_limits<float>::max();
    
    // Mel scale conversion functions
    double hz_to_mel(double hz) {
        return 2595.0 * log10(1.0 + hz / 700.0);
    }
    
    double mel_to_hz(double mel) {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    }
    
    // DCT for MFCC computation
    std::vector<float> compute_dct(const std::vector<float>& input, uint32_t num_coeffs) {
        std::vector<float> dct_result(num_coeffs);
        uint32_t N = input.size();
        
        for (uint32_t k = 0; k < num_coeffs; k++) {
            float sum = 0.0f;
            for (uint32_t n = 0; n < N; n++) {
                sum += input[n] * cosf(PI * k * (n + 0.5f) / N);
            }
            dct_result[k] = sum;
        }
        
        return dct_result;
    }
    
    // Compute distance between feature vectors
    float compute_feature_distance(
        const float* feature1,
        const float* feature2,
        uint32_t feature_dim,
        dtw_distance_metric_t metric) {
        
        switch (metric) {
            case DTW_DISTANCE_EUCLIDEAN: {
                float sum = 0.0f;
                for (uint32_t i = 0; i < feature_dim; i++) {
                    float diff = feature1[i] - feature2[i];
                    sum += diff * diff;
                }
                return sqrtf(sum);
            }
            
            case DTW_DISTANCE_MANHATTAN: {
                float sum = 0.0f;
                for (uint32_t i = 0; i < feature_dim; i++) {
                    sum += fabsf(feature1[i] - feature2[i]);
                }
                return sum;
            }
            
            case DTW_DISTANCE_COSINE: {
                float dot_product = 0.0f;
                float norm1 = 0.0f;
                float norm2 = 0.0f;
                
                for (uint32_t i = 0; i < feature_dim; i++) {
                    dot_product += feature1[i] * feature2[i];
                    norm1 += feature1[i] * feature1[i];
                    norm2 += feature2[i] * feature2[i];
                }
                
                if (norm1 < MIN_FLOAT || norm2 < MIN_FLOAT) {
                    return 1.0f; // Maximum distance
                }
                
                float cosine_sim = dot_product / (sqrtf(norm1) * sqrtf(norm2));
                return 1.0f - cosine_sim; // Convert similarity to distance
            }
            
            case DTW_DISTANCE_CORRELATION: {
                // Compute correlation coefficient and convert to distance
                float mean1 = 0.0f, mean2 = 0.0f;
                for (uint32_t i = 0; i < feature_dim; i++) {
                    mean1 += feature1[i];
                    mean2 += feature2[i];
                }
                mean1 /= feature_dim;
                mean2 /= feature_dim;
                
                float numerator = 0.0f;
                float sum_sq1 = 0.0f, sum_sq2 = 0.0f;
                
                for (uint32_t i = 0; i < feature_dim; i++) {
                    float diff1 = feature1[i] - mean1;
                    float diff2 = feature2[i] - mean2;
                    numerator += diff1 * diff2;
                    sum_sq1 += diff1 * diff1;
                    sum_sq2 += diff2 * diff2;
                }
                
                if (sum_sq1 < MIN_FLOAT || sum_sq2 < MIN_FLOAT) {
                    return 1.0f;
                }
                
                float correlation = numerator / sqrtf(sum_sq1 * sum_sq2);
                return 1.0f - correlation;
            }
            
            default:
                return compute_feature_distance(feature1, feature2, feature_dim, DTW_DISTANCE_EUCLIDEAN);
        }
    }
    
    // Check if a step is valid according to the step pattern
    bool is_valid_step(
        uint32_t ref_prev, uint32_t target_prev,
        uint32_t ref_curr, uint32_t target_curr,
        dtw_step_pattern_t pattern) {
        
        int32_t ref_step = static_cast<int32_t>(ref_curr) - static_cast<int32_t>(ref_prev);
        int32_t target_step = static_cast<int32_t>(target_curr) - static_cast<int32_t>(target_prev);
        
        switch (pattern) {
            case DTW_STEP_SYMMETRIC:
                // Allow (1,0), (0,1), (1,1)
                return (ref_step == 1 && target_step == 0) ||
                       (ref_step == 0 && target_step == 1) ||
                       (ref_step == 1 && target_step == 1);
                       
            case DTW_STEP_ASYMMETRIC:
                // Allow more flexible steps: (1,0), (0,1), (1,1), (2,1), (1,2)
                return (ref_step >= 0 && ref_step <= 2) &&
                       (target_step >= 0 && target_step <= 2) &&
                       (ref_step + target_step > 0) &&
                       (ref_step + target_step <= 3);
                       
            case DTW_STEP_ITAKURA:
                // Itakura parallelogram constraints
                return (ref_step == 1 && target_step == 0) ||
                       (ref_step == 0 && target_step == 1) ||
                       (ref_step == 1 && target_step == 1);
                       
            case DTW_STEP_SAKOE_CHIBA:
            default:
                // Standard symmetric pattern
                return (ref_step == 1 && target_step == 0) ||
                       (ref_step == 0 && target_step == 1) ||
                       (ref_step == 1 && target_step == 1);
        }
    }
}

// C Interface Implementation
int32_t dtw_create_context(const dtw_config_t* config, dtw_context_t** context) {
    if (!config || !context) {
        return DTW_ERROR_INVALID_PARAMS;
    }
    
    auto* ctx = new dtw_context();
    if (!ctx) {
        return DTW_ERROR_MEMORY_ALLOCATION;
    }
    
    ctx->config = *config;
    
    // Create FFT context for feature extraction
    fft_config_t fft_config = fft_get_default_config(config->feature_window_size);
    int32_t result = fft_create_context(&fft_config, &ctx->fft_context);
    if (result != FFT_SUCCESS) {
        delete ctx;
        return result;
    }
    
    // Initialize mel filter bank if needed
    if (config->feature_type == DTW_FEATURE_MFCC) {
        // Create mel filter bank (simplified implementation)
        uint32_t num_filters = 26; // Standard number of mel filters
        uint32_t fft_size = config->feature_window_size;
        ctx->mel_filter_bank.resize(num_filters);
        
        // Simple triangular mel filters
        for (uint32_t i = 0; i < num_filters; i++) {
            ctx->mel_filter_bank[i].resize(fft_size / 2 + 1, 0.0f);
            // Simplified mel filter generation (would be more complex in practice)
            uint32_t start = i * (fft_size / 2) / num_filters;
            uint32_t peak = (i + 1) * (fft_size / 2) / num_filters;
            uint32_t end = (i + 2) * (fft_size / 2) / num_filters;
            
            for (uint32_t j = start; j < peak && j < fft_size / 2 + 1; j++) {
                ctx->mel_filter_bank[i][j] = static_cast<float>(j - start) / (peak - start);
            }
            for (uint32_t j = peak; j < end && j < fft_size / 2 + 1; j++) {
                ctx->mel_filter_bank[i][j] = static_cast<float>(end - j) / (end - peak);
            }
        }
    }
    
    ctx->is_initialized = true;
    *context = ctx;
    
    return DTW_SUCCESS;
}

void dtw_destroy_context(dtw_context_t* context) {
    if (!context) return;
    
    if (context->fft_context) {
        fft_destroy_context(context->fft_context);
    }
    
    delete context;
}

dtw_config_t dtw_get_default_config(void) {
    dtw_config_t config;
    
    config.step_pattern = DTW_STEP_SYMMETRIC;
    config.feature_type = DTW_FEATURE_MFCC;
    config.distance_metric = DTW_DISTANCE_EUCLIDEAN;
    
    config.max_warp_samples = 1000;
    config.constraint_radius = 0.1;
    config.min_path_length = 10;
    config.enable_slope_constraint = true;
    
    config.feature_window_size = 2048;
    config.feature_hop_length = 512;
    config.num_mfcc_coeffs = 13;
    config.num_chroma_bins = 12;
    
    config.enable_early_termination = true;
    config.early_termination_threshold = 10.0;
    config.enable_diagonal_optimization = true;
    
    config.max_sequence_length = 100000;
    config.enable_banded_computation = true;
    
    return config;
}

// C++ Interface Implementation
namespace ProductionDTW {

DTWAligner::DTWAligner(const dtw_config_t& config) 
    : context_(nullptr), config_(config) {
    
    int32_t result = dtw_create_context(&config_, &context_);
    if (result != DTW_SUCCESS) {
        throw std::runtime_error("Failed to create DTW context: " + 
                               std::string(dtw_get_error_string(result)));
    }
    
    // Create FFT processor for feature extraction
    fft_config_t fft_config = fft_get_default_config(config_.feature_window_size);
    fft_processor_ = std::make_unique<ProductionFFT::FFTProcessor>(fft_config);
}

DTWAligner::~DTWAligner() {
    if (context_) {
        dtw_destroy_context(context_);
    }
}

dtw_result_t DTWAligner::align(
    const std::vector<float>& reference,
    const std::vector<float>& target,
    double sample_rate,
    double initial_offset) {
    
    dtw_result_t result = {};
    
    try {
        // Extract features based on configuration
        feature_matrix_t ref_features, target_features;
        
        switch (config_.feature_type) {
            case DTW_FEATURE_MFCC:
                ref_features = extractMFCC(reference, sample_rate, config_.num_mfcc_coeffs);
                target_features = extractMFCC(target, sample_rate, config_.num_mfcc_coeffs);
                break;
                
            case DTW_FEATURE_CHROMA:
                ref_features = extractChroma(reference, sample_rate, config_.num_chroma_bins);
                target_features = extractChroma(target, sample_rate, config_.num_chroma_bins);
                break;
                
            case DTW_FEATURE_SPECTRAL:
                ref_features = extractSpectralFeatures(reference, sample_rate);
                target_features = extractSpectralFeatures(target, sample_rate);
                break;
                
            case DTW_FEATURE_RAW:
            default:
                // Use downsampled raw audio as features
                uint32_t downsample_factor = config_.feature_hop_length;
                ref_features.num_frames = reference.size() / downsample_factor;
                ref_features.num_features = 1;
                ref_features.features = new float[ref_features.num_frames];
                
                for (uint32_t i = 0; i < ref_features.num_frames; i++) {
                    ref_features.features[i] = reference[i * downsample_factor];
                }
                
                target_features.num_frames = target.size() / downsample_factor;
                target_features.num_features = 1;
                target_features.features = new float[target_features.num_frames];
                
                for (uint32_t i = 0; i < target_features.num_frames; i++) {
                    target_features.features[i] = target[i * downsample_factor];
                }
                break;
        }
        
        // Perform DTW alignment
        result = alignFeatureMatrices(ref_features, target_features, initial_offset);
        
        // Cleanup
        dtw_free_features(&ref_features);
        dtw_free_features(&target_features);
        
    } catch (...) {
        result.error_code = DTW_ERROR_FEATURE_EXTRACTION_FAILED;
        result.is_reliable = false;
    }
    
    return result;
}

dtw_result_t DTWAligner::alignFeatureMatrices(
    const feature_matrix_t& ref_features,
    const feature_matrix_t& target_features,
    double initial_offset) {
    
    dtw_result_t result = {};
    
    try {
        // Check dimensions
        if (ref_features.num_features != target_features.num_features) {
            result.error_code = DTW_ERROR_INVALID_PARAMS;
            return result;
        }
        
        // Compute distance matrix
        auto distance_matrix = computeDistanceMatrix(ref_features, target_features);
        
        // Find optimal path
        auto path = findOptimalPath(distance_matrix, ref_features.num_frames, target_features.num_frames);
        
        if (path.empty()) {
            result.error_code = DTW_ERROR_NO_VALID_PATH;
            return result;
        }
        
        // Compute total distance
        float total_distance = 0.0f;
        for (const auto& point : path) {
            total_distance += distance_matrix[point.first][point.second];
        }
        
        // Fill result
        result.path_length = path.size();
        result.total_distance = total_distance;
        result.normalized_distance = total_distance / path.size();
        
        // Allocate path arrays
        result.reference_indices = static_cast<uint32_t*>(malloc(path.size() * sizeof(uint32_t)));
        result.target_indices = static_cast<uint32_t*>(malloc(path.size() * sizeof(uint32_t)));
        
        if (!result.reference_indices || !result.target_indices) {
            result.error_code = DTW_ERROR_MEMORY_ALLOCATION;
            return result;
        }
        
        for (size_t i = 0; i < path.size(); i++) {
            result.reference_indices[i] = path[i].first;
            result.target_indices[i] = path[i].second;
        }
        
        // Compute refined offset
        // Simple approach: use the median offset from the path
        std::vector<double> offsets;
        for (size_t i = 0; i < path.size(); i++) {
            double ref_time = path[i].first / ref_features.frame_rate;
            double target_time = path[i].second / target_features.frame_rate;
            offsets.push_back(target_time - ref_time);
        }
        
        std::sort(offsets.begin(), offsets.end());
        result.refined_offset_seconds = offsets[offsets.size() / 2] + initial_offset;
        
        // Compute quality metrics
        result.path_straightness = ProductionDTW::PathAnalyzer::computePathStraightness(path);
        result.diagonal_deviation = ProductionDTW::PathAnalyzer::computeDiagonalDeviation(
            path, ref_features.num_frames, target_features.num_frames);
        result.local_consistency = ProductionDTW::PathAnalyzer::computeLocalConsistency(path);
        
        // Compute confidence based on distance and path quality
        result.confidence_score = std::max(0.0, 1.0 - result.normalized_distance / 10.0) * 
                                 result.path_straightness * result.local_consistency;
        
        // Assess reliability
        assessReliability(result);
        
        result.error_code = DTW_SUCCESS;
        
    } catch (...) {
        result.error_code = DTW_ERROR_MEMORY_ALLOCATION;
        result.is_reliable = false;
    }
    
    return result;
}

std::vector<std::vector<float>> DTWAligner::computeDistanceMatrix(
    const feature_matrix_t& ref_features,
    const feature_matrix_t& target_features) {
    
    uint32_t ref_frames = ref_features.num_frames;
    uint32_t target_frames = target_features.num_frames;
    uint32_t feature_dim = ref_features.num_features;
    
    std::vector<std::vector<float>> distance_matrix(ref_frames, std::vector<float>(target_frames));
    
    for (uint32_t i = 0; i < ref_frames; i++) {
        for (uint32_t j = 0; j < target_frames; j++) {
            const float* ref_feature = &ref_features.features[i * feature_dim];
            const float* target_feature = &target_features.features[j * feature_dim];
            
            distance_matrix[i][j] = computeDistance(ref_feature, target_feature, 
                                                  feature_dim, config_.distance_metric);
        }
    }
    
    return distance_matrix;
}

std::vector<std::pair<uint32_t, uint32_t>> DTWAligner::findOptimalPath(
    const std::vector<std::vector<float>>& distance_matrix,
    uint32_t ref_length,
    uint32_t target_length) {
    
    // Dynamic programming matrix
    std::vector<std::vector<float>> dp(ref_length, std::vector<float>(target_length, MAX_FLOAT));
    
    // Initialize
    dp[0][0] = distance_matrix[0][0];
    
    // Fill first row and column
    for (uint32_t i = 1; i < ref_length; i++) {
        dp[i][0] = dp[i-1][0] + distance_matrix[i][0];
    }
    for (uint32_t j = 1; j < target_length; j++) {
        dp[0][j] = dp[0][j-1] + distance_matrix[0][j];
    }
    
    // Fill the DP matrix
    for (uint32_t i = 1; i < ref_length; i++) {
        for (uint32_t j = 1; j < target_length; j++) {
            // Apply step pattern constraints
            float min_cost = MAX_FLOAT;
            
            // (1,1) step
            min_cost = std::min(min_cost, dp[i-1][j-1]);
            
            // (1,0) step
            min_cost = std::min(min_cost, dp[i-1][j]);
            
            // (0,1) step
            min_cost = std::min(min_cost, dp[i][j-1]);
            
            dp[i][j] = distance_matrix[i][j] + min_cost;
        }
    }
    
    // Backtrack to find optimal path
    std::vector<std::pair<uint32_t, uint32_t>> path;
    uint32_t i = ref_length - 1;
    uint32_t j = target_length - 1;
    
    while (i > 0 || j > 0) {
        path.emplace_back(i, j);
        
        if (i == 0) {
            j--;
        } else if (j == 0) {
            i--;
        } else {
            // Find the predecessor with minimum cost
            float cost_diag = dp[i-1][j-1];
            float cost_up = dp[i-1][j];
            float cost_left = dp[i][j-1];
            
            if (cost_diag <= cost_up && cost_diag <= cost_left) {
                i--; j--;
            } else if (cost_up <= cost_left) {
                i--;
            } else {
                j--;
            }
        }
    }
    
    path.emplace_back(0, 0);
    std::reverse(path.begin(), path.end());
    
    return path;
}

double DTWAligner::computeDistance(
    const float* feature1,
    const float* feature2,
    uint32_t feature_dim,
    dtw_distance_metric_t metric) {
    
    return compute_feature_distance(feature1, feature2, feature_dim, metric);
}

feature_matrix_t DTWAligner::extractMFCC(
    const std::vector<float>& audio,
    double sample_rate,
    uint32_t num_coeffs) {
    
    feature_matrix_t features = {};
    
    // Simple MFCC extraction (simplified for this implementation)
    uint32_t window_size = config_.feature_window_size;
    uint32_t hop_length = config_.feature_hop_length;
    uint32_t num_frames = (audio.size() - window_size) / hop_length + 1;
    
    features.num_frames = num_frames;
    features.num_features = num_coeffs;
    features.sample_rate = sample_rate;
    features.frame_rate = sample_rate / hop_length;
    features.features = new float[num_frames * num_coeffs];
    
    // Extract MFCC for each frame
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        uint32_t start_idx = frame * hop_length;
        std::vector<float> window(audio.begin() + start_idx, 
                                audio.begin() + start_idx + window_size);
        
        // Compute FFT
        auto spectrum = fft_processor_->computeFFT(window);
        
        // Compute magnitude spectrum
        std::vector<float> magnitude(spectrum.size());
        for (size_t i = 0; i < spectrum.size(); i++) {
            magnitude[i] = std::abs(spectrum[i]);
        }
        
        // Apply mel filter bank (simplified)
        std::vector<float> mel_energies(26, 0.0f); // Standard 26 mel filters
        for (size_t i = 0; i < mel_energies.size() && i < magnitude.size(); i++) {
            mel_energies[i] = magnitude[i];
        }
        
        // Take log
        for (auto& energy : mel_energies) {
            energy = logf(std::max(energy, MIN_FLOAT));
        }
        
        // Compute DCT
        auto mfcc = compute_dct(mel_energies, num_coeffs);
        
        // Store MFCC coefficients
        for (uint32_t c = 0; c < num_coeffs; c++) {
            features.features[frame * num_coeffs + c] = mfcc[c];
        }
    }
    
    return features;
}

feature_matrix_t DTWAligner::extractChroma(
    const std::vector<float>& audio,
    double sample_rate,
    uint32_t num_bins) {
    
    feature_matrix_t features = {};
    
    // Simple chroma extraction (simplified implementation)
    uint32_t window_size = config_.feature_window_size;
    uint32_t hop_length = config_.feature_hop_length;
    uint32_t num_frames = (audio.size() - window_size) / hop_length + 1;
    
    features.num_frames = num_frames;
    features.num_features = num_bins;
    features.sample_rate = sample_rate;
    features.frame_rate = sample_rate / hop_length;
    features.features = new float[num_frames * num_bins];
    
    // Extract chroma for each frame
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        uint32_t start_idx = frame * hop_length;
        std::vector<float> window(audio.begin() + start_idx, 
                                audio.begin() + start_idx + window_size);
        
        // Compute FFT
        auto spectrum = fft_processor_->computeFFT(window);
        
        // Compute magnitude spectrum
        std::vector<float> magnitude(spectrum.size());
        for (size_t i = 0; i < spectrum.size(); i++) {
            magnitude[i] = std::abs(spectrum[i]);
        }
        
        // Simple chroma mapping (map frequency bins to chroma bins)
        std::vector<float> chroma(num_bins, 0.0f);
        double freq_per_bin = sample_rate / (2.0 * magnitude.size());
        
        for (size_t i = 1; i < magnitude.size(); i++) { // Skip DC
            double freq = i * freq_per_bin;
            if (freq > 80.0 && freq < 8000.0) { // Focus on musical range
                // Convert frequency to pitch class (simplified)
                double log_freq = log2(freq / 440.0); // A4 = 440 Hz
                double pitch_class = fmod(log_freq * 12.0 + 69.0, 12.0); // MIDI note mod 12
                if (pitch_class < 0) pitch_class += 12.0;
                
                uint32_t chroma_bin = static_cast<uint32_t>(pitch_class);
                if (chroma_bin < num_bins) {
                    chroma[chroma_bin] += magnitude[i];
                }
            }
        }
        
        // Normalize chroma
        float chroma_sum = 0.0f;
        for (float val : chroma) {
            chroma_sum += val;
        }
        if (chroma_sum > 0.0f) {
            for (auto& val : chroma) {
                val /= chroma_sum;
            }
        }
        
        // Store chroma features
        for (uint32_t c = 0; c < num_bins; c++) {
            features.features[frame * num_bins + c] = chroma[c];
        }
    }
    
    return features;
}

feature_matrix_t DTWAligner::extractSpectralFeatures(
    const std::vector<float>& audio,
    double sample_rate) {
    
    feature_matrix_t features = {};
    
    // Extract spectral features: centroid, rolloff, flux, ZCR
    uint32_t window_size = config_.feature_window_size;
    uint32_t hop_length = config_.feature_hop_length;
    uint32_t num_frames = (audio.size() - window_size) / hop_length + 1;
    uint32_t num_spectral_features = 4; // centroid, rolloff, flux, ZCR
    
    features.num_frames = num_frames;
    features.num_features = num_spectral_features;
    features.sample_rate = sample_rate;
    features.frame_rate = sample_rate / hop_length;
    features.features = new float[num_frames * num_spectral_features];
    
    std::vector<float> prev_magnitude; // For spectral flux
    
    for (uint32_t frame = 0; frame < num_frames; frame++) {
        uint32_t start_idx = frame * hop_length;
        std::vector<float> window(audio.begin() + start_idx, 
                                audio.begin() + start_idx + window_size);
        
        // Compute FFT
        auto spectrum = fft_processor_->computeFFT(window);
        
        // Compute magnitude spectrum
        std::vector<float> magnitude(spectrum.size());
        for (size_t i = 0; i < spectrum.size(); i++) {
            magnitude[i] = std::abs(spectrum[i]);
        }
        
        // 1. Spectral Centroid
        float centroid = 0.0f;
        float magnitude_sum = 0.0f;
        double freq_per_bin = sample_rate / (2.0 * magnitude.size());
        
        for (size_t i = 1; i < magnitude.size(); i++) {
            float freq = i * freq_per_bin;
            centroid += freq * magnitude[i];
            magnitude_sum += magnitude[i];
        }
        centroid = magnitude_sum > 0.0f ? centroid / magnitude_sum : 0.0f;
        
        // 2. Spectral Rolloff (85th percentile)
        float rolloff = 0.0f;
        float cumulative_sum = 0.0f;
        float target_sum = magnitude_sum * 0.85f;
        
        for (size_t i = 1; i < magnitude.size() && cumulative_sum < target_sum; i++) {
            cumulative_sum += magnitude[i];
            rolloff = i * freq_per_bin;
        }
        
        // 3. Spectral Flux
        float flux = 0.0f;
        if (!prev_magnitude.empty()) {
            for (size_t i = 0; i < std::min(magnitude.size(), prev_magnitude.size()); i++) {
                float diff = magnitude[i] - prev_magnitude[i];
                flux += std::max(0.0f, diff);
            }
        }
        prev_magnitude = magnitude;
        
        // 4. Zero Crossing Rate
        float zcr = 0.0f;
        for (uint32_t i = 1; i < window_size; i++) {
            if ((window[i] >= 0) != (window[i-1] >= 0)) {
                zcr += 1.0f;
            }
        }
        zcr /= (window_size - 1);
        
        // Store features (normalize to reasonable ranges)
        features.features[frame * num_spectral_features + 0] = centroid / (sample_rate / 2); // Normalize centroid
        features.features[frame * num_spectral_features + 1] = rolloff / (sample_rate / 2);  // Normalize rolloff
        features.features[frame * num_spectral_features + 2] = std::min(flux / magnitude_sum, 1.0f); // Normalize flux
        features.features[frame * num_spectral_features + 3] = zcr; // ZCR already normalized
    }
    
    return features;
}

void DTWAligner::assessReliability(dtw_result_t& result) {
    result.is_reliable = true;
    
    // Check various quality metrics
    if (result.confidence_score < 0.3) {
        result.is_reliable = false;
    }
    
    if (result.path_straightness < 0.5) {
        result.is_reliable = false;
    }
    
    if (result.diagonal_deviation > 0.5) {
        result.is_reliable = false;
    }
    
    if (result.normalized_distance > 5.0) {
        result.is_reliable = false;
    }
}

// PathAnalyzer implementation
double PathAnalyzer::computePathStraightness(
    const std::vector<std::pair<uint32_t, uint32_t>>& path) {
    
    if (path.size() < 3) return 1.0;
    
    // Compute how close the path is to a straight line
    double total_deviation = 0.0;
    double path_length = path.size();
    
    for (size_t i = 1; i < path.size() - 1; i++) {
        // Compute expected position on straight line
        double t = static_cast<double>(i) / (path_length - 1);
        double expected_ref = path[0].first + t * (path.back().first - path[0].first);
        double expected_target = path[0].second + t * (path.back().second - path[0].second);
        
        // Compute deviation
        double deviation = sqrt(pow(path[i].first - expected_ref, 2) + 
                              pow(path[i].second - expected_target, 2));
        total_deviation += deviation;
    }
    
    double avg_deviation = total_deviation / (path_length - 2);
    return std::max(0.0, 1.0 - avg_deviation / sqrt(path_length));
}

double PathAnalyzer::computeDiagonalDeviation(
    const std::vector<std::pair<uint32_t, uint32_t>>& path,
    uint32_t ref_length,
    uint32_t target_length) {
    
    if (path.empty()) return 1.0;
    
    double total_deviation = 0.0;
    double diagonal_slope = static_cast<double>(target_length) / ref_length;
    
    for (const auto& point : path) {
        double expected_target = point.first * diagonal_slope;
        double deviation = fabs(point.second - expected_target);
        total_deviation += deviation;
    }
    
    return total_deviation / (path.size() * target_length);
}

double PathAnalyzer::computeLocalConsistency(
    const std::vector<std::pair<uint32_t, uint32_t>>& path) {
    
    if (path.size() < 3) return 1.0;
    
    double total_consistency = 0.0;
    
    for (size_t i = 1; i < path.size() - 1; i++) {
        // Compute local slopes
        double slope_prev = static_cast<double>(path[i].second - path[i-1].second) / 
                           std::max(1.0, static_cast<double>(path[i].first - path[i-1].first));
        double slope_next = static_cast<double>(path[i+1].second - path[i].second) / 
                           std::max(1.0, static_cast<double>(path[i+1].first - path[i].first));
        
        double consistency = 1.0 / (1.0 + fabs(slope_next - slope_prev));
        total_consistency += consistency;
    }
    
    return total_consistency / (path.size() - 2);
}

} // namespace ProductionDTW

void dtw_free_result(dtw_result_t* result) {
    if (!result) return;
    
    if (result->reference_indices) {
        free(result->reference_indices);
        result->reference_indices = nullptr;
    }
    
    if (result->target_indices) {
        free(result->target_indices);
        result->target_indices = nullptr;
    }
    
    result->path_length = 0;
}

void dtw_free_features(feature_matrix_t* features) {
    if (!features) return;
    
    if (features->features) {
        delete[] features->features;
        features->features = nullptr;
    }
    
    features->num_frames = 0;
    features->num_features = 0;
}

const char* dtw_get_error_string(int32_t error_code) {
    switch (error_code) {
        case DTW_SUCCESS: return "Success";
        case DTW_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case DTW_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case DTW_ERROR_SEQUENCE_TOO_LONG: return "Sequence too long";
        case DTW_ERROR_FEATURE_EXTRACTION_FAILED: return "Feature extraction failed";
        case DTW_ERROR_NO_VALID_PATH: return "No valid path found";
        default: return "Unknown error";
    }
}