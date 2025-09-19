#pragma once

#include "ProductionFFT.h"
#include <stdint.h>
#include <stdbool.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// DTW step patterns
typedef enum {
    DTW_STEP_SYMMETRIC = 0,     // Symmetric pattern (1,1), (1,0), (0,1)
    DTW_STEP_ASYMMETRIC,        // Asymmetric pattern allowing more flexibility
    DTW_STEP_ITAKURA,          // Itakura parallelogram constraint
    DTW_STEP_SAKOE_CHIBA       // Sakoe-Chiba band constraint
} dtw_step_pattern_t;

// Feature types for DTW
typedef enum {
    DTW_FEATURE_RAW = 0,        // Raw audio samples
    DTW_FEATURE_MFCC,           // Mel-Frequency Cepstral Coefficients
    DTW_FEATURE_CHROMA,         // Chroma features (pitch class)
    DTW_FEATURE_SPECTRAL,       // Spectral features (centroid, rolloff, etc.)
    DTW_FEATURE_ENERGY,         // Energy-based features
    DTW_FEATURE_ZCR             // Zero Crossing Rate
} dtw_feature_type_t;

// Distance metrics for DTW
typedef enum {
    DTW_DISTANCE_EUCLIDEAN = 0, // Standard Euclidean distance
    DTW_DISTANCE_MANHATTAN,     // Manhattan (L1) distance
    DTW_DISTANCE_COSINE,        // Cosine distance
    DTW_DISTANCE_CORRELATION    // Correlation-based distance
} dtw_distance_metric_t;

// DTW configuration
typedef struct {
    dtw_step_pattern_t step_pattern;        // Step pattern constraint
    dtw_feature_type_t feature_type;        // Feature type to use
    dtw_distance_metric_t distance_metric;  // Distance metric
    
    // Constraints
    uint32_t max_warp_samples;              // Maximum warping in samples
    double constraint_radius;               // Constraint radius (0.0-1.0)
    uint32_t min_path_length;               // Minimum path length
    bool enable_slope_constraint;           // Enable slope constraints
    
    // Feature extraction parameters
    uint32_t feature_window_size;           // Window size for feature extraction
    uint32_t feature_hop_length;            // Hop length for features
    uint32_t num_mfcc_coeffs;              // Number of MFCC coefficients
    uint32_t num_chroma_bins;              // Number of chroma bins
    
    // Performance optimizations
    bool enable_early_termination;         // Early termination for poor matches
    double early_termination_threshold;    // Threshold for early termination
    bool enable_diagonal_optimization;     // Optimize for near-diagonal paths
    
    // Memory management
    uint32_t max_sequence_length;          // Maximum sequence length to process
    bool enable_banded_computation;        // Use banded DTW for efficiency
} dtw_config_t;

// DTW alignment result
typedef struct {
    double refined_offset_seconds;          // Refined offset from DTW
    double confidence_score;                // Confidence of alignment
    double total_distance;                  // Total DTW distance
    double normalized_distance;             // Distance normalized by path length
    
    // Path information
    uint32_t path_length;                   // Length of optimal path
    uint32_t* reference_indices;            // Reference sequence indices
    uint32_t* target_indices;               // Target sequence indices
    
    // Quality metrics
    double path_straightness;               // How straight the path is
    double diagonal_deviation;              // Deviation from diagonal
    double local_consistency;               // Local consistency metric
    
    bool is_reliable;                       // Overall reliability assessment
    int32_t error_code;                     // Error code
} dtw_result_t;

// Feature matrix structure
typedef struct {
    float* features;                        // Feature matrix (row-major)
    uint32_t num_frames;                    // Number of time frames
    uint32_t num_features;                  // Number of features per frame
    double sample_rate;                     // Original sample rate
    double frame_rate;                      // Frame rate of features
} feature_matrix_t;

// DTW context
typedef struct dtw_context dtw_context_t;

// Error codes (extending existing ones)
#define DTW_SUCCESS 0
#define DTW_ERROR_INVALID_PARAMS -1
#define DTW_ERROR_MEMORY_ALLOCATION -2
#define DTW_ERROR_SEQUENCE_TOO_LONG -3
#define DTW_ERROR_FEATURE_EXTRACTION_FAILED -4
#define DTW_ERROR_NO_VALID_PATH -5

// Context management
int32_t dtw_create_context(const dtw_config_t* config, dtw_context_t** context);
void dtw_destroy_context(dtw_context_t* context);

// Main DTW alignment
int32_t dtw_align_sequences(
    dtw_context_t* context,
    const float* reference_audio,
    const float* target_audio,
    uint32_t reference_length,
    uint32_t target_length,
    double sample_rate,
    double initial_offset_seconds,
    dtw_result_t* result
);

// Feature extraction
int32_t dtw_extract_features(
    dtw_context_t* context,
    const float* audio,
    uint32_t audio_length,
    double sample_rate,
    feature_matrix_t* features
);

// Direct feature-based alignment
int32_t dtw_align_features(
    dtw_context_t* context,
    const feature_matrix_t* reference_features,
    const feature_matrix_t* target_features,
    double initial_offset_seconds,
    dtw_result_t* result
);

// Utility functions
dtw_config_t dtw_get_default_config(void);
void dtw_free_result(dtw_result_t* result);
void dtw_free_features(feature_matrix_t* features);
const char* dtw_get_error_string(int32_t error_code);

#ifdef __cplusplus
}

// C++ Interface for DTW
namespace ProductionDTW {

class DTWAligner {
public:
    explicit DTWAligner(const dtw_config_t& config);
    ~DTWAligner();
    
    // Main alignment methods
    dtw_result_t align(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        double initial_offset = 0.0
    );
    
    dtw_result_t alignWithFeatures(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        dtw_feature_type_t feature_type,
        double initial_offset = 0.0
    );
    
    // Advanced alignment with custom features
    dtw_result_t alignFeatureMatrices(
        const feature_matrix_t& ref_features,
        const feature_matrix_t& target_features,
        double initial_offset = 0.0
    );
    
    // Configuration
    void updateConfig(const dtw_config_t& config);
    dtw_config_t getConfig() const { return config_; }
    
    // Feature extraction utilities
    feature_matrix_t extractMFCC(
        const std::vector<float>& audio,
        double sample_rate,
        uint32_t num_coeffs = 13
    );
    
    feature_matrix_t extractChroma(
        const std::vector<float>& audio,
        double sample_rate,
        uint32_t num_bins = 12
    );
    
    feature_matrix_t extractSpectralFeatures(
        const std::vector<float>& audio,
        double sample_rate
    );
    
private:
    dtw_context_t* context_;
    dtw_config_t config_;
    std::unique_ptr<ProductionFFT::FFTProcessor> fft_processor_;
    
    // Internal processing methods
    std::vector<std::vector<float>> computeDistanceMatrix(
        const feature_matrix_t& ref_features,
        const feature_matrix_t& target_features
    );
    
    std::vector<std::pair<uint32_t, uint32_t>> findOptimalPath(
        const std::vector<std::vector<float>>& distance_matrix,
        uint32_t ref_length,
        uint32_t target_length
    );
    
    double computeDistance(
        const float* feature1,
        const float* feature2,
        uint32_t feature_dim,
        dtw_distance_metric_t metric
    );
    
    bool isValidStep(
        uint32_t ref_prev, uint32_t target_prev,
        uint32_t ref_curr, uint32_t target_curr,
        dtw_step_pattern_t pattern
    );
    
    void assessReliability(dtw_result_t& result);
};

// Utility classes
class FeatureExtractor {
public:
    FeatureExtractor(uint32_t fft_size = 2048, double sample_rate = 48000.0);
    
    // MFCC extraction
    feature_matrix_t extractMFCC(
        const std::vector<float>& audio,
        uint32_t num_coeffs = 13,
        uint32_t num_filters = 26
    );
    
    // Chroma extraction
    feature_matrix_t extractChroma(
        const std::vector<float>& audio,
        uint32_t num_bins = 12
    );
    
    // Spectral features
    std::vector<float> computeSpectralCentroid(const std::vector<float>& magnitude_spectrum);
    std::vector<float> computeSpectralRolloff(const std::vector<float>& magnitude_spectrum, float percentile = 0.85f);
    std::vector<float> computeZeroCrossingRate(const std::vector<float>& audio, uint32_t frame_size);
    
    // Mel filter bank
    std::vector<std::vector<float>> createMelFilterBank(
        uint32_t num_filters,
        uint32_t fft_size,
        double sample_rate,
        double low_freq = 0.0,
        double high_freq = -1.0  // -1 means Nyquist
    );
    
private:
    std::unique_ptr<ProductionFFT::FFTProcessor> fft_processor_;
    double sample_rate_;
    uint32_t fft_size_;
    std::vector<std::vector<float>> mel_filter_bank_;
    
    void initializeMelFilterBank(uint32_t num_filters);
    double melToHz(double mel);
    double hzToMel(double hz);
};

class PathAnalyzer {
public:
    // Path quality metrics
    static double computePathStraightness(
        const std::vector<std::pair<uint32_t, uint32_t>>& path
    );
    
    static double computeDiagonalDeviation(
        const std::vector<std::pair<uint32_t, uint32_t>>& path,
        uint32_t ref_length,
        uint32_t target_length
    );
    
    static double computeLocalConsistency(
        const std::vector<std::pair<uint32_t, uint32_t>>& path
    );
    
    // Path smoothing and optimization
    static std::vector<std::pair<uint32_t, uint32_t>> smoothPath(
        const std::vector<std::pair<uint32_t, uint32_t>>& path,
        uint32_t smoothing_window = 5
    );
    
    static std::vector<std::pair<uint32_t, uint32_t>> optimizePath(
        const std::vector<std::pair<uint32_t, uint32_t>>& path,
        const std::vector<std::vector<float>>& distance_matrix
    );
};

} // namespace ProductionDTW

#endif