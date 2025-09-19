#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// RANSAC drift estimation types
typedef enum {
    RANSAC_MODEL_LINEAR = 0,        // Linear drift (constant rate)
    RANSAC_MODEL_QUADRATIC,         // Quadratic drift (acceleration)
    RANSAC_MODEL_PIECEWISE_LINEAR,  // Piecewise linear (multiple segments)
    RANSAC_MODEL_THERMAL            // Thermal drift model
} ransac_drift_model_t;

// RANSAC configuration
typedef struct {
    ransac_drift_model_t model_type;        // Type of drift model
    
    // RANSAC parameters
    uint32_t max_iterations;                // Maximum RANSAC iterations
    double inlier_threshold_seconds;        // Threshold for inlier classification
    uint32_t min_inliers;                   // Minimum number of inliers
    double consensus_threshold;             // Consensus threshold (0.0-1.0)
    uint32_t sample_size;                   // Number of points to sample
    
    // Anchor point detection
    uint32_t anchor_window_size;            // Window size for anchor detection
    uint32_t anchor_hop_length;             // Hop length between anchors
    double anchor_confidence_threshold;     // Minimum confidence for anchors
    uint32_t max_anchor_points;             // Maximum number of anchor points
    
    // Drift detection sensitivity
    double min_drift_rate_ppm;              // Minimum detectable drift (parts per million)
    double max_drift_rate_ppm;              // Maximum expected drift
    bool enable_drift_validation;          // Enable drift model validation
    
    // Advanced options
    bool enable_outlier_rejection;         // Enable advanced outlier rejection
    bool enable_temporal_weighting;        // Weight anchor points by time
    double temporal_decay_factor;          // Decay factor for temporal weighting
} ransac_config_t;

// Anchor point structure
typedef struct {
    double reference_time_seconds;          // Time in reference signal
    double target_time_seconds;             // Time in target signal
    double confidence_score;                // Confidence of this anchor point
    double local_offset_seconds;            // Local offset at this point
} anchor_point_t;

// Drift model parameters
typedef struct {
    ransac_drift_model_t model_type;        // Model type
    
    // Linear model: offset = slope * time + intercept
    double slope;                           // Drift rate (samples/second)
    double intercept;                       // Initial offset (seconds)
    
    // Quadratic model: offset = a * time^2 + b * time + c
    double quadratic_a;                     // Quadratic coefficient
    double quadratic_b;                     // Linear coefficient  
    double quadratic_c;                     // Constant term
    
    // Model quality metrics
    double rmse;                            // Root mean square error
    double r_squared;                       // Coefficient of determination
    uint32_t num_inliers;                   // Number of inlier points
    double inlier_ratio;                    // Ratio of inliers to total points
} drift_model_t;

// RANSAC drift estimation result
typedef struct {
    drift_model_t drift_model;              // Best fit drift model
    
    // Drift characteristics
    double drift_rate_ppm;                  // Drift rate in parts per million
    double initial_offset_seconds;          // Initial offset
    double final_offset_seconds;            // Final offset after drift
    bool has_significant_drift;             // Whether drift is significant
    
    // Quality metrics
    double confidence_score;                // Overall confidence (0.0-1.0)
    double model_reliability;               // Model reliability score
    uint32_t num_anchor_points;             // Number of anchor points used
    
    // Anchor point data
    anchor_point_t* anchor_points;          // Array of anchor points
    uint32_t* inlier_indices;              // Indices of inlier anchor points
    uint32_t* outlier_indices;             // Indices of outlier anchor points
    uint32_t num_inliers;                  // Number of inlier points
    uint32_t num_outliers;                 // Number of outlier points
    
    int32_t error_code;                     // Error code
} ransac_result_t;

// RANSAC context
typedef struct ransac_context ransac_context_t;

// Error codes
#define RANSAC_SUCCESS 0
#define RANSAC_ERROR_INVALID_PARAMS -1
#define RANSAC_ERROR_MEMORY_ALLOCATION -2
#define RANSAC_ERROR_INSUFFICIENT_ANCHORS -3
#define RANSAC_ERROR_NO_CONSENSUS -4
#define RANSAC_ERROR_MODEL_FITTING_FAILED -5

// Context management
int32_t ransac_create_context(const ransac_config_t* config, ransac_context_t** context);
void ransac_destroy_context(ransac_context_t* context);

// Main RANSAC drift estimation
int32_t ransac_estimate_drift(
    ransac_context_t* context,
    const float* reference_audio,
    const float* target_audio,
    uint32_t reference_length,
    uint32_t target_length,
    double sample_rate,
    double initial_offset_seconds,
    ransac_result_t* result
);

// Anchor point detection
int32_t ransac_detect_anchor_points(
    ransac_context_t* context,
    const float* reference_audio,
    const float* target_audio,
    uint32_t reference_length,
    uint32_t target_length,
    double sample_rate,
    double initial_offset_seconds,
    anchor_point_t** anchor_points,
    uint32_t* num_anchor_points
);

// Drift model fitting
int32_t ransac_fit_drift_model(
    ransac_context_t* context,
    const anchor_point_t* anchor_points,
    uint32_t num_anchor_points,
    drift_model_t* model
);

// Utility functions
ransac_config_t ransac_get_default_config(void);
void ransac_free_result(ransac_result_t* result);
const char* ransac_get_error_string(int32_t error_code);

// Model evaluation
double ransac_evaluate_model(
    const drift_model_t* model,
    const anchor_point_t* anchor_points,
    uint32_t num_anchor_points
);

double ransac_compute_drift_rate_ppm(const drift_model_t* model, double duration_seconds);

#ifdef __cplusplus
}

// C++ Interface for RANSAC
namespace ProductionRANSAC {

class DriftEstimator {
public:
    explicit DriftEstimator(const ransac_config_t& config);
    ~DriftEstimator();
    
    // Main drift estimation method
    ransac_result_t estimateDrift(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        double initial_offset = 0.0
    );
    
    // Advanced drift estimation with custom anchor points
    ransac_result_t estimateDriftWithAnchors(
        const std::vector<anchor_point_t>& anchor_points,
        double duration_seconds
    );
    
    // Configuration
    void updateConfig(const ransac_config_t& config);
    ransac_config_t getConfig() const { return config_; }
    
    // Anchor point utilities
    std::vector<anchor_point_t> detectAnchorPoints(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        double initial_offset = 0.0
    );
    
    // Model fitting utilities
    drift_model_t fitLinearModel(const std::vector<anchor_point_t>& anchors);
    drift_model_t fitQuadraticModel(const std::vector<anchor_point_t>& anchors);
    
private:
    ransac_context_t* context_;
    ransac_config_t config_;
    
    // Internal processing methods
    std::vector<anchor_point_t> findAnchorPointsRANSAC(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        double initial_offset
    );
    
    drift_model_t performRANSAC(
        const std::vector<anchor_point_t>& anchor_points
    );
    
    bool validateDriftModel(
        const drift_model_t& model,
        const std::vector<anchor_point_t>& anchor_points
    );
    
    void assessModelReliability(
        ransac_result_t& result,
        const std::vector<anchor_point_t>& anchor_points
    );
};

// Utility classes
class AnchorPointDetector {
public:
    AnchorPointDetector(uint32_t window_size = 4096, double confidence_threshold = 0.7);
    
    // GCC-PHAT based anchor detection
    std::vector<anchor_point_t> detectWithGCCPHAT(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        uint32_t hop_length = 1024
    );
    
    // DTW based anchor detection
    std::vector<anchor_point_t> detectWithDTW(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        uint32_t hop_length = 1024
    );
    
    // Fingerprint based anchor detection
    std::vector<anchor_point_t> detectWithFingerprints(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate
    );
    
private:
    uint32_t window_size_;
    double confidence_threshold_;
    
    double computeAnchorConfidence(
        const std::vector<float>& ref_window,
        const std::vector<float>& target_window,
        double sample_rate
    );
};

class ModelFitter {
public:
    // RANSAC model fitting
    static drift_model_t fitRANSACLinear(
        const std::vector<anchor_point_t>& points,
        const ransac_config_t& config
    );
    
    static drift_model_t fitRANSACQuadratic(
        const std::vector<anchor_point_t>& points,
        const ransac_config_t& config
    );
    
    // Least squares fitting
    static drift_model_t fitLeastSquaresLinear(
        const std::vector<anchor_point_t>& points
    );
    
    static drift_model_t fitLeastSquaresQuadratic(
        const std::vector<anchor_point_t>& points
    );
    
    // Model evaluation
    static double evaluateModel(
        const drift_model_t& model,
        const std::vector<anchor_point_t>& points
    );
    
    static std::vector<uint32_t> findInliers(
        const drift_model_t& model,
        const std::vector<anchor_point_t>& points,
        double threshold
    );
    
private:
    static std::vector<uint32_t> sampleRandomPoints(
        uint32_t total_points,
        uint32_t sample_size
    );
    
    static double computeModelError(
        const drift_model_t& model,
        const anchor_point_t& point
    );
};

class DriftAnalyzer {
public:
    // Drift characterization
    static double computeDriftRatePPM(
        const drift_model_t& model,
        double duration_seconds
    );
    
    static bool isSignificantDrift(
        const drift_model_t& model,
        double min_ppm_threshold = 1.0
    );
    
    // Drift prediction
    static double predictOffset(
        const drift_model_t& model,
        double time_seconds
    );
    
    static std::vector<double> predictOffsetsOverTime(
        const drift_model_t& model,
        double duration_seconds,
        uint32_t num_points = 100
    );
    
    // Drift validation
    static bool validateDriftModel(
        const drift_model_t& model,
        const std::vector<anchor_point_t>& validation_points
    );
    
    static double computeModelConfidence(
        const drift_model_t& model,
        const std::vector<anchor_point_t>& points
    );
};

} // namespace ProductionRANSAC

#endif