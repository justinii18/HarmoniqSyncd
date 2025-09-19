#include "include/ProductionRANSAC.h"
#include "include/AdvancedGCCPHAT.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <limits>

// Internal context structure
struct ransac_context {
    ransac_config_t config;
    bool is_initialized;
    
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // Advanced GCC-PHAT context for anchor detection
    advanced_gcc_phat_context_t* gcc_phat_context;
    
    // Memory pools for efficiency
    std::vector<anchor_point_t> anchor_pool;
    std::vector<uint32_t> index_pool;
    
    ransac_context() : uniform_dist(0.0, 1.0), gcc_phat_context(nullptr), is_initialized(false) {
        rng.seed(std::random_device{}());
    }
};

namespace {
    const double PI = 3.14159265358979323846;
    const double MIN_ANCHOR_SEPARATION = 1.0; // Minimum separation between anchors (seconds)
    const double MAX_ANCHOR_SEPARATION = 10.0; // Maximum separation for reliable drift detection
    
    // Compute offset between two time points using a drift model
    double compute_model_offset(const drift_model_t* model, double time_seconds) {
        switch (model->model_type) {
            case RANSAC_MODEL_LINEAR:
                return model->slope * time_seconds + model->intercept;
                
            case RANSAC_MODEL_QUADRATIC:
                return model->quadratic_a * time_seconds * time_seconds + 
                       model->quadratic_b * time_seconds + 
                       model->quadratic_c;
                
            case RANSAC_MODEL_PIECEWISE_LINEAR:
            case RANSAC_MODEL_THERMAL:
            default:
                // Fallback to linear model
                return model->slope * time_seconds + model->intercept;
        }
    }
    
    // Compute model error for a single anchor point
    double compute_point_error(const drift_model_t* model, const anchor_point_t* anchor) {
        double predicted_offset = compute_model_offset(model, anchor->reference_time_seconds);
        double actual_offset = anchor->target_time_seconds - anchor->reference_time_seconds;
        return std::abs(predicted_offset - actual_offset);
    }
    
    // Fit linear model using least squares
    drift_model_t fit_linear_least_squares(const anchor_point_t* anchors, uint32_t num_anchors) {
        drift_model_t model = {};
        model.model_type = RANSAC_MODEL_LINEAR;
        
        if (num_anchors < 2) {
            return model;
        }
        
        // Compute sums for least squares
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        
        for (uint32_t i = 0; i < num_anchors; i++) {
            double x = anchors[i].reference_time_seconds;
            double y = anchors[i].target_time_seconds - anchors[i].reference_time_seconds;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        double n = static_cast<double>(num_anchors);
        double denominator = n * sum_x2 - sum_x * sum_x;
        
        if (std::abs(denominator) > 1e-10) {
            model.slope = (n * sum_xy - sum_x * sum_y) / denominator;
            model.intercept = (sum_y - model.slope * sum_x) / n;
        }
        
        // Compute RMSE and R-squared
        double sse = 0.0, sst = 0.0;
        double mean_y = sum_y / n;
        
        for (uint32_t i = 0; i < num_anchors; i++) {
            double x = anchors[i].reference_time_seconds;
            double y = anchors[i].target_time_seconds - anchors[i].reference_time_seconds;
            double predicted = model.slope * x + model.intercept;
            
            sse += (y - predicted) * (y - predicted);
            sst += (y - mean_y) * (y - mean_y);
        }
        
        model.rmse = std::sqrt(sse / n);
        model.r_squared = sst > 0 ? 1.0 - (sse / sst) : 0.0;
        model.num_inliers = num_anchors;
        model.inlier_ratio = 1.0;
        
        return model;
    }
    
    // Sample random subset of points
    std::vector<uint32_t> sample_random_indices(uint32_t total_size, uint32_t sample_size, std::mt19937& rng) {
        std::vector<uint32_t> indices(total_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::shuffle(indices.begin(), indices.end(), rng);
        
        indices.resize(std::min(sample_size, total_size));
        return indices;
    }
}

// C Interface Implementation
int32_t ransac_create_context(const ransac_config_t* config, ransac_context_t** context) {
    if (!config || !context) {
        return RANSAC_ERROR_INVALID_PARAMS;
    }
    
    auto* ctx = new ransac_context();
    if (!ctx) {
        return RANSAC_ERROR_MEMORY_ALLOCATION;
    }
    
    ctx->config = *config;
    
    // Create GCC-PHAT context for anchor detection
    auto gcc_params = advanced_gcc_phat_get_default_params();
    gcc_params.fft_size = config->anchor_window_size;
    gcc_params.enable_subsample_accuracy = true;
    gcc_params.enable_coherence_analysis = true;
    
    int32_t result = advanced_gcc_phat_create_context(&gcc_params, &ctx->gcc_phat_context);
    if (result != FFT_SUCCESS) {
        delete ctx;
        return result;
    }
    
    ctx->is_initialized = true;
    *context = ctx;
    
    return RANSAC_SUCCESS;
}

void ransac_destroy_context(ransac_context_t* context) {
    if (!context) return;
    
    if (context->gcc_phat_context) {
        advanced_gcc_phat_destroy_context(context->gcc_phat_context);
    }
    
    delete context;
}

ransac_config_t ransac_get_default_config(void) {
    ransac_config_t config;
    
    config.model_type = RANSAC_MODEL_LINEAR;
    
    // RANSAC parameters
    config.max_iterations = 1000;
    config.inlier_threshold_seconds = 0.001; // 1ms threshold
    config.min_inliers = 10;
    config.consensus_threshold = 0.8;
    config.sample_size = 3;
    
    // Anchor point detection
    config.anchor_window_size = 4096;
    config.anchor_hop_length = 1024;
    config.anchor_confidence_threshold = 0.7;
    config.max_anchor_points = 100;
    
    // Drift detection sensitivity
    config.min_drift_rate_ppm = 1.0;
    config.max_drift_rate_ppm = 1000.0;
    config.enable_drift_validation = true;
    
    // Advanced options
    config.enable_outlier_rejection = true;
    config.enable_temporal_weighting = false;
    config.temporal_decay_factor = 0.9;
    
    return config;
}

// C++ Interface Implementation
namespace ProductionRANSAC {

DriftEstimator::DriftEstimator(const ransac_config_t& config) 
    : context_(nullptr), config_(config) {
    
    int32_t result = ransac_create_context(&config_, &context_);
    if (result != RANSAC_SUCCESS) {
        throw std::runtime_error("Failed to create RANSAC context: " + 
                               std::string(ransac_get_error_string(result)));
    }
}

DriftEstimator::~DriftEstimator() {
    if (context_) {
        ransac_destroy_context(context_);
    }
}

ransac_result_t DriftEstimator::estimateDrift(
    const std::vector<float>& reference,
    const std::vector<float>& target,
    double sample_rate,
    double initial_offset) {
    
    ransac_result_t result = {};
    
    try {
        // Step 1: Detect anchor points throughout the audio
        auto anchor_points = findAnchorPointsRANSAC(reference, target, sample_rate, initial_offset);
        
        if (anchor_points.size() < config_.min_inliers) {
            result.error_code = RANSAC_ERROR_INSUFFICIENT_ANCHORS;
            return result;
        }
        
        // Step 2: Perform RANSAC to find best drift model
        auto drift_model = performRANSAC(anchor_points);
        
        if (drift_model.num_inliers < config_.min_inliers) {
            result.error_code = RANSAC_ERROR_NO_CONSENSUS;
            return result;
        }
        
        // Step 3: Fill result structure
        result.drift_model = drift_model;
        
        // Compute drift characteristics
        double duration = reference.size() / sample_rate;
        result.drift_rate_ppm = DriftAnalyzer::computeDriftRatePPM(drift_model, duration);
        result.initial_offset_seconds = drift_model.intercept + initial_offset;
        result.final_offset_seconds = compute_model_offset(&drift_model, duration) + initial_offset;
        result.has_significant_drift = std::abs(result.drift_rate_ppm) > config_.min_drift_rate_ppm;
        
        // Store anchor points
        result.num_anchor_points = anchor_points.size();
        result.anchor_points = static_cast<anchor_point_t*>(
            malloc(anchor_points.size() * sizeof(anchor_point_t))
        );
        if (result.anchor_points) {
            std::copy(anchor_points.begin(), anchor_points.end(), result.anchor_points);
        }
        
        // Classify inliers/outliers
        auto inlier_indices = ModelFitter::findInliers(drift_model, anchor_points, config_.inlier_threshold_seconds);
        result.num_inliers = inlier_indices.size();
        result.num_outliers = anchor_points.size() - inlier_indices.size();
        
        if (result.num_inliers > 0) {
            result.inlier_indices = static_cast<uint32_t*>(malloc(result.num_inliers * sizeof(uint32_t)));
            if (result.inlier_indices) {
                std::copy(inlier_indices.begin(), inlier_indices.end(), result.inlier_indices);
            }
        }
        
        // Assess overall reliability
        assessModelReliability(result, anchor_points);
        
        result.error_code = RANSAC_SUCCESS;
        
    } catch (...) {
        result.error_code = RANSAC_ERROR_MEMORY_ALLOCATION;
    }
    
    return result;
}

std::vector<anchor_point_t> DriftEstimator::findAnchorPointsRANSAC(
    const std::vector<float>& reference,
    const std::vector<float>& target,
    double sample_rate,
    double initial_offset) {
    
    std::vector<anchor_point_t> anchor_points;
    
    uint32_t window_size = config_.anchor_window_size;
    uint32_t hop_length = config_.anchor_hop_length;
    uint32_t num_windows = (reference.size() - window_size) / hop_length;
    
    // Create GCC-PHAT aligner for anchor detection
    auto gcc_params = advanced_gcc_phat_get_default_params();
    gcc_params.fft_size = window_size;
    gcc_params.enable_subsample_accuracy = true;
    
    AdvancedGCCPHAT::AdvancedAligner aligner(gcc_params);
    
    for (uint32_t i = 0; i < num_windows && anchor_points.size() < config_.max_anchor_points; i++) {
        uint32_t start_idx = i * hop_length;
        
        // Extract reference window
        std::vector<float> ref_window(reference.begin() + start_idx, 
                                    reference.begin() + start_idx + window_size);
        
        // Calculate target start index based on initial offset
        int64_t target_start = static_cast<int64_t>(start_idx + initial_offset * sample_rate);
        if (target_start < 0 || target_start + window_size >= target.size()) {
            continue;
        }
        
        // Extract target window
        std::vector<float> target_window(target.begin() + target_start,
                                       target.begin() + target_start + window_size);
        
        // Perform local alignment
        auto alignment_result = aligner.align(ref_window, target_window, sample_rate);
        
        if (alignment_result.error_code == FFT_SUCCESS && 
            alignment_result.is_reliable && 
            alignment_result.confidence_score > config_.anchor_confidence_threshold) {
            
            anchor_point_t anchor;
            anchor.reference_time_seconds = start_idx / sample_rate;
            anchor.target_time_seconds = anchor.reference_time_seconds + 
                                       initial_offset + alignment_result.fine_offset_seconds;
            anchor.confidence_score = alignment_result.confidence_score;
            anchor.local_offset_seconds = alignment_result.fine_offset_seconds;
            
            anchor_points.push_back(anchor);
        }
    }
    
    return anchor_points;
}

drift_model_t DriftEstimator::performRANSAC(const std::vector<anchor_point_t>& anchor_points) {
    drift_model_t best_model = {};
    best_model.model_type = config_.model_type;
    uint32_t best_inlier_count = 0;
    
    for (uint32_t iteration = 0; iteration < config_.max_iterations; iteration++) {
        // Sample random subset of points
        auto sample_indices = sample_random_indices(anchor_points.size(), config_.sample_size, context_->rng);
        
        if (sample_indices.size() < config_.sample_size) continue;
        
        // Create sample points
        std::vector<anchor_point_t> sample_points;
        for (uint32_t idx : sample_indices) {
            sample_points.push_back(anchor_points[idx]);
        }
        
        // Fit model to sample
        auto sample_model = fit_linear_least_squares(sample_points.data(), sample_points.size());
        
        // Count inliers
        uint32_t inlier_count = 0;
        for (const auto& anchor : anchor_points) {
            double error = compute_point_error(&sample_model, &anchor);
            if (error < config_.inlier_threshold_seconds) {
                inlier_count++;
            }
        }
        
        // Check if this is the best model so far
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_model = sample_model;
            best_model.num_inliers = inlier_count;
            best_model.inlier_ratio = static_cast<double>(inlier_count) / anchor_points.size();
            
            // Early termination if we have enough consensus
            if (best_model.inlier_ratio >= config_.consensus_threshold) {
                break;
            }
        }
    }
    
    // Refine model using all inliers
    if (best_inlier_count >= config_.min_inliers) {
        std::vector<anchor_point_t> inlier_points;
        for (const auto& anchor : anchor_points) {
            double error = compute_point_error(&best_model, &anchor);
            if (error < config_.inlier_threshold_seconds) {
                inlier_points.push_back(anchor);
            }
        }
        
        if (!inlier_points.empty()) {
            best_model = fit_linear_least_squares(inlier_points.data(), inlier_points.size());
            best_model.num_inliers = inlier_points.size();
            best_model.inlier_ratio = static_cast<double>(inlier_points.size()) / anchor_points.size();
        }
    }
    
    return best_model;
}

void DriftEstimator::assessModelReliability(
    ransac_result_t& result,
    const std::vector<anchor_point_t>& anchor_points) {
    
    // Compute confidence based on multiple factors
    double inlier_confidence = result.drift_model.inlier_ratio;
    double model_confidence = std::max(0.0, 1.0 - result.drift_model.rmse / config_.inlier_threshold_seconds);
    double anchor_confidence = 0.0;
    
    // Average anchor point confidence
    if (!anchor_points.empty()) {
        for (const auto& anchor : anchor_points) {
            anchor_confidence += anchor.confidence_score;
        }
        anchor_confidence /= anchor_points.size();
    }
    
    result.confidence_score = (inlier_confidence + model_confidence + anchor_confidence) / 3.0;
    result.model_reliability = result.drift_model.r_squared * inlier_confidence;
}

// ModelFitter implementation
std::vector<uint32_t> ModelFitter::findInliers(
    const drift_model_t& model,
    const std::vector<anchor_point_t>& points,
    double threshold) {
    
    std::vector<uint32_t> inlier_indices;
    
    for (size_t i = 0; i < points.size(); i++) {
        double error = compute_point_error(&model, &points[i]);
        if (error < threshold) {
            inlier_indices.push_back(static_cast<uint32_t>(i));
        }
    }
    
    return inlier_indices;
}

// DriftAnalyzer implementation
double DriftAnalyzer::computeDriftRatePPM(
    const drift_model_t& model,
    double duration_seconds) {
    
    if (duration_seconds <= 0.0) return 0.0;
    
    // For linear model, drift rate is the slope
    double drift_samples_per_second = model.slope;
    double drift_ratio = drift_samples_per_second / duration_seconds;
    
    return drift_ratio * 1e6; // Convert to parts per million
}

bool DriftAnalyzer::isSignificantDrift(
    const drift_model_t& model,
    double min_ppm_threshold) {
    
    return std::abs(model.slope * 1e6) > min_ppm_threshold;
}

} // namespace ProductionRANSAC

void ransac_free_result(ransac_result_t* result) {
    if (!result) return;
    
    if (result->anchor_points) {
        free(result->anchor_points);
        result->anchor_points = nullptr;
    }
    
    if (result->inlier_indices) {
        free(result->inlier_indices);
        result->inlier_indices = nullptr;
    }
    
    if (result->outlier_indices) {
        free(result->outlier_indices);
        result->outlier_indices = nullptr;
    }
    
    result->num_anchor_points = 0;
    result->num_inliers = 0;
    result->num_outliers = 0;
}

const char* ransac_get_error_string(int32_t error_code) {
    switch (error_code) {
        case RANSAC_SUCCESS: return "Success";
        case RANSAC_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case RANSAC_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case RANSAC_ERROR_INSUFFICIENT_ANCHORS: return "Insufficient anchor points";
        case RANSAC_ERROR_NO_CONSENSUS: return "No consensus found";
        case RANSAC_ERROR_MODEL_FITTING_FAILED: return "Model fitting failed";
        default: return "Unknown error";
    }
}