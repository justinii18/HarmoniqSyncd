#include "sync_core.h"
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>

namespace {
constexpr const char kSyncCoreVersion[] = "0.1.0-alpha";

// Helper functions for FFT and signal processing
class FFTProcessor {
public:
    static std::vector<std::complex<float>> computeFFT(const std::vector<float>& input);
    static std::vector<float> computeIFFT(const std::vector<std::complex<float>>& input);
    static std::vector<std::complex<float>> applyPHATWeighting(
        const std::vector<std::complex<float>>& cross_spectrum);
};

// GCC-PHAT implementation
class GCCPHATAligner {
public:
    static int32_t computeOffset(
        const sync_audio_buffer_t* reference,
        const sync_audio_buffer_t* target,
        const sync_gcc_phat_params_t* params,
        double* offset_seconds,
        double* confidence);

private:
    static std::vector<float> extractWindow(
        const sync_audio_buffer_t* buffer, 
        uint32_t start_sample, 
        uint32_t window_size);
    static double findPeakOffset(
        const std::vector<float>& correlation,
        double sample_rate,
        double max_offset);
};

// DTW implementation
class DTWAligner {
public:
    static int32_t computeAlignment(
        const sync_audio_buffer_t* reference,
        const sync_audio_buffer_t* target,
        const sync_dtw_params_t* params,
        double initial_offset_seconds,
        sync_alignment_result_t* result);

private:
    static std::vector<std::vector<float>> computeDistanceMatrix(
        const std::vector<float>& ref_features,
        const std::vector<float>& target_features);
    static std::vector<std::pair<int, int>> findOptimalPath(
        const std::vector<std::vector<float>>& distance_matrix,
        const sync_dtw_params_t* params);
};

// RANSAC drift estimation
class RANSACDriftEstimator {
public:
    static int32_t estimateDrift(
        const sync_audio_buffer_t* reference,
        const sync_audio_buffer_t* target,
        const sync_ransac_params_t* params,
        double initial_offset_seconds,
        sync_alignment_result_t* result);

private:
    static std::vector<std::pair<double, double>> findAnchorPoints(
        const sync_audio_buffer_t* reference,
        const sync_audio_buffer_t* target,
        double initial_offset_seconds);
    static bool fitLinearModel(
        const std::vector<std::pair<double, double>>& points,
        double* slope,
        double* intercept);
};

} // anonymous namespace

// Public API implementation
const char *sync_core_version(void) {
    return kSyncCoreVersion;
}

int32_t sync_core_initialize(void) {
    return SYNC_SUCCESS;
}

int32_t sync_compute_gcc_phat_offset(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_gcc_phat_params_t *params,
    double *offset_seconds,
    double *confidence) {
    
    if (!reference || !target || !params || !offset_seconds || !confidence) {
        return SYNC_ERROR_INVALID_PARAMS;
    }
    
    if (reference->sample_count < params->fft_size || 
        target->sample_count < params->fft_size) {
        return SYNC_ERROR_INSUFFICIENT_DATA;
    }
    
    return GCCPHATAligner::computeOffset(reference, target, params, offset_seconds, confidence);
}

int32_t sync_compute_dtw_alignment(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_dtw_params_t *params,
    double initial_offset_seconds,
    sync_alignment_result_t *result) {
    
    if (!reference || !target || !params || !result) {
        return SYNC_ERROR_INVALID_PARAMS;
    }
    
    return DTWAligner::computeAlignment(reference, target, params, initial_offset_seconds, result);
}

int32_t sync_estimate_drift_ransac(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_ransac_params_t *params,
    double initial_offset_seconds,
    sync_alignment_result_t *result) {
    
    if (!reference || !target || !params || !result) {
        return SYNC_ERROR_INVALID_PARAMS;
    }
    
    return RANSACDriftEstimator::estimateDrift(reference, target, params, initial_offset_seconds, result);
}

int32_t sync_align_audio_signals(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_gcc_phat_params_t *gcc_params,
    const sync_dtw_params_t *dtw_params,
    const sync_ransac_params_t *ransac_params,
    sync_alignment_result_t *result) {
    
    if (!reference || !target || !gcc_params || !dtw_params || !ransac_params || !result) {
        return SYNC_ERROR_INVALID_PARAMS;
    }
    
    // Initialize result
    result->error_code = SYNC_SUCCESS;
    result->coarse_offset_seconds = 0.0;
    result->refined_offset_seconds = 0.0;
    result->drift_rate = 0.0;
    result->confidence_score = 0.0;
    result->num_anchor_points = 0;
    result->has_drift = false;
    
    // Step 1: Coarse alignment with GCC-PHAT
    double coarse_offset, coarse_confidence;
    int32_t gcc_result = sync_compute_gcc_phat_offset(reference, target, gcc_params, 
                                                      &coarse_offset, &coarse_confidence);
    if (gcc_result != SYNC_SUCCESS) {
        result->error_code = gcc_result;
        return gcc_result;
    }
    
    result->coarse_offset_seconds = coarse_offset;
    
    // Step 2: Fine alignment with DTW
    sync_alignment_result_t dtw_result;
    int32_t dtw_status = sync_compute_dtw_alignment(reference, target, dtw_params, 
                                                    coarse_offset, &dtw_result);
    if (dtw_status == SYNC_SUCCESS) {
        result->refined_offset_seconds = dtw_result.refined_offset_seconds;
        result->confidence_score = std::max(coarse_confidence, dtw_result.confidence_score);
    } else {
        result->refined_offset_seconds = coarse_offset;
        result->confidence_score = coarse_confidence;
    }
    
    // Step 3: Drift estimation with RANSAC
    sync_alignment_result_t ransac_result;
    int32_t ransac_status = sync_estimate_drift_ransac(reference, target, ransac_params,
                                                       result->refined_offset_seconds, &ransac_result);
    if (ransac_status == SYNC_SUCCESS) {
        result->drift_rate = ransac_result.drift_rate;
        result->has_drift = std::abs(ransac_result.drift_rate) > 1e-6;
        result->num_anchor_points = ransac_result.num_anchor_points;
    }
    
    return SYNC_SUCCESS;
}

sync_gcc_phat_params_t sync_get_default_gcc_phat_params(double sample_rate) {
    sync_gcc_phat_params_t params;
    params.fft_size = 4096;
    params.hop_length = 1024;
    params.max_offset_seconds = 60.0;
    params.frequency_weight_alpha = 0.8;
    params.enable_prefiltering = true;
    params.prefilter_low_hz = 80.0;
    params.prefilter_high_hz = sample_rate / 2.0 * 0.8;
    return params;
}

sync_dtw_params_t sync_get_default_dtw_params(double sample_rate) {
    sync_dtw_params_t params;
    params.max_warp_samples = static_cast<uint32_t>(sample_rate * 0.5); // 500ms max warp
    params.constraint_radius = 0.1;
    params.step_pattern_penalty = 1.0;
    params.min_path_length = 10;
    params.enable_slope_constraint = true;
    return params;
}

sync_ransac_params_t sync_get_default_ransac_params(void) {
    sync_ransac_params_t params;
    params.max_iterations = 1000;
    params.inlier_threshold_seconds = 0.001; // 1ms threshold
    params.min_inliers = 10;
    params.consensus_threshold = 0.8;
    params.sample_size = 3;
    return params;
}

namespace {

// FFT implementation using simple DFT (for demonstration - production would use FFTW)
std::vector<std::complex<float>> FFTProcessor::computeFFT(const std::vector<float>& input) {
    const size_t N = input.size();
    std::vector<std::complex<float>> output(N);
    
    const float pi2 = -2.0f * M_PI;
    
    for (size_t k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float angle = pi2 * k * n / N;
            sum += input[n] * std::complex<float>(cos(angle), sin(angle));
        }
        output[k] = sum;
    }
    
    return output;
}

std::vector<float> FFTProcessor::computeIFFT(const std::vector<std::complex<float>>& input) {
    const size_t N = input.size();
    std::vector<std::complex<float>> temp(N);
    
    const float pi2 = 2.0f * M_PI;
    
    for (size_t n = 0; n < N; ++n) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t k = 0; k < N; ++k) {
            float angle = pi2 * k * n / N;
            sum += input[k] * std::complex<float>(cos(angle), sin(angle));
        }
        temp[n] = sum / static_cast<float>(N);
    }
    
    std::vector<float> output(N);
    for (size_t i = 0; i < N; ++i) {
        output[i] = temp[i].real();
    }
    
    return output;
}

std::vector<std::complex<float>> FFTProcessor::applyPHATWeighting(
    const std::vector<std::complex<float>>& cross_spectrum) {
    
    std::vector<std::complex<float>> weighted(cross_spectrum.size());
    
    for (size_t i = 0; i < cross_spectrum.size(); ++i) {
        float magnitude = std::abs(cross_spectrum[i]);
        if (magnitude > 1e-8f) {
            weighted[i] = cross_spectrum[i] / magnitude;
        } else {
            weighted[i] = std::complex<float>(0.0f, 0.0f);
        }
    }
    
    return weighted;
}

// GCC-PHAT implementation
int32_t GCCPHATAligner::computeOffset(
    const sync_audio_buffer_t* reference,
    const sync_audio_buffer_t* target,
    const sync_gcc_phat_params_t* params,
    double* offset_seconds,
    double* confidence) {
    
    try {
        // Extract windows from both signals
        auto ref_window = extractWindow(reference, 0, params->fft_size);
        auto target_window = extractWindow(target, 0, params->fft_size);
        
        // Compute FFTs
        auto ref_fft = FFTProcessor::computeFFT(ref_window);
        auto target_fft = FFTProcessor::computeFFT(target_window);
        
        // Compute cross-correlation spectrum
        std::vector<std::complex<float>> cross_spectrum(ref_fft.size());
        for (size_t i = 0; i < ref_fft.size(); ++i) {
            cross_spectrum[i] = ref_fft[i] * std::conj(target_fft[i]);
        }
        
        // Apply PHAT weighting
        auto weighted_spectrum = FFTProcessor::applyPHATWeighting(cross_spectrum);
        
        // Compute inverse FFT to get correlation
        auto correlation = FFTProcessor::computeIFFT(weighted_spectrum);
        
        // Find peak offset
        double offset = findPeakOffset(correlation, reference->sample_rate, params->max_offset_seconds);
        
        // Compute confidence based on peak height
        auto max_iter = std::max_element(correlation.begin(), correlation.end());
        float peak_value = *max_iter;
        float mean_value = 0.0f;
        for (float val : correlation) {
            mean_value += std::abs(val);
        }
        mean_value /= correlation.size();
        
        *confidence = std::min(1.0, static_cast<double>(peak_value / (mean_value + 1e-8f)));
        *offset_seconds = offset;
        
        return SYNC_SUCCESS;
        
    } catch (...) {
        return SYNC_ERROR_FFT_FAILURE;
    }
}

std::vector<float> GCCPHATAligner::extractWindow(
    const sync_audio_buffer_t* buffer, 
    uint32_t start_sample, 
    uint32_t window_size) {
    
    std::vector<float> window(window_size, 0.0f);
    uint32_t copy_size = std::min(window_size, buffer->sample_count - start_sample);
    
    for (uint32_t i = 0; i < copy_size; ++i) {
        window[i] = buffer->samples[start_sample + i];
    }
    
    return window;
}

double GCCPHATAligner::findPeakOffset(
    const std::vector<float>& correlation,
    double sample_rate,
    double max_offset) {
    
    const uint32_t max_offset_samples = static_cast<uint32_t>(max_offset * sample_rate);
    const uint32_t N = correlation.size();
    
    // Find peak in both positive and negative lag regions
    float max_value = correlation[0];
    uint32_t max_index = 0;
    
    // Check positive lags
    for (uint32_t i = 1; i < std::min(max_offset_samples, N/2); ++i) {
        if (correlation[i] > max_value) {
            max_value = correlation[i];
            max_index = i;
        }
    }
    
    // Check negative lags
    for (uint32_t i = N - max_offset_samples; i < N; ++i) {
        if (correlation[i] > max_value) {
            max_value = correlation[i];
            max_index = i;
        }
    }
    
    // Convert to seconds
    double offset_samples = (max_index > N/2) ? static_cast<double>(max_index) - N : max_index;
    return offset_samples / sample_rate;
}

// DTW implementation (simplified)
int32_t DTWAligner::computeAlignment(
    const sync_audio_buffer_t* reference,
    const sync_audio_buffer_t* target,
    const sync_dtw_params_t* params,
    double initial_offset_seconds,
    sync_alignment_result_t* result) {
    
    // For demonstration, return a refined offset close to the initial
    result->refined_offset_seconds = initial_offset_seconds;
    result->confidence_score = 0.8;
    result->error_code = SYNC_SUCCESS;
    
    return SYNC_SUCCESS;
}

// RANSAC implementation (simplified)
int32_t RANSACDriftEstimator::estimateDrift(
    const sync_audio_buffer_t* reference,
    const sync_audio_buffer_t* target,
    const sync_ransac_params_t* params,
    double initial_offset_seconds,
    sync_alignment_result_t* result) {
    
    // For demonstration, assume no drift
    result->drift_rate = 0.0;
    result->has_drift = false;
    result->num_anchor_points = 0;
    result->error_code = SYNC_SUCCESS;
    
    return SYNC_SUCCESS;
}

} // anonymous namespace