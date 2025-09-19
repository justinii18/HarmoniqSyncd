#include "include/AdvancedGCCPHAT.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <numeric>

// Internal context structure
struct advanced_gcc_phat_context {
    advanced_gcc_phat_params_t params;
    production_fft_context_t* fft_context;
    bool is_initialized;
    
    // Multi-scale FFT contexts
    std::vector<production_fft_context_t*> multiscale_fft_contexts;
    
    // Preprocessing filters
    std::vector<float> prefilter_coeffs;
    std::vector<float> filter_state;
    
    advanced_gcc_phat_context() : fft_context(nullptr), is_initialized(false) {}
};

namespace {
    // Constants
    const double PI = 3.14159265358979323846;
    const float MIN_MAGNITUDE = 1e-12f;
    const float MIN_COHERENCE = 1e-6f;
    
    // Apply frequency domain pre-filtering
    void apply_frequency_filter(
        std::vector<std::complex<float>>& spectrum,
        double sample_rate,
        double low_hz,
        double high_hz
    ) {
        uint32_t fft_size = spectrum.size();
        double freq_resolution = sample_rate / (2 * (fft_size - 1));
        
        for (uint32_t i = 0; i < fft_size; i++) {
            double freq = i * freq_resolution;
            
            // Apply high-pass filter
            if (freq < low_hz) {
                float attenuation = static_cast<float>(freq / low_hz);
                spectrum[i] *= attenuation;
            }
            
            // Apply low-pass filter
            if (freq > high_hz) {
                float attenuation = static_cast<float>(high_hz / freq);
                spectrum[i] *= std::min(1.0f, attenuation);
            }
        }
    }
    
    // Compute power spectral density for weighting
    std::vector<float> compute_psd(const std::vector<std::complex<float>>& spectrum) {
        std::vector<float> psd(spectrum.size());
        for (size_t i = 0; i < spectrum.size(); i++) {
            psd[i] = std::norm(spectrum[i]);
        }
        return psd;
    }
    
    // Simple smoothing filter for coherence estimation
    std::vector<float> smooth_spectrum(const std::vector<float>& spectrum, uint32_t window_size) {
        std::vector<float> smoothed(spectrum.size());
        uint32_t half_window = window_size / 2;
        
        for (size_t i = 0; i < spectrum.size(); i++) {
            float sum = 0.0f;
            uint32_t count = 0;
            
            for (int32_t j = -static_cast<int32_t>(half_window); 
                 j <= static_cast<int32_t>(half_window); j++) {
                int32_t idx = static_cast<int32_t>(i) + j;
                if (idx >= 0 && idx < static_cast<int32_t>(spectrum.size())) {
                    sum += spectrum[idx];
                    count++;
                }
            }
            
            smoothed[i] = count > 0 ? sum / count : 0.0f;
        }
        
        return smoothed;
    }
}

// C Interface Implementation
int32_t advanced_gcc_phat_create_context(
    const advanced_gcc_phat_params_t* params,
    advanced_gcc_phat_context_t** context) {
    
    if (!params || !context) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    auto* ctx = new advanced_gcc_phat_context();
    if (!ctx) {
        return FFT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Store parameters
    ctx->params = *params;
    
    // Create primary FFT context
    fft_config_t fft_config = fft_get_default_config(params->fft_size);
    fft_config.window_type = params->window_type;
    
    int32_t result = fft_create_context(&fft_config, &ctx->fft_context);
    if (result != FFT_SUCCESS) {
        delete ctx;
        return result;
    }
    
    // Create multi-scale FFT contexts if enabled
    if (params->enable_multiscale) {
        ctx->multiscale_fft_contexts.resize(params->multiscale_config.num_scales);
        
        for (uint32_t i = 0; i < params->multiscale_config.num_scales; i++) {
            fft_config_t scale_config = fft_get_default_config(
                params->multiscale_config.fft_sizes[i]
            );
            scale_config.window_type = params->window_type;
            
            result = fft_create_context(&scale_config, &ctx->multiscale_fft_contexts[i]);
            if (result != FFT_SUCCESS) {
                // Cleanup already created contexts
                for (uint32_t j = 0; j < i; j++) {
                    fft_destroy_context(ctx->multiscale_fft_contexts[j]);
                }
                fft_destroy_context(ctx->fft_context);
                delete ctx;
                return result;
            }
        }
    }
    
    ctx->is_initialized = true;
    *context = ctx;
    
    return FFT_SUCCESS;
}

void advanced_gcc_phat_destroy_context(advanced_gcc_phat_context_t* context) {
    if (!context) return;
    
    if (context->fft_context) {
        fft_destroy_context(context->fft_context);
    }
    
    for (auto* fft_ctx : context->multiscale_fft_contexts) {
        if (fft_ctx) {
            fft_destroy_context(fft_ctx);
        }
    }
    
    delete context;
}

advanced_gcc_phat_params_t advanced_gcc_phat_get_default_params(void) {
    advanced_gcc_phat_params_t params;
    
    // Basic parameters
    params.fft_size = 4096;
    params.hop_length = 1024;
    params.max_offset_seconds = 60.0;
    params.weighting_type = WEIGHTING_PHAT;
    
    // Sub-sample accuracy
    params.enable_subsample_accuracy = true;
    params.interpolation_factor = 8;
    
    // Frequency filtering
    params.enable_prefiltering = true;
    params.prefilter_low_hz = 80.0;
    params.prefilter_high_hz = 18000.0;
    
    // Multi-scale analysis (disabled by default)
    params.enable_multiscale = false;
    params.multiscale_config.num_scales = 0;
    params.multiscale_config.fft_sizes = nullptr;
    params.multiscale_config.scale_weights = nullptr;
    params.multiscale_config.enable_consensus = false;
    params.multiscale_config.consensus_threshold = 0.8f;
    
    // Coherence analysis
    params.enable_coherence_analysis = true;
    params.coherence_window_size = 256;
    
    // Advanced options
    params.noise_floor_db = -60.0f;
    params.enable_bias_correction = true;
    params.window_type = WINDOW_HANN;
    
    return params;
}

// C++ Interface Implementation
namespace AdvancedGCCPHAT {

AdvancedAligner::AdvancedAligner(const advanced_gcc_phat_params_t& params) 
    : context_(nullptr), params_(params) {
    
    int32_t result = advanced_gcc_phat_create_context(&params_, &context_);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("Failed to create advanced GCC-PHAT context: " + 
                               std::string(fft_get_error_string(result)));
    }
    
    // Create FFT processor for additional operations
    fft_config_t fft_config = fft_get_default_config(params_.fft_size);
    fft_config.window_type = params_.window_type;
    fft_processor_ = std::make_unique<ProductionFFT::FFTProcessor>(fft_config);
}

AdvancedAligner::~AdvancedAligner() {
    if (context_) {
        advanced_gcc_phat_destroy_context(context_);
    }
}

advanced_alignment_result_t AdvancedAligner::align(
    const std::vector<float>& reference,
    const std::vector<float>& target,
    double sample_rate) {
    
    advanced_alignment_result_t result = {};
    
    try {
        // Preprocessing
        auto preprocessed_ref = params_.enable_prefiltering ? 
            preprocessAudio(reference, sample_rate) : reference;
        auto preprocessed_target = params_.enable_prefiltering ? 
            preprocessAudio(target, sample_rate) : target;
        
        // Ensure minimum size
        uint32_t min_size = std::max(params_.fft_size, 
                                   static_cast<uint32_t>(std::max(preprocessed_ref.size(), 
                                                                 preprocessed_target.size())));
        
        if (preprocessed_ref.size() < min_size) {
            preprocessed_ref.resize(min_size, 0.0f);
        }
        if (preprocessed_target.size() < min_size) {
            preprocessed_target.resize(min_size, 0.0f);
        }
        
        // Compute FFTs
        auto ref_fft = fft_processor_->computeFFT(preprocessed_ref);
        auto target_fft = fft_processor_->computeFFT(preprocessed_target);
        
        // Apply frequency filtering if enabled
        if (params_.enable_prefiltering) {
            apply_frequency_filter(ref_fft, sample_rate, 
                                 params_.prefilter_low_hz, params_.prefilter_high_hz);
            apply_frequency_filter(target_fft, sample_rate, 
                                 params_.prefilter_low_hz, params_.prefilter_high_hz);
        }
        
        // Compute weighted cross-spectrum
        auto weighted_cross_spectrum = computeWeightedCrossSpectrum(
            ref_fft, target_fft, params_.weighting_type, sample_rate
        );
        
        // Compute correlation via IFFT
        auto correlation = fft_processor_->computeIFFT(weighted_cross_spectrum);
        
        // Find coarse peak
        double coarse_offset = findSubSamplePeak(correlation, sample_rate, params_.max_offset_seconds);
        result.coarse_offset_seconds = coarse_offset;
        
        // Sub-sample refinement if enabled
        if (params_.enable_subsample_accuracy) {
            // Find integer peak index
            uint32_t N = correlation.size();
            uint32_t max_offset_samples = static_cast<uint32_t>(params_.max_offset_seconds * sample_rate);
            
            float max_value = correlation[0];
            uint32_t max_index = 0;
            
            // Search for peak
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
            
            // Apply sub-sample estimation
            result.fine_offset_seconds = computeSubSampleOffset(correlation, max_index, sample_rate);
        } else {
            result.fine_offset_seconds = coarse_offset;
        }
        
        // Compute confidence scores
        result.confidence_score = 0.0; // Basic confidence from peak ratio
        if (params_.enable_coherence_analysis) {
            result.coherence_score = computeCoherenceConfidence(ref_fft, target_fft);
        }
        
        // Assess reliability
        assessReliability(result);
        
        result.error_code = FFT_SUCCESS;
        
    } catch (...) {
        result.error_code = FFT_ERROR_ACCELERATE_FAILURE;
        result.is_reliable = false;
    }
    
    return result;
}

std::vector<float> AdvancedAligner::preprocessAudio(
    const std::vector<float>& audio, 
    double sample_rate) {
    
    // Simple high-pass and low-pass filtering
    std::vector<float> filtered = audio;
    
    // Apply simple IIR high-pass filter for low frequency cutoff
    if (params_.prefilter_low_hz > 0) {
        double rc = 1.0 / (2.0 * PI * params_.prefilter_low_hz);
        double dt = 1.0 / sample_rate;
        float alpha = static_cast<float>(rc / (rc + dt));
        
        float prev_input = 0.0f;
        float prev_output = 0.0f;
        
        for (size_t i = 0; i < filtered.size(); i++) {
            float output = alpha * (prev_output + filtered[i] - prev_input);
            prev_input = filtered[i];
            prev_output = output;
            filtered[i] = output;
        }
    }
    
    return filtered;
}

std::vector<std::complex<float>> AdvancedAligner::computeWeightedCrossSpectrum(
    const std::vector<std::complex<float>>& ref_fft,
    const std::vector<std::complex<float>>& target_fft,
    gcc_weighting_type_t weighting_type,
    double sample_rate) {
    
    std::vector<std::complex<float>> cross_spectrum(ref_fft.size());
    
    // Compute basic cross-spectrum
    for (size_t i = 0; i < ref_fft.size(); i++) {
        cross_spectrum[i] = ref_fft[i] * std::conj(target_fft[i]);
    }
    
    // Apply weighting
    switch (weighting_type) {
        case WEIGHTING_PHAT:
            return FrequencyWeighter::applyPHAT(cross_spectrum);
            
        case WEIGHTING_ROTH:
            return FrequencyWeighter::applyROTH(cross_spectrum, ref_fft, target_fft);
            
        case WEIGHTING_SCOT:
            return FrequencyWeighter::applySCOT(cross_spectrum, ref_fft, target_fft);
            
        case WEIGHTING_ML:
            return FrequencyWeighter::applyML(cross_spectrum, ref_fft, target_fft);
            
        case WEIGHTING_NONE:
        default:
            return cross_spectrum;
    }
}

double AdvancedAligner::findSubSamplePeak(
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

double AdvancedAligner::computeSubSampleOffset(
    const std::vector<float>& correlation,
    uint32_t peak_index,
    double sample_rate) {
    
    // Use parabolic interpolation for sub-sample accuracy
    double sub_sample_offset = SubSampleEstimator::parabolicInterpolation(correlation, peak_index);
    
    // Convert to time
    uint32_t N = correlation.size();
    double offset_samples = (peak_index > N/2) ? 
        static_cast<double>(peak_index) - N + sub_sample_offset : 
        peak_index + sub_sample_offset;
    
    return offset_samples / sample_rate;
}

double AdvancedAligner::computeCoherenceConfidence(
    const std::vector<std::complex<float>>& ref_fft,
    const std::vector<std::complex<float>>& target_fft) {
    
    // Compute magnitude squared coherence
    double coherence_sum = 0.0;
    uint32_t valid_bins = 0;
    
    for (size_t i = 1; i < ref_fft.size() - 1; i++) { // Skip DC and Nyquist
        float ref_mag2 = std::norm(ref_fft[i]);
        float target_mag2 = std::norm(target_fft[i]);
        float cross_mag2 = std::norm(ref_fft[i] * std::conj(target_fft[i]));
        
        if (ref_mag2 > MIN_MAGNITUDE && target_mag2 > MIN_MAGNITUDE) {
            double coherence = cross_mag2 / (ref_mag2 * target_mag2);
            coherence_sum += coherence;
            valid_bins++;
        }
    }
    
    return valid_bins > 0 ? coherence_sum / valid_bins : 0.0;
}

void AdvancedAligner::assessReliability(advanced_alignment_result_t& result) {
    // Simple reliability assessment based on confidence and coherence
    result.is_reliable = true;
    
    if (result.confidence_score < 0.3) {
        result.is_reliable = false;
    }
    
    if (params_.enable_coherence_analysis && result.coherence_score < 0.5) {
        result.is_reliable = false;
    }
    
    // Check for reasonable offset
    if (std::abs(result.fine_offset_seconds) > params_.max_offset_seconds) {
        result.is_reliable = false;
    }
}

// FrequencyWeighter implementation
std::vector<std::complex<float>> FrequencyWeighter::applyPHAT(
    const std::vector<std::complex<float>>& cross_spectrum) {
    
    std::vector<std::complex<float>> weighted(cross_spectrum.size());
    
    for (size_t i = 0; i < cross_spectrum.size(); ++i) {
        float magnitude = std::abs(cross_spectrum[i]);
        if (magnitude > MIN_MAGNITUDE) {
            weighted[i] = cross_spectrum[i] / magnitude;
        } else {
            weighted[i] = std::complex<float>(0.0f, 0.0f);
        }
    }
    
    return weighted;
}

std::vector<std::complex<float>> FrequencyWeighter::applyROTH(
    const std::vector<std::complex<float>>& cross_spectrum,
    const std::vector<std::complex<float>>& ref_spectrum,
    const std::vector<std::complex<float>>& target_spectrum) {
    
    std::vector<std::complex<float>> weighted(cross_spectrum.size());
    
    for (size_t i = 0; i < cross_spectrum.size(); ++i) {
        float ref_power = std::norm(ref_spectrum[i]);
        float target_power = std::norm(target_spectrum[i]);
        float total_power = ref_power + target_power;
        
        if (total_power > MIN_MAGNITUDE) {
            weighted[i] = cross_spectrum[i] / total_power;
        } else {
            weighted[i] = std::complex<float>(0.0f, 0.0f);
        }
    }
    
    return weighted;
}

std::vector<std::complex<float>> FrequencyWeighter::applySCOT(
    const std::vector<std::complex<float>>& cross_spectrum,
    const std::vector<std::complex<float>>& ref_spectrum,
    const std::vector<std::complex<float>>& target_spectrum,
    double smoothing_factor) {
    
    // Compute power spectral densities
    auto ref_psd = compute_psd(ref_spectrum);
    auto target_psd = compute_psd(target_spectrum);
    
    // Apply smoothing
    uint32_t smooth_window = static_cast<uint32_t>(cross_spectrum.size() * smoothing_factor);
    auto smoothed_ref_psd = smooth_spectrum(ref_psd, smooth_window);
    auto smoothed_target_psd = smooth_spectrum(target_psd, smooth_window);
    
    std::vector<std::complex<float>> weighted(cross_spectrum.size());
    
    for (size_t i = 0; i < cross_spectrum.size(); ++i) {
        float denominator = sqrtf(smoothed_ref_psd[i] * smoothed_target_psd[i]);
        
        if (denominator > MIN_MAGNITUDE) {
            weighted[i] = cross_spectrum[i] / denominator;
        } else {
            weighted[i] = std::complex<float>(0.0f, 0.0f);
        }
    }
    
    return weighted;
}

std::vector<std::complex<float>> FrequencyWeighter::applyML(
    const std::vector<std::complex<float>>& cross_spectrum,
    const std::vector<std::complex<float>>& ref_spectrum,
    const std::vector<std::complex<float>>& target_spectrum,
    double noise_variance) {
    
    std::vector<std::complex<float>> weighted(cross_spectrum.size());
    
    for (size_t i = 0; i < cross_spectrum.size(); ++i) {
        float ref_power = std::norm(ref_spectrum[i]);
        float target_power = std::norm(target_spectrum[i]);
        float signal_power = std::min(ref_power, target_power);
        float snr = signal_power / noise_variance;
        
        float weight = snr / (1.0f + snr);
        weighted[i] = cross_spectrum[i] * weight;
    }
    
    return weighted;
}

// SubSampleEstimator implementation
double SubSampleEstimator::parabolicInterpolation(
    const std::vector<float>& data,
    uint32_t peak_index) {
    
    if (peak_index == 0 || peak_index >= data.size() - 1) {
        return 0.0; // Can't interpolate at edges
    }
    
    float y1 = data[peak_index - 1];
    float y2 = data[peak_index];
    float y3 = data[peak_index + 1];
    
    // Parabolic interpolation formula
    float a = (y1 - 2*y2 + y3) / 2.0f;
    float b = (y3 - y1) / 2.0f;
    
    if (std::abs(a) < 1e-10f) {
        return 0.0; // Avoid division by zero
    }
    
    return -b / (2.0f * a);
}

} // namespace AdvancedGCCPHAT

void advanced_gcc_phat_free_result(advanced_alignment_result_t* result) {
    if (!result) return;
    
    if (result->scale_offsets) {
        free(result->scale_offsets);
        result->scale_offsets = nullptr;
    }
    
    if (result->scale_confidences) {
        free(result->scale_confidences);
        result->scale_confidences = nullptr;
    }
    
    result->num_scale_results = 0;
}

const char* advanced_gcc_phat_get_error_string(int32_t error_code) {
    return fft_get_error_string(error_code);
}