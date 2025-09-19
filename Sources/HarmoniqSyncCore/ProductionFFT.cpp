#include "include/ProductionFFT.h"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

// Internal context structure
struct production_fft_context {
    fft_config_t config;
    FFTSetup fft_setup;                    // vDSP FFT setup
    DSPSplitComplex split_complex_buffer;  // Split complex buffer for vDSP
    float* window_function;                // Precomputed window
    uint32_t log2_fft_size;               // log2(fft_size) for vDSP
    bool is_initialized;
    
    production_fft_context() :
        fft_setup(nullptr),
        window_function(nullptr),
        log2_fft_size(0),
        is_initialized(false) {
        split_complex_buffer.realp = nullptr;
        split_complex_buffer.imagp = nullptr;
    }
};

namespace {
    // Calculate log2 of a number
    uint32_t calculate_log2(uint32_t n) {
        uint32_t log2_n = 0;
        while ((1U << log2_n) < n) {
            log2_n++;
        }
        return log2_n;
    }
    
    // Generate window function
    void generate_window(window_type_t type, float* window, uint32_t size, float kaiser_beta = 8.0f) {
        switch (type) {
            case WINDOW_HANN:
                vDSP_hann_window(window, size, vDSP_HANN_NORM);
                break;
                
            case WINDOW_HAMMING:
                vDSP_hamm_window(window, size, 0);
                break;
                
            case WINDOW_BLACKMAN:
                vDSP_blkman_window(window, size, 0);
                break;
                
            case WINDOW_KAISER: {
                // Custom Kaiser window implementation using modified Bessel function
                float beta = kaiser_beta;
                
                // Simple approximation of I0 (modified Bessel function of the first kind)
                auto i0_approx = [](float x) -> float {
                    float sum = 1.0f;
                    float term = 1.0f;
                    float x_half_squared = (x * 0.5f) * (x * 0.5f);
                    
                    for (int k = 1; k < 50; k++) {
                        term *= x_half_squared / (k * k);
                        sum += term;
                        if (term < 1e-8f) break;
                    }
                    return sum;
                };
                
                float i0_beta = i0_approx(beta);
                
                for (uint32_t i = 0; i < size; i++) {
                    float x = 2.0f * i / (size - 1) - 1.0f;
                    float arg = beta * sqrtf(1.0f - x * x);
                    window[i] = i0_approx(arg) / i0_beta;
                }
                break;
            }
            
            case WINDOW_RECTANGULAR:
            default:
                // Rectangular window (all ones)
                for (uint32_t i = 0; i < size; i++) {
                    window[i] = 1.0f;
                }
                break;
        }
    }
    
    // Convert interleaved complex to split complex
    void interleaved_to_split(const std::vector<std::complex<float>>& interleaved, 
                             float* real, float* imag) {
        for (size_t i = 0; i < interleaved.size(); i++) {
            real[i] = interleaved[i].real();
            imag[i] = interleaved[i].imag();
        }
    }
    
    // Convert split complex to interleaved
    std::vector<std::complex<float>> split_to_interleaved(const float* real, const float* imag, 
                                                         size_t size) {
        std::vector<std::complex<float>> result(size);
        for (size_t i = 0; i < size; i++) {
            result[i] = std::complex<float>(real[i], imag[i]);
        }
        return result;
    }
}

// C Interface Implementation
int32_t fft_create_context(const fft_config_t* config, production_fft_context_t** context) {
    if (!config || !context) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    if (!fft_is_power_of_two(config->fft_size)) {
        return FFT_ERROR_INVALID_SIZE;
    }
    
    auto* ctx = new production_fft_context();
    if (!ctx) {
        return FFT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Store configuration
    ctx->config = *config;
    ctx->log2_fft_size = calculate_log2(config->fft_size);
    
    // Create vDSP FFT setup
    ctx->fft_setup = vDSP_create_fftsetup(ctx->log2_fft_size, kFFTRadix2);
    if (!ctx->fft_setup) {
        delete ctx;
        return FFT_ERROR_ACCELERATE_FAILURE;
    }
    
    // Allocate split complex buffers
    uint32_t half_size = config->fft_size / 2;
    ctx->split_complex_buffer.realp = static_cast<float*>(malloc(half_size * sizeof(float)));
    ctx->split_complex_buffer.imagp = static_cast<float*>(malloc(half_size * sizeof(float)));
    
    if (!ctx->split_complex_buffer.realp || !ctx->split_complex_buffer.imagp) {
        fft_destroy_context(ctx);
        return FFT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Precompute window function
    ctx->window_function = static_cast<float*>(malloc(config->fft_size * sizeof(float)));
    if (!ctx->window_function) {
        fft_destroy_context(ctx);
        return FFT_ERROR_MEMORY_ALLOCATION;
    }
    
    generate_window(config->window_type, ctx->window_function, config->fft_size, config->kaiser_beta);
    
    ctx->is_initialized = true;
    *context = ctx;
    
    return FFT_SUCCESS;
}

void fft_destroy_context(production_fft_context_t* context) {
    if (!context) return;
    
    if (context->fft_setup) {
        vDSP_destroy_fftsetup(context->fft_setup);
    }
    
    if (context->split_complex_buffer.realp) {
        free(context->split_complex_buffer.realp);
    }
    
    if (context->split_complex_buffer.imagp) {
        free(context->split_complex_buffer.imagp);
    }
    
    if (context->window_function) {
        free(context->window_function);
    }
    
    delete context;
}

int32_t fft_compute_forward(
    production_fft_context_t* context,
    const float* input,
    uint32_t input_size,
    float* real_output,
    float* imag_output) {
    
    if (!context || !input || !real_output || !imag_output || !context->is_initialized) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    if (input_size != context->config.fft_size) {
        return FFT_ERROR_INVALID_SIZE;
    }
    
    // Prepare input data with windowing
    std::vector<float> windowed_input(input_size);
    std::memcpy(windowed_input.data(), input, input_size * sizeof(float));
    
    // Apply window function
    vDSP_vmul(windowed_input.data(), 1, context->window_function, 1, 
              windowed_input.data(), 1, input_size);
    
    // Convert to split complex format for vDSP
    // vDSP expects data in a specific format for real-to-complex FFT
    DSPSplitComplex split_input;
    split_input.realp = context->split_complex_buffer.realp;
    split_input.imagp = context->split_complex_buffer.imagp;
    
    // Pack real input into split complex format (real parts in even indices, imag in odd)
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(windowed_input.data()), 2, 
              &split_input, 1, input_size / 2);
    
    // Perform FFT
    vDSP_fft_zrip(context->fft_setup, &split_input, 1, context->log2_fft_size, kFFTDirection_Forward);
    
    // Copy results
    uint32_t half_size = input_size / 2;
    std::memcpy(real_output, split_input.realp, half_size * sizeof(float));
    std::memcpy(imag_output, split_input.imagp, half_size * sizeof(float));
    
    // Handle DC and Nyquist components (stored specially by vDSP)
    real_output[0] = split_input.realp[0];  // DC component
    imag_output[0] = 0.0f;
    
    if (half_size > 1) {
        real_output[half_size] = split_input.imagp[0];  // Nyquist component
        imag_output[half_size] = 0.0f;
    }
    
    // Normalize if requested
    if (context->config.normalize_output) {
        float scale = 1.0f / input_size;
        vDSP_vsmul(real_output, 1, &scale, real_output, 1, half_size + 1);
        vDSP_vsmul(imag_output, 1, &scale, imag_output, 1, half_size + 1);
    }
    
    return FFT_SUCCESS;
}

int32_t fft_compute_inverse(
    production_fft_context_t* context,
    const float* real_input,
    const float* imag_input,
    uint32_t input_size,
    float* output) {
    
    if (!context || !real_input || !imag_input || !output || !context->is_initialized) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    uint32_t full_size = context->config.fft_size;
    uint32_t half_size = full_size / 2;
    
    if (input_size != half_size + 1) {
        return FFT_ERROR_INVALID_SIZE;
    }
    
    // Prepare split complex input
    DSPSplitComplex split_input;
    split_input.realp = context->split_complex_buffer.realp;
    split_input.imagp = context->split_complex_buffer.imagp;
    
    // Copy input data (handle DC and Nyquist specially)
    std::memcpy(split_input.realp, real_input, half_size * sizeof(float));
    std::memcpy(split_input.imagp, imag_input, half_size * sizeof(float));
    
    // Set DC and Nyquist in vDSP format
    split_input.realp[0] = real_input[0];        // DC
    split_input.imagp[0] = real_input[half_size]; // Nyquist (if exists)
    
    // Perform IFFT
    vDSP_fft_zrip(context->fft_setup, &split_input, 1, context->log2_fft_size, kFFTDirection_Inverse);
    
    // Convert back to real output
    vDSP_ztoc(&split_input, 1, reinterpret_cast<DSPComplex*>(output), 2, half_size);
    
    // Scale by 1/N (vDSP doesn't do this automatically for inverse FFT)
    float scale = 1.0f / full_size;
    vDSP_vsmul(output, 1, &scale, output, 1, full_size);
    
    return FFT_SUCCESS;
}

int32_t fft_compute_spectrum(
    production_fft_context_t* context,
    const float* input,
    uint32_t input_size,
    double sample_rate,
    spectral_result_t* result) {
    
    if (!context || !input || !result || !context->is_initialized) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    uint32_t fft_size = context->config.fft_size;
    uint32_t bin_count = fft_size / 2 + 1;
    
    // Allocate output arrays
    result->magnitude = static_cast<float*>(malloc(bin_count * sizeof(float)));
    result->phase = static_cast<float*>(malloc(bin_count * sizeof(float)));
    result->power = static_cast<float*>(malloc(bin_count * sizeof(float)));
    
    if (!result->magnitude || !result->phase || !result->power) {
        fft_free_spectral_result(result);
        return FFT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate temporary complex output
    std::vector<float> real_output(bin_count);
    std::vector<float> imag_output(bin_count);
    
    // Compute FFT
    int32_t fft_result = fft_compute_forward(context, input, input_size, 
                                           real_output.data(), imag_output.data());
    if (fft_result != FFT_SUCCESS) {
        fft_free_spectral_result(result);
        return fft_result;
    }
    
    // Compute magnitude, phase, and power
    for (uint32_t i = 0; i < bin_count; i++) {
        float real = real_output[i];
        float imag = imag_output[i];
        
        result->magnitude[i] = sqrtf(real * real + imag * imag);
        result->phase[i] = atan2f(imag, real);
        result->power[i] = real * real + imag * imag;
    }
    
    // Fill metadata
    result->bin_count = bin_count;
    result->frequency_resolution = sample_rate / fft_size;
    result->sample_rate = sample_rate;
    
    return FFT_SUCCESS;
}

int32_t fft_apply_window(window_type_t window_type, float* data, uint32_t size, float kaiser_beta) {
    if (!data || size == 0) {
        return FFT_ERROR_INVALID_PARAMS;
    }
    
    std::vector<float> window(size);
    generate_window(window_type, window.data(), size, kaiser_beta);
    
    // Apply window
    vDSP_vmul(data, 1, window.data(), 1, data, 1, size);
    
    return FFT_SUCCESS;
}

// Utility functions
fft_config_t fft_get_default_config(uint32_t fft_size) {
    fft_config_t config;
    config.fft_size = fft_is_power_of_two(fft_size) ? fft_size : fft_next_power_of_two(fft_size);
    config.hop_length = config.fft_size / 4;  // 75% overlap
    config.window_type = WINDOW_HANN;
    config.kaiser_beta = 8.0f;
    config.zero_phase = false;
    config.normalize_output = true;
    return config;
}

uint32_t fft_next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    
    uint32_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

bool fft_is_power_of_two(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void fft_free_spectral_result(spectral_result_t* result) {
    if (!result) return;
    
    if (result->magnitude) {
        free(result->magnitude);
        result->magnitude = nullptr;
    }
    
    if (result->phase) {
        free(result->phase);
        result->phase = nullptr;
    }
    
    if (result->power) {
        free(result->power);
        result->power = nullptr;
    }
    
    result->bin_count = 0;
}

const char* fft_get_error_string(int32_t error_code) {
    switch (error_code) {
        case FFT_SUCCESS: return "Success";
        case FFT_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case FFT_ERROR_INVALID_SIZE: return "Invalid FFT size (must be power of 2)";
        case FFT_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case FFT_ERROR_ACCELERATE_FAILURE: return "Accelerate framework failure";
        default: return "Unknown error";
    }
}

// C++ Interface Implementation
namespace ProductionFFT {

FFTProcessor::FFTProcessor(const fft_config_t& config) 
    : context_(nullptr), config_(config), window_precomputed_(false), zero_padding_enabled_(false) {
    
    int32_t result = fft_create_context(&config_, &context_);
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("Failed to create FFT context: " + std::string(fft_get_error_string(result)));
    }
    
    precomputeWindow();
}

FFTProcessor::~FFTProcessor() {
    if (context_) {
        fft_destroy_context(context_);
    }
}

std::vector<std::complex<float>> FFTProcessor::computeFFT(const std::vector<float>& input) {
    std::vector<float> input_copy = input;
    ensureValidSize(input_copy);
    
    uint32_t output_size = config_.fft_size / 2 + 1;
    std::vector<float> real_output(output_size);
    std::vector<float> imag_output(output_size);
    
    int32_t result = fft_compute_forward(context_, input_copy.data(), input_copy.size(),
                                       real_output.data(), imag_output.data());
    
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("FFT computation failed: " + std::string(fft_get_error_string(result)));
    }
    
    return split_to_interleaved(real_output.data(), imag_output.data(), output_size);
}

std::vector<float> FFTProcessor::computeIFFT(const std::vector<std::complex<float>>& input) {
    std::vector<float> real_input(input.size());
    std::vector<float> imag_input(input.size());
    
    interleaved_to_split(input, real_input.data(), imag_input.data());
    
    std::vector<float> output(config_.fft_size);
    
    int32_t result = fft_compute_inverse(context_, real_input.data(), imag_input.data(),
                                       input.size(), output.data());
    
    if (result != FFT_SUCCESS) {
        throw std::runtime_error("IFFT computation failed: " + std::string(fft_get_error_string(result)));
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

void FFTProcessor::ensureValidSize(std::vector<float>& data) {
    if (data.size() < config_.fft_size) {
        if (zero_padding_enabled_) {
            data.resize(config_.fft_size, 0.0f);
        } else {
            throw std::runtime_error("Input size too small for FFT");
        }
    } else if (data.size() > config_.fft_size) {
        data.resize(config_.fft_size);
    }
}

void FFTProcessor::precomputeWindow() {
    window_function_.resize(config_.fft_size);
    generate_window(config_.window_type, window_function_.data(), 
                   config_.fft_size, config_.kaiser_beta);
    window_precomputed_ = true;
}

} // namespace ProductionFFT