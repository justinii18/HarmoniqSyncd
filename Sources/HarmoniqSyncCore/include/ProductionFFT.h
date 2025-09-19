#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <complex>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for C interface
typedef struct production_fft_context production_fft_context_t;

// Window types for spectral analysis
typedef enum {
    WINDOW_HANN = 0,
    WINDOW_HAMMING,
    WINDOW_BLACKMAN,
    WINDOW_KAISER,
    WINDOW_RECTANGULAR
} window_type_t;

// FFT configuration
typedef struct {
    uint32_t fft_size;              // FFT size (must be power of 2)
    uint32_t hop_length;            // Overlap hop length
    window_type_t window_type;      // Window function type
    float kaiser_beta;              // Kaiser window beta parameter
    bool zero_phase;                // Zero-phase windowing
    bool normalize_output;          // Normalize FFT output
} fft_config_t;

// Spectral analysis results
typedef struct {
    float* magnitude;               // Magnitude spectrum
    float* phase;                   // Phase spectrum
    float* power;                   // Power spectrum
    uint32_t bin_count;             // Number of frequency bins
    double frequency_resolution;    // Hz per bin
    double sample_rate;             // Original sample rate
} spectral_result_t;

// Error codes
#define FFT_SUCCESS 0
#define FFT_ERROR_INVALID_PARAMS -1
#define FFT_ERROR_INVALID_SIZE -2
#define FFT_ERROR_MEMORY_ALLOCATION -3
#define FFT_ERROR_ACCELERATE_FAILURE -4

// Context management
int32_t fft_create_context(const fft_config_t* config, production_fft_context_t** context);
void fft_destroy_context(production_fft_context_t* context);

// FFT operations
int32_t fft_compute_forward(
    production_fft_context_t* context,
    const float* input,
    uint32_t input_size,
    float* real_output,
    float* imag_output
);

int32_t fft_compute_inverse(
    production_fft_context_t* context,
    const float* real_input,
    const float* imag_input,
    uint32_t input_size,
    float* output
);

// Spectral analysis
int32_t fft_compute_spectrum(
    production_fft_context_t* context,
    const float* input,
    uint32_t input_size,
    double sample_rate,
    spectral_result_t* result
);

// Windowing functions
int32_t fft_apply_window(
    window_type_t window_type,
    float* data,
    uint32_t size,
    float kaiser_beta
);

// Utility functions
fft_config_t fft_get_default_config(uint32_t fft_size);
uint32_t fft_next_power_of_two(uint32_t n);
bool fft_is_power_of_two(uint32_t n);
void fft_free_spectral_result(spectral_result_t* result);
const char* fft_get_error_string(int32_t error_code);

// Advanced operations
int32_t fft_compute_cross_correlation(
    production_fft_context_t* context,
    const float* signal1,
    const float* signal2,
    uint32_t size,
    float* correlation_output
);

int32_t fft_compute_convolution(
    production_fft_context_t* context,
    const float* signal1,
    const float* signal2,
    uint32_t size,
    float* convolution_output
);

#ifdef __cplusplus
}

// C++ interface for easier usage
namespace ProductionFFT {

class FFTProcessor {
public:
    FFTProcessor(const fft_config_t& config);
    ~FFTProcessor();
    
    // Core FFT operations
    std::vector<std::complex<float>> computeFFT(const std::vector<float>& input);
    std::vector<float> computeIFFT(const std::vector<std::complex<float>>& input);
    
    // Spectral analysis
    spectral_result_t computeSpectrum(const std::vector<float>& input, double sample_rate);
    
    // Audio processing operations
    std::vector<std::complex<float>> applyPHATWeighting(const std::vector<std::complex<float>>& cross_spectrum);
    std::vector<float> computeCrossCorrelation(const std::vector<float>& signal1, const std::vector<float>& signal2);
    
    // Window functions
    std::vector<float> applyWindow(const std::vector<float>& input, window_type_t window_type = WINDOW_HANN);
    
    // Configuration
    void updateConfig(const fft_config_t& config);
    fft_config_t getConfig() const { return config_; }
    
    // Performance optimizations
    void precomputeWindow();
    void enableZeroPadding(bool enable) { zero_padding_enabled_ = enable; }
    
private:
    production_fft_context_t* context_;
    fft_config_t config_;
    std::vector<float> window_function_;
    bool window_precomputed_;
    bool zero_padding_enabled_;
    
    void ensureValidSize(std::vector<float>& data);
    void applyPrecomputedWindow(std::vector<float>& data);
};

// Utility classes
class SpectralAnalyzer {
public:
    SpectralAnalyzer(uint32_t fft_size = 2048, double sample_rate = 48000.0);
    
    // STFT (Short-Time Fourier Transform)
    std::vector<std::vector<std::complex<float>>> computeSTFT(
        const std::vector<float>& audio,
        uint32_t hop_length = 0
    );
    
    // Mel-scale analysis
    std::vector<float> computeMelSpectrum(const std::vector<float>& audio, uint32_t n_mels = 128);
    
    // Spectral features
    float computeSpectralCentroid(const std::vector<float>& magnitude_spectrum, double sample_rate);
    float computeSpectralRolloff(const std::vector<float>& magnitude_spectrum, double sample_rate, float percentile = 0.85f);
    float computeZeroCrossingRate(const std::vector<float>& audio);
    
private:
    FFTProcessor fft_processor_;
    double sample_rate_;
    std::vector<float> mel_filter_bank_;
    
    void initializeMelFilterBank(uint32_t n_mels, uint32_t fft_size);
};

} // namespace ProductionFFT

#endif