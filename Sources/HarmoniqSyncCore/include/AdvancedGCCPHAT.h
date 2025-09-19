#pragma once

#include "ProductionFFT.h"
#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

// Frequency weighting types for GCC-PHAT
typedef enum {
    WEIGHTING_PHAT = 0,      // Phase Transform (standard PHAT)
    WEIGHTING_ROTH,          // Roth weighting
    WEIGHTING_SCOT,          // Smoothed Coherence Transform
    WEIGHTING_ML,            // Maximum Likelihood
    WEIGHTING_HT,            // Hannan-Thomson
    WEIGHTING_NONE           // No weighting (basic cross-correlation)
} gcc_weighting_type_t;

// Multi-scale analysis configuration
typedef struct {
    uint32_t num_scales;              // Number of scales to analyze
    uint32_t* fft_sizes;              // FFT sizes for each scale
    float* scale_weights;             // Relative weights for each scale
    bool enable_consensus;            // Use consensus across scales
    float consensus_threshold;        // Agreement threshold for consensus
} multiscale_config_t;

// Enhanced GCC-PHAT parameters
typedef struct {
    uint32_t fft_size;                    // Primary FFT size
    uint32_t hop_length;                  // Hop length for overlapping analysis
    double max_offset_seconds;            // Maximum offset to search
    gcc_weighting_type_t weighting_type;  // Frequency weighting method
    
    // Sub-sample accuracy
    bool enable_subsample_accuracy;       // Enable parabolic interpolation
    uint32_t interpolation_factor;        // Upsampling factor for fine resolution
    
    // Frequency filtering
    bool enable_prefiltering;             // Apply frequency filtering
    double prefilter_low_hz;              // High-pass frequency
    double prefilter_high_hz;             // Low-pass frequency
    
    // Multi-scale analysis
    bool enable_multiscale;               // Enable multi-scale analysis
    multiscale_config_t multiscale_config; // Multi-scale configuration
    
    // Coherence analysis
    bool enable_coherence_analysis;       // Compute coherence for confidence
    uint32_t coherence_window_size;       // Window size for coherence
    
    // Advanced options
    float noise_floor_db;                 // Noise floor threshold in dB
    bool enable_bias_correction;          // Correct for windowing bias
    window_type_t window_type;            // Window function type
} advanced_gcc_phat_params_t;

// Enhanced alignment result
typedef struct {
    double coarse_offset_seconds;         // Coarse offset (sample accurate)
    double fine_offset_seconds;           // Fine offset (sub-sample accurate)
    double confidence_score;              // Overall confidence [0-1]
    double coherence_score;               // Coherence-based confidence [0-1]
    double snr_db;                        // Signal-to-noise ratio in dB
    
    // Multi-scale results
    uint32_t num_scale_results;           // Number of scale results
    double* scale_offsets;                // Offset from each scale
    double* scale_confidences;            // Confidence from each scale
    
    // Quality metrics
    double peak_sharpness;                // Peak sharpness metric
    double spurious_peak_ratio;           // Ratio of second peak to main peak
    bool is_reliable;                     // Overall reliability assessment
    
    int32_t error_code;                   // Error code
} advanced_alignment_result_t;

// Context for advanced GCC-PHAT processor
typedef struct advanced_gcc_phat_context advanced_gcc_phat_context_t;

// Context management
int32_t advanced_gcc_phat_create_context(
    const advanced_gcc_phat_params_t* params,
    advanced_gcc_phat_context_t** context
);

void advanced_gcc_phat_destroy_context(advanced_gcc_phat_context_t* context);

// Main alignment function
int32_t advanced_gcc_phat_compute_alignment(
    advanced_gcc_phat_context_t* context,
    const float* reference_audio,
    const float* target_audio,
    uint32_t audio_length,
    double sample_rate,
    advanced_alignment_result_t* result
);

// Utility functions
advanced_gcc_phat_params_t advanced_gcc_phat_get_default_params(void);
void advanced_gcc_phat_free_result(advanced_alignment_result_t* result);
const char* advanced_gcc_phat_get_error_string(int32_t error_code);

#ifdef __cplusplus
}

// C++ Interface for Advanced GCC-PHAT
namespace AdvancedGCCPHAT {

class AdvancedAligner {
public:
    explicit AdvancedAligner(const advanced_gcc_phat_params_t& params);
    ~AdvancedAligner();
    
    // Main alignment methods
    advanced_alignment_result_t align(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate
    );
    
    advanced_alignment_result_t alignWithPreprocessing(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate
    );
    
    // Multi-scale alignment
    advanced_alignment_result_t multiscaleAlign(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate
    );
    
    // Configuration
    void updateParams(const advanced_gcc_phat_params_t& params);
    advanced_gcc_phat_params_t getParams() const { return params_; }
    
    // Advanced features
    std::vector<float> computeCoherence(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate
    );
    
    std::vector<float> applyFrequencyWeighting(
        const std::vector<std::complex<float>>& cross_spectrum,
        gcc_weighting_type_t weighting_type,
        double sample_rate
    );
    
    double computeSubSampleOffset(
        const std::vector<float>& correlation,
        uint32_t peak_index,
        double sample_rate
    );
    
private:
    advanced_gcc_phat_context_t* context_;
    advanced_gcc_phat_params_t params_;
    std::unique_ptr<ProductionFFT::FFTProcessor> fft_processor_;
    
    // Internal processing methods
    std::vector<float> preprocessAudio(const std::vector<float>& audio, double sample_rate);
    std::vector<std::complex<float>> computeWeightedCrossSpectrum(
        const std::vector<std::complex<float>>& ref_fft,
        const std::vector<std::complex<float>>& target_fft,
        gcc_weighting_type_t weighting_type,
        double sample_rate
    );
    
    double findSubSamplePeak(
        const std::vector<float>& correlation,
        double sample_rate,
        double max_offset
    );
    
    double computeCoherenceConfidence(
        const std::vector<std::complex<float>>& ref_fft,
        const std::vector<std::complex<float>>& target_fft
    );
    
    void assessReliability(advanced_alignment_result_t& result);
};

// Utility classes
class FrequencyWeighter {
public:
    static std::vector<std::complex<float>> applyPHAT(
        const std::vector<std::complex<float>>& cross_spectrum
    );
    
    static std::vector<std::complex<float>> applyROTH(
        const std::vector<std::complex<float>>& cross_spectrum,
        const std::vector<std::complex<float>>& ref_spectrum,
        const std::vector<std::complex<float>>& target_spectrum
    );
    
    static std::vector<std::complex<float>> applySCOT(
        const std::vector<std::complex<float>>& cross_spectrum,
        const std::vector<std::complex<float>>& ref_spectrum,
        const std::vector<std::complex<float>>& target_spectrum,
        double smoothing_factor = 0.1
    );
    
    static std::vector<std::complex<float>> applyML(
        const std::vector<std::complex<float>>& cross_spectrum,
        const std::vector<std::complex<float>>& ref_spectrum,
        const std::vector<std::complex<float>>& target_spectrum,
        double noise_variance = 1e-6
    );
};

class SubSampleEstimator {
public:
    // Parabolic interpolation for sub-sample peak estimation
    static double parabolicInterpolation(
        const std::vector<float>& data,
        uint32_t peak_index
    );
    
    // Gaussian interpolation
    static double gaussianInterpolation(
        const std::vector<float>& data,
        uint32_t peak_index
    );
    
    // Sinc interpolation using zero-padding
    static double sincInterpolation(
        const std::vector<float>& correlation,
        uint32_t peak_index,
        uint32_t upsample_factor
    );
};

class CoherenceAnalyzer {
public:
    CoherenceAnalyzer(uint32_t fft_size, uint32_t overlap = 0);
    
    std::vector<float> computeMagnitudeSquaredCoherence(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        double sample_rate
    );
    
    double computeAverageCoherence(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        double sample_rate,
        double freq_low = 0.0,
        double freq_high = -1.0  // -1 means Nyquist
    );
    
private:
    uint32_t fft_size_;
    uint32_t overlap_;
    std::unique_ptr<ProductionFFT::FFTProcessor> fft_processor_;
};

} // namespace AdvancedGCCPHAT

#endif