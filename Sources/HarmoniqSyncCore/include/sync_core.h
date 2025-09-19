#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version and initialization
const char *sync_core_version(void);
int32_t sync_core_initialize(void);

// Audio buffer structure for C interface
typedef struct {
    const float *samples;
    uint32_t sample_count;
    double sample_rate;
    uint32_t channel_count;
} sync_audio_buffer_t;

// Alignment result structure
typedef struct {
    double coarse_offset_seconds;
    double refined_offset_seconds;
    double drift_rate;
    double confidence_score;
    uint32_t num_anchor_points;
    bool has_drift;
    int32_t error_code;
} sync_alignment_result_t;

// GCC-PHAT parameters
typedef struct {
    uint32_t fft_size;
    uint32_t hop_length;
    double max_offset_seconds;
    double frequency_weight_alpha;
    bool enable_prefiltering;
    double prefilter_low_hz;
    double prefilter_high_hz;
} sync_gcc_phat_params_t;

// DTW parameters
typedef struct {
    uint32_t max_warp_samples;
    double constraint_radius;
    double step_pattern_penalty;
    uint32_t min_path_length;
    bool enable_slope_constraint;
} sync_dtw_params_t;

// RANSAC parameters
typedef struct {
    uint32_t max_iterations;
    double inlier_threshold_seconds;
    uint32_t min_inliers;
    double consensus_threshold;
    uint32_t sample_size;
} sync_ransac_params_t;

// Main alignment functions
int32_t sync_compute_gcc_phat_offset(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_gcc_phat_params_t *params,
    double *offset_seconds,
    double *confidence
);

int32_t sync_compute_dtw_alignment(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_dtw_params_t *params,
    double initial_offset_seconds,
    sync_alignment_result_t *result
);

int32_t sync_estimate_drift_ransac(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_ransac_params_t *params,
    double initial_offset_seconds,
    sync_alignment_result_t *result
);

// Combined alignment pipeline
int32_t sync_align_audio_signals(
    const sync_audio_buffer_t *reference,
    const sync_audio_buffer_t *target,
    const sync_gcc_phat_params_t *gcc_params,
    const sync_dtw_params_t *dtw_params,
    const sync_ransac_params_t *ransac_params,
    sync_alignment_result_t *result
);

// Default parameter getters
sync_gcc_phat_params_t sync_get_default_gcc_phat_params(double sample_rate);
sync_dtw_params_t sync_get_default_dtw_params(double sample_rate);
sync_ransac_params_t sync_get_default_ransac_params(void);

// Error codes
#define SYNC_SUCCESS 0
#define SYNC_ERROR_INVALID_PARAMS -1
#define SYNC_ERROR_INSUFFICIENT_DATA -2
#define SYNC_ERROR_MEMORY_ALLOCATION -3
#define SYNC_ERROR_FFT_FAILURE -4
#define SYNC_ERROR_CONVERGENCE_FAILURE -5
#define SYNC_ERROR_NO_ALIGNMENT_FOUND -6

#ifdef __cplusplus
}
#endif