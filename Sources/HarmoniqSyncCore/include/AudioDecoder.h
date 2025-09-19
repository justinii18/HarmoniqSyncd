#pragma once

#include "FFmpegWrapper.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// High-level audio decoder interface
typedef struct audio_decoder_context audio_decoder_context_t;

// Audio configuration
typedef struct {
    int target_sample_rate;    // Desired output sample rate (0 = keep original)
    int target_channels;       // Desired output channels (0 = keep original)
    bool normalize_audio;      // Apply audio normalization
    bool apply_prefilter;      // Apply low/high pass filtering
    double prefilter_low_hz;   // High-pass filter frequency
    double prefilter_high_hz;  // Low-pass filter frequency
} audio_decoder_config_t;

// Audio metadata
typedef struct {
    const char* filename;
    const char* format_name;
    const char* codec_name;
    int64_t duration_ms;
    int original_sample_rate;
    int original_channels;
    int output_sample_rate;
    int output_channels;
    uint64_t total_samples;
    bool has_variable_bitrate;
} audio_metadata_t;

// Chunk reading result
typedef struct {
    float* samples;            // Interleaved audio samples
    uint32_t sample_count;     // Total samples (all channels)
    uint32_t frame_count;      // Number of audio frames
    int64_t timestamp_ms;      // Timestamp in milliseconds
    bool is_end_of_file;       // True if this is the last chunk
} audio_chunk_t;

// Error codes
#define AUDIO_DECODER_SUCCESS 0
#define AUDIO_DECODER_ERROR_INVALID_PARAMS -1
#define AUDIO_DECODER_ERROR_FILE_ERROR -2
#define AUDIO_DECODER_ERROR_UNSUPPORTED -3
#define AUDIO_DECODER_ERROR_OUT_OF_MEMORY -4
#define AUDIO_DECODER_ERROR_DECODE_FAILED -5

// Decoder management
int32_t audio_decoder_create(
    const char* filename,
    const audio_decoder_config_t* config,
    audio_decoder_context_t** context
);

void audio_decoder_destroy(audio_decoder_context_t* context);

// Audio information
int32_t audio_decoder_get_metadata(
    const char* filename,
    audio_metadata_t* metadata
);

int32_t audio_decoder_get_context_metadata(
    audio_decoder_context_t* context,
    audio_metadata_t* metadata
);

// Streaming audio reading
int32_t audio_decoder_read_chunk(
    audio_decoder_context_t* context,
    uint32_t max_frames,
    audio_chunk_t* chunk
);

// Seeking
int32_t audio_decoder_seek(
    audio_decoder_context_t* context,
    int64_t timestamp_ms
);

// Utility functions
void audio_decoder_free_chunk(audio_chunk_t* chunk);
audio_decoder_config_t audio_decoder_get_default_config(void);
const char* audio_decoder_get_error_string(int32_t error_code);

// Batch processing
int32_t audio_decoder_read_entire_file(
    const char* filename,
    const audio_decoder_config_t* config,
    float** samples,
    uint64_t* sample_count,
    audio_metadata_t* metadata
);

void audio_decoder_free_samples(float* samples);

#ifdef __cplusplus
}
#endif