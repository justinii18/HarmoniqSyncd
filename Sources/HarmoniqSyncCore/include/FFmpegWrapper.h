#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Audio format information
typedef struct {
    const char* filename;
    const char* format_name;
    const char* codec_name;
    int64_t duration_ms;
    int sample_rate;
    int channels;
    int bits_per_sample;
    bool is_float;
} ffmpeg_audio_info_t;

// Audio buffer for decoded samples
typedef struct {
    float* samples;           // Interleaved audio samples
    uint32_t sample_count;    // Total samples (all channels)
    uint32_t frame_count;     // Number of audio frames
    int sample_rate;
    int channels;
    int64_t timestamp_ms;     // Timestamp in milliseconds
} ffmpeg_audio_buffer_t;

// Decoder context
typedef struct ffmpeg_decoder_context ffmpeg_decoder_context_t;

// Error codes
#define FFMPEG_SUCCESS 0
#define FFMPEG_ERROR_FILE_NOT_FOUND -1
#define FFMPEG_ERROR_UNSUPPORTED_FORMAT -2
#define FFMPEG_ERROR_NO_AUDIO_STREAM -3
#define FFMPEG_ERROR_DECODER_INIT -4
#define FFMPEG_ERROR_MEMORY_ALLOCATION -5
#define FFMPEG_ERROR_READ_ERROR -6
#define FFMPEG_ERROR_END_OF_FILE -7

// Initialization and cleanup
int32_t ffmpeg_initialize(void);
void ffmpeg_cleanup(void);

// Audio information
int32_t ffmpeg_get_audio_info(const char* filename, ffmpeg_audio_info_t* info);

// Decoder management
int32_t ffmpeg_create_decoder(const char* filename, ffmpeg_decoder_context_t** context);
void ffmpeg_destroy_decoder(ffmpeg_decoder_context_t* context);

// Audio decoding
int32_t ffmpeg_decode_audio_chunk(
    ffmpeg_decoder_context_t* context,
    ffmpeg_audio_buffer_t* buffer,
    uint32_t max_frames
);

// Seeking
int32_t ffmpeg_seek_to_timestamp(ffmpeg_decoder_context_t* context, int64_t timestamp_ms);

// Utility functions
void ffmpeg_free_audio_buffer(ffmpeg_audio_buffer_t* buffer);
const char* ffmpeg_get_error_string(int32_t error_code);

#ifdef __cplusplus
}
#endif