#include "include/AudioDecoder.h"
#include "include/FFmpegWrapper.h"
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// Internal context structure
struct audio_decoder_context {
    ffmpeg_decoder_context_t* ffmpeg_ctx;
    audio_decoder_config_t config;
    audio_metadata_t metadata;
    bool is_initialized;
    
    // Audio processing state
    std::vector<float> filter_history;
    double rms_accumulator;
    uint64_t rms_sample_count;
    
    audio_decoder_context() :
        ffmpeg_ctx(nullptr),
        is_initialized(false),
        rms_accumulator(0.0),
        rms_sample_count(0) {}
};

namespace {
    // Simple IIR filter for pre-filtering
    class SimpleFilter {
    public:
        SimpleFilter(double cutoff_hz, double sample_rate, bool is_highpass) {
            // Simple one-pole filter
            double rc = 1.0 / (2.0 * M_PI * cutoff_hz);
            double dt = 1.0 / sample_rate;
            alpha = dt / (rc + dt);
            
            if (is_highpass) {
                alpha = rc / (rc + dt);
            }
            
            prev_input = 0.0;
            prev_output = 0.0;
        }
        
        float process(float input) {
            float output = alpha * input + (1.0f - alpha) * prev_output;
            if (alpha < 0.5f) { // High-pass
                output = input - prev_input + (1.0f - alpha) * prev_output;
            }
            
            prev_input = input;
            prev_output = output;
            return output;
        }
        
    private:
        float alpha;
        float prev_input;
        float prev_output;
    };
    
    // Apply audio normalization
    void normalize_audio(float* samples, uint32_t sample_count, float target_rms = 0.1f) {
        if (!samples || sample_count == 0) return;
        
        // Calculate RMS
        double rms = 0.0;
        for (uint32_t i = 0; i < sample_count; i++) {
            rms += samples[i] * samples[i];
        }
        rms = std::sqrt(rms / sample_count);
        
        if (rms > 1e-6) { // Avoid division by zero
            float scale = target_rms / static_cast<float>(rms);
            // Limit scaling to prevent clipping
            scale = std::min(scale, 1.0f);
            
            for (uint32_t i = 0; i < sample_count; i++) {
                samples[i] *= scale;
            }
        }
    }
    
    // Apply pre-filtering
    void apply_prefilter(float* samples, uint32_t sample_count, int channels,
                        double low_hz, double high_hz, double sample_rate,
                        std::vector<float>& filter_state) {
        
        if (!samples || sample_count == 0) return;
        
        // Ensure filter state is properly sized
        if (filter_state.size() < static_cast<size_t>(channels * 4)) {
            filter_state.resize(channels * 4, 0.0f);
        }
        
        // Create filters per channel
        std::vector<SimpleFilter> high_pass_filters;
        std::vector<SimpleFilter> low_pass_filters;
        
        for (int ch = 0; ch < channels; ch++) {
            high_pass_filters.emplace_back(low_hz, sample_rate, true);
            low_pass_filters.emplace_back(high_hz, sample_rate, false);
        }
        
        // Process samples
        for (uint32_t i = 0; i < sample_count; i += channels) {
            for (int ch = 0; ch < channels; ch++) {
                float sample = samples[i + ch];
                
                // Apply high-pass filter
                if (low_hz > 0) {
                    sample = high_pass_filters[ch].process(sample);
                }
                
                // Apply low-pass filter
                if (high_hz > 0 && high_hz < sample_rate / 2) {
                    sample = low_pass_filters[ch].process(sample);
                }
                
                samples[i + ch] = sample;
            }
        }
    }
}

// Implementation
int32_t audio_decoder_create(
    const char* filename,
    const audio_decoder_config_t* config,
    audio_decoder_context_t** context) {
    
    if (!filename || !context) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    // Initialize FFmpeg if needed
    ffmpeg_initialize();
    
    // Create context
    auto* ctx = new audio_decoder_context();
    if (!ctx) {
        return AUDIO_DECODER_ERROR_OUT_OF_MEMORY;
    }
    
    // Store configuration
    ctx->config = config ? *config : audio_decoder_get_default_config();
    
    // Create FFmpeg decoder
    int32_t result = ffmpeg_create_decoder(filename, &ctx->ffmpeg_ctx);
    if (result != FFMPEG_SUCCESS) {
        delete ctx;
        return AUDIO_DECODER_ERROR_FILE_ERROR;
    }
    
    // Get metadata
    ffmpeg_audio_info_t ffmpeg_info;
    result = ffmpeg_get_audio_info(filename, &ffmpeg_info);
    if (result != FFMPEG_SUCCESS) {
        audio_decoder_destroy(ctx);
        return AUDIO_DECODER_ERROR_FILE_ERROR;
    }
    
    // Fill metadata
    ctx->metadata.filename = filename;
    ctx->metadata.format_name = ffmpeg_info.format_name;
    ctx->metadata.codec_name = ffmpeg_info.codec_name;
    ctx->metadata.duration_ms = ffmpeg_info.duration_ms;
    ctx->metadata.original_sample_rate = ffmpeg_info.sample_rate;
    ctx->metadata.original_channels = ffmpeg_info.channels;
    
    // Determine output format
    ctx->metadata.output_sample_rate = ctx->config.target_sample_rate > 0 ? 
        ctx->config.target_sample_rate : ffmpeg_info.sample_rate;
    ctx->metadata.output_channels = ctx->config.target_channels > 0 ? 
        ctx->config.target_channels : ffmpeg_info.channels;
    
    // Calculate total samples
    ctx->metadata.total_samples = static_cast<uint64_t>(
        (ctx->metadata.duration_ms / 1000.0) * 
        ctx->metadata.output_sample_rate * 
        ctx->metadata.output_channels
    );
    
    ctx->is_initialized = true;
    *context = ctx;
    
    return AUDIO_DECODER_SUCCESS;
}

void audio_decoder_destroy(audio_decoder_context_t* context) {
    if (!context) return;
    
    if (context->ffmpeg_ctx) {
        ffmpeg_destroy_decoder(context->ffmpeg_ctx);
    }
    
    delete context;
}

int32_t audio_decoder_get_metadata(const char* filename, audio_metadata_t* metadata) {
    if (!filename || !metadata) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    ffmpeg_initialize();
    
    ffmpeg_audio_info_t info;
    int32_t result = ffmpeg_get_audio_info(filename, &info);
    if (result != FFMPEG_SUCCESS) {
        return AUDIO_DECODER_ERROR_FILE_ERROR;
    }
    
    // Fill metadata with original file info
    metadata->filename = filename;
    metadata->format_name = info.format_name;
    metadata->codec_name = info.codec_name;
    metadata->duration_ms = info.duration_ms;
    metadata->original_sample_rate = info.sample_rate;
    metadata->original_channels = info.channels;
    metadata->output_sample_rate = info.sample_rate;
    metadata->output_channels = info.channels;
    metadata->total_samples = static_cast<uint64_t>(
        (info.duration_ms / 1000.0) * info.sample_rate * info.channels
    );
    metadata->has_variable_bitrate = false; // TODO: Detect VBR
    
    return AUDIO_DECODER_SUCCESS;
}

int32_t audio_decoder_get_context_metadata(
    audio_decoder_context_t* context,
    audio_metadata_t* metadata) {
    
    if (!context || !metadata || !context->is_initialized) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    *metadata = context->metadata;
    return AUDIO_DECODER_SUCCESS;
}

int32_t audio_decoder_read_chunk(
    audio_decoder_context_t* context,
    uint32_t max_frames,
    audio_chunk_t* chunk) {
    
    if (!context || !chunk || !context->is_initialized) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    // Initialize chunk
    chunk->samples = nullptr;
    chunk->sample_count = 0;
    chunk->frame_count = 0;
    chunk->timestamp_ms = 0;
    chunk->is_end_of_file = false;
    
    // Decode audio from FFmpeg
    ffmpeg_audio_buffer_t ffmpeg_buffer;
    int32_t result = ffmpeg_decode_audio_chunk(context->ffmpeg_ctx, &ffmpeg_buffer, max_frames);
    
    if (result == FFMPEG_ERROR_END_OF_FILE) {
        chunk->is_end_of_file = true;
        return AUDIO_DECODER_SUCCESS;
    }
    
    if (result != FFMPEG_SUCCESS) {
        return AUDIO_DECODER_ERROR_DECODE_FAILED;
    }
    
    // Copy samples and apply processing
    chunk->sample_count = ffmpeg_buffer.sample_count;
    chunk->frame_count = ffmpeg_buffer.frame_count;
    chunk->timestamp_ms = ffmpeg_buffer.timestamp_ms;
    
    chunk->samples = static_cast<float*>(malloc(ffmpeg_buffer.sample_count * sizeof(float)));
    if (!chunk->samples) {
        ffmpeg_free_audio_buffer(&ffmpeg_buffer);
        return AUDIO_DECODER_ERROR_OUT_OF_MEMORY;
    }
    
    std::memcpy(chunk->samples, ffmpeg_buffer.samples, 
               ffmpeg_buffer.sample_count * sizeof(float));
    
    // Apply audio processing
    if (context->config.apply_prefilter) {
        apply_prefilter(chunk->samples, chunk->sample_count, 
                       context->metadata.output_channels,
                       context->config.prefilter_low_hz,
                       context->config.prefilter_high_hz,
                       context->metadata.output_sample_rate,
                       context->filter_history);
    }
    
    if (context->config.normalize_audio) {
        normalize_audio(chunk->samples, chunk->sample_count);
    }
    
    ffmpeg_free_audio_buffer(&ffmpeg_buffer);
    return AUDIO_DECODER_SUCCESS;
}

int32_t audio_decoder_seek(audio_decoder_context_t* context, int64_t timestamp_ms) {
    if (!context || !context->is_initialized) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    int32_t result = ffmpeg_seek_to_timestamp(context->ffmpeg_ctx, timestamp_ms);
    if (result != FFMPEG_SUCCESS) {
        return AUDIO_DECODER_ERROR_DECODE_FAILED;
    }
    
    // Reset filter state after seeking
    context->filter_history.clear();
    context->rms_accumulator = 0.0;
    context->rms_sample_count = 0;
    
    return AUDIO_DECODER_SUCCESS;
}

void audio_decoder_free_chunk(audio_chunk_t* chunk) {
    if (chunk && chunk->samples) {
        free(chunk->samples);
        chunk->samples = nullptr;
        chunk->sample_count = 0;
        chunk->frame_count = 0;
    }
}

audio_decoder_config_t audio_decoder_get_default_config(void) {
    audio_decoder_config_t config;
    config.target_sample_rate = 48000;  // Standard professional sample rate
    config.target_channels = 2;         // Stereo output
    config.normalize_audio = false;     // Don't normalize by default
    config.apply_prefilter = true;      // Apply filtering
    config.prefilter_low_hz = 80.0;     // High-pass at 80Hz
    config.prefilter_high_hz = 18000.0; // Low-pass at 18kHz
    return config;
}

const char* audio_decoder_get_error_string(int32_t error_code) {
    switch (error_code) {
        case AUDIO_DECODER_SUCCESS: return "Success";
        case AUDIO_DECODER_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case AUDIO_DECODER_ERROR_FILE_ERROR: return "File error";
        case AUDIO_DECODER_ERROR_UNSUPPORTED: return "Unsupported format";
        case AUDIO_DECODER_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case AUDIO_DECODER_ERROR_DECODE_FAILED: return "Decode failed";
        default: return "Unknown error";
    }
}

int32_t audio_decoder_read_entire_file(
    const char* filename,
    const audio_decoder_config_t* config,
    float** samples,
    uint64_t* sample_count,
    audio_metadata_t* metadata) {
    
    if (!filename || !samples || !sample_count) {
        return AUDIO_DECODER_ERROR_INVALID_PARAMS;
    }
    
    audio_decoder_context_t* context = nullptr;
    int32_t result = audio_decoder_create(filename, config, &context);
    if (result != AUDIO_DECODER_SUCCESS) {
        return result;
    }
    
    // Get metadata if requested
    if (metadata) {
        audio_decoder_get_context_metadata(context, metadata);
    }
    
    // Read all chunks
    std::vector<float> all_samples;
    const uint32_t chunk_frames = 8192; // Read in 8K frame chunks
    
    while (true) {
        audio_chunk_t chunk;
        result = audio_decoder_read_chunk(context, chunk_frames, &chunk);
        
        if (result != AUDIO_DECODER_SUCCESS) {
            audio_decoder_destroy(context);
            return result;
        }
        
        if (chunk.is_end_of_file) {
            break;
        }
        
        // Append samples
        all_samples.insert(all_samples.end(), chunk.samples, 
                          chunk.samples + chunk.sample_count);
        
        audio_decoder_free_chunk(&chunk);
    }
    
    audio_decoder_destroy(context);
    
    // Copy to output
    if (!all_samples.empty()) {
        *sample_count = all_samples.size();
        *samples = static_cast<float*>(malloc(all_samples.size() * sizeof(float)));
        if (!*samples) {
            return AUDIO_DECODER_ERROR_OUT_OF_MEMORY;
        }
        
        std::memcpy(*samples, all_samples.data(), all_samples.size() * sizeof(float));
    } else {
        *samples = nullptr;
        *sample_count = 0;
    }
    
    return AUDIO_DECODER_SUCCESS;
}

void audio_decoder_free_samples(float* samples) {
    if (samples) {
        free(samples);
    }
}