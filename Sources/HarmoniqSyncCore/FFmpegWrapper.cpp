#include "include/FFmpegWrapper.h"
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

namespace {
    bool g_ffmpeg_initialized = false;
}

// Internal decoder context structure
struct ffmpeg_decoder_context {
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    SwrContext* swr_ctx;
    int audio_stream_index;
    AVPacket* packet;
    AVFrame* frame;
    
    // Resampling parameters
    int target_sample_rate;
    int target_channels;
    enum AVSampleFormat target_format;
    
    ffmpeg_decoder_context() :
        format_ctx(nullptr),
        codec_ctx(nullptr),
        swr_ctx(nullptr),
        audio_stream_index(-1),
        packet(nullptr),
        frame(nullptr),
        target_sample_rate(48000),
        target_channels(2),
        target_format(AV_SAMPLE_FMT_FLT) {}
};

// Implementation
int32_t ffmpeg_initialize(void) {
    if (g_ffmpeg_initialized) {
        return FFMPEG_SUCCESS;
    }
    
    // Initialize FFmpeg
    av_log_set_level(AV_LOG_WARNING);
    
    g_ffmpeg_initialized = true;
    return FFMPEG_SUCCESS;
}

void ffmpeg_cleanup(void) {
    g_ffmpeg_initialized = false;
}

int32_t ffmpeg_get_audio_info(const char* filename, ffmpeg_audio_info_t* info) {
    if (!filename || !info) {
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    AVFormatContext* format_ctx = nullptr;
    
    // Open file
    int ret = avformat_open_input(&format_ctx, filename, nullptr, nullptr);
    if (ret < 0) {
        return FFMPEG_ERROR_FILE_NOT_FOUND;
    }
    
    // Find stream information
    ret = avformat_find_stream_info(format_ctx, nullptr);
    if (ret < 0) {
        avformat_close_input(&format_ctx);
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Find audio stream
    int audio_stream_index = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }
    
    if (audio_stream_index == -1) {
        avformat_close_input(&format_ctx);
        return FFMPEG_ERROR_NO_AUDIO_STREAM;
    }
    
    AVStream* audio_stream = format_ctx->streams[audio_stream_index];
    AVCodecParameters* codecpar = audio_stream->codecpar;
    
    // Fill info structure
    info->filename = filename;
    info->format_name = format_ctx->iformat->name;
    
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    info->codec_name = codec ? codec->name : "unknown";
    
    info->duration_ms = format_ctx->duration / 1000; // Convert from microseconds
    info->sample_rate = codecpar->sample_rate;
    info->channels = codecpar->ch_layout.nb_channels;
    info->bits_per_sample = av_get_bits_per_sample(codecpar->codec_id);
    info->is_float = (codecpar->format == AV_SAMPLE_FMT_FLT || 
                     codecpar->format == AV_SAMPLE_FMT_FLTP ||
                     codecpar->format == AV_SAMPLE_FMT_DBL ||
                     codecpar->format == AV_SAMPLE_FMT_DBLP);
    
    avformat_close_input(&format_ctx);
    return FFMPEG_SUCCESS;
}

int32_t ffmpeg_create_decoder(const char* filename, ffmpeg_decoder_context_t** context) {
    if (!filename || !context) {
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Create context
    auto* ctx = new ffmpeg_decoder_context();
    
    // Open file
    int ret = avformat_open_input(&ctx->format_ctx, filename, nullptr, nullptr);
    if (ret < 0) {
        delete ctx;
        return FFMPEG_ERROR_FILE_NOT_FOUND;
    }
    
    // Find stream information
    ret = avformat_find_stream_info(ctx->format_ctx, nullptr);
    if (ret < 0) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Find audio stream
    for (unsigned int i = 0; i < ctx->format_ctx->nb_streams; i++) {
        if (ctx->format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            ctx->audio_stream_index = i;
            break;
        }
    }
    
    if (ctx->audio_stream_index == -1) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_NO_AUDIO_STREAM;
    }
    
    // Get codec parameters
    AVCodecParameters* codecpar = ctx->format_ctx->streams[ctx->audio_stream_index]->codecpar;
    
    // Find decoder
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Create codec context
    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy codec parameters
    ret = avcodec_parameters_to_context(ctx->codec_ctx, codecpar);
    if (ret < 0) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_DECODER_INIT;
    }
    
    // Open codec
    ret = avcodec_open2(ctx->codec_ctx, codec, nullptr);
    if (ret < 0) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_DECODER_INIT;
    }
    
    // Allocate packet and frame
    ctx->packet = av_packet_alloc();
    ctx->frame = av_frame_alloc();
    if (!ctx->packet || !ctx->frame) {
        ffmpeg_destroy_decoder(ctx);
        return FFMPEG_ERROR_MEMORY_ALLOCATION;
    }
    
    // Setup resampler if needed
    if (ctx->codec_ctx->sample_fmt != ctx->target_format ||
        ctx->codec_ctx->sample_rate != ctx->target_sample_rate ||
        ctx->codec_ctx->ch_layout.nb_channels != ctx->target_channels) {
        
        // Initialize resampler using FFmpeg 7.x API
        AVChannelLayout target_layout;
        if (ctx->target_channels == 1) {
            target_layout = AV_CHANNEL_LAYOUT_MONO;
        } else {
            target_layout = AV_CHANNEL_LAYOUT_STEREO;
        }
        
        int ret_swr = swr_alloc_set_opts2(&ctx->swr_ctx,
                           &target_layout,
                           ctx->target_format,
                           ctx->target_sample_rate,
                           &ctx->codec_ctx->ch_layout,
                           ctx->codec_ctx->sample_fmt,
                           ctx->codec_ctx->sample_rate,
                           0, nullptr);
        
        if (ret_swr < 0 || !ctx->swr_ctx) {
            ffmpeg_destroy_decoder(ctx);
            return FFMPEG_ERROR_DECODER_INIT;
        }
        
        ret = swr_init(ctx->swr_ctx);
        if (ret < 0) {
            ffmpeg_destroy_decoder(ctx);
            return FFMPEG_ERROR_DECODER_INIT;
        }
    }
    
    *context = ctx;
    return FFMPEG_SUCCESS;
}

void ffmpeg_destroy_decoder(ffmpeg_decoder_context_t* context) {
    if (!context) return;
    
    if (context->swr_ctx) {
        swr_free(&context->swr_ctx);
    }
    
    if (context->frame) {
        av_frame_free(&context->frame);
    }
    
    if (context->packet) {
        av_packet_free(&context->packet);
    }
    
    if (context->codec_ctx) {
        avcodec_free_context(&context->codec_ctx);
    }
    
    if (context->format_ctx) {
        avformat_close_input(&context->format_ctx);
    }
    
    delete context;
}

int32_t ffmpeg_decode_audio_chunk(
    ffmpeg_decoder_context_t* context,
    ffmpeg_audio_buffer_t* buffer,
    uint32_t max_frames) {
    
    if (!context || !buffer) {
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Initialize buffer
    buffer->samples = nullptr;
    buffer->sample_count = 0;
    buffer->frame_count = 0;
    buffer->sample_rate = context->target_sample_rate;
    buffer->channels = context->target_channels;
    buffer->timestamp_ms = 0;
    
    // Read packets until we have enough audio data
    std::vector<float> decoded_samples;
    uint32_t frames_decoded = 0;
    
    while (frames_decoded < max_frames) {
        int ret = av_read_frame(context->format_ctx, context->packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                break; // End of file
            }
            return FFMPEG_ERROR_READ_ERROR;
        }
        
        // Skip non-audio packets
        if (context->packet->stream_index != context->audio_stream_index) {
            av_packet_unref(context->packet);
            continue;
        }
        
        // Send packet to decoder
        ret = avcodec_send_packet(context->codec_ctx, context->packet);
        av_packet_unref(context->packet);
        
        if (ret < 0) {
            continue; // Skip problematic packets
        }
        
        // Receive decoded frames
        while (ret >= 0) {
            ret = avcodec_receive_frame(context->codec_ctx, context->frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                return FFMPEG_ERROR_READ_ERROR;
            }
            
            // Convert/resample if needed
            if (context->swr_ctx) {
                // Calculate output frame count
                int out_samples = swr_get_out_samples(context->swr_ctx, context->frame->nb_samples);
                
                // Allocate output buffer
                uint8_t* out_buffer = nullptr;
                ret = av_samples_alloc(&out_buffer, nullptr, context->target_channels,
                                     out_samples, context->target_format, 0);
                if (ret < 0) {
                    av_frame_unref(context->frame);
                    return FFMPEG_ERROR_MEMORY_ALLOCATION;
                }
                
                // Convert samples
                out_samples = swr_convert(context->swr_ctx, &out_buffer, out_samples,
                                        (const uint8_t**)context->frame->data, context->frame->nb_samples);
                
                if (out_samples > 0) {
                    // Copy converted samples
                    float* float_samples = reinterpret_cast<float*>(out_buffer);
                    size_t sample_count = out_samples * context->target_channels;
                    decoded_samples.insert(decoded_samples.end(), float_samples, float_samples + sample_count);
                    frames_decoded += out_samples;
                }
                
                av_freep(&out_buffer);
            } else {
                // Direct copy (already in target format)
                float* float_samples = reinterpret_cast<float*>(context->frame->data[0]);
                size_t sample_count = context->frame->nb_samples * context->target_channels;
                decoded_samples.insert(decoded_samples.end(), float_samples, float_samples + sample_count);
                frames_decoded += context->frame->nb_samples;
            }
            
            av_frame_unref(context->frame);
        }
    }
    
    // Copy results to output buffer
    if (!decoded_samples.empty()) {
        buffer->frame_count = frames_decoded;
        buffer->sample_count = decoded_samples.size();
        buffer->samples = static_cast<float*>(malloc(decoded_samples.size() * sizeof(float)));
        if (!buffer->samples) {
            return FFMPEG_ERROR_MEMORY_ALLOCATION;
        }
        
        std::memcpy(buffer->samples, decoded_samples.data(), decoded_samples.size() * sizeof(float));
        return FFMPEG_SUCCESS;
    }
    
    return FFMPEG_ERROR_END_OF_FILE;
}

int32_t ffmpeg_seek_to_timestamp(ffmpeg_decoder_context_t* context, int64_t timestamp_ms) {
    if (!context) {
        return FFMPEG_ERROR_UNSUPPORTED_FORMAT;
    }
    
    // Convert timestamp to stream time base
    AVStream* stream = context->format_ctx->streams[context->audio_stream_index];
    int64_t seek_target = av_rescale_q(timestamp_ms * 1000, AV_TIME_BASE_Q, stream->time_base);
    
    int ret = av_seek_frame(context->format_ctx, context->audio_stream_index, 
                           seek_target, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        return FFMPEG_ERROR_READ_ERROR;
    }
    
    // Flush codec buffers
    avcodec_flush_buffers(context->codec_ctx);
    
    return FFMPEG_SUCCESS;
}

void ffmpeg_free_audio_buffer(ffmpeg_audio_buffer_t* buffer) {
    if (buffer && buffer->samples) {
        free(buffer->samples);
        buffer->samples = nullptr;
        buffer->sample_count = 0;
        buffer->frame_count = 0;
    }
}

const char* ffmpeg_get_error_string(int32_t error_code) {
    switch (error_code) {
        case FFMPEG_SUCCESS: return "Success";
        case FFMPEG_ERROR_FILE_NOT_FOUND: return "File not found";
        case FFMPEG_ERROR_UNSUPPORTED_FORMAT: return "Unsupported format";
        case FFMPEG_ERROR_NO_AUDIO_STREAM: return "No audio stream found";
        case FFMPEG_ERROR_DECODER_INIT: return "Decoder initialization failed";
        case FFMPEG_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case FFMPEG_ERROR_READ_ERROR: return "Read error";
        case FFMPEG_ERROR_END_OF_FILE: return "End of file";
        default: return "Unknown error";
    }
}