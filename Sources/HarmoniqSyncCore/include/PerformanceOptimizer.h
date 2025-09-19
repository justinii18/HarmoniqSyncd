#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

// Performance configuration
typedef struct {
    // Threading options
    uint32_t num_threads;                    // Number of worker threads (0 = auto-detect)
    bool enable_parallel_fft;               // Enable parallel FFT computation
    bool enable_parallel_dtw;               // Enable parallel DTW computation
    bool enable_parallel_ransac;            // Enable parallel RANSAC iterations
    bool enable_concurrent_pairs;           // Enable concurrent audio pair processing
    
    // SIMD options
    bool enable_neon_optimization;          // Enable ARM64 NEON SIMD
    bool enable_vectorized_audio_ops;       // Enable vectorized audio operations
    bool enable_vectorized_correlation;     // Enable vectorized correlation
    bool enable_vectorized_dtw_distance;    // Enable vectorized DTW distance
    
    // Memory optimization
    bool enable_streaming_processing;       // Enable streaming for large files
    uint32_t chunk_size_samples;            // Chunk size for streaming
    uint32_t buffer_pool_size;              // Size of audio buffer pool
    bool enable_fft_caching;                // Enable FFT result caching
    uint32_t fft_cache_size;                // Maximum cached FFT results
    
    // Performance tuning
    uint32_t dtw_parallel_threshold;        // Minimum matrix size for parallel DTW
    uint32_t ransac_parallel_threshold;     // Minimum iterations for parallel RANSAC
    double memory_limit_gb;                 // Memory usage limit in GB
    bool enable_early_termination;         // Enable early termination optimizations
} performance_config_t;

// Performance statistics
typedef struct {
    // Timing information
    double total_processing_time_ms;        // Total processing time
    double fft_time_ms;                     // Time spent in FFT
    double gcc_phat_time_ms;                // Time spent in GCC-PHAT
    double dtw_time_ms;                     // Time spent in DTW
    double ransac_time_ms;                  // Time spent in RANSAC
    
    // Threading statistics
    uint32_t threads_used;                  // Number of threads actually used
    double thread_efficiency;              // Thread utilization efficiency
    double parallel_speedup;               // Speedup factor from parallelization
    
    // Memory statistics
    uint64_t peak_memory_bytes;             // Peak memory usage
    uint32_t cache_hits;                    // FFT cache hits
    uint32_t cache_misses;                  // FFT cache misses
    double cache_hit_ratio;                 // Cache hit ratio
    
    // SIMD statistics
    bool simd_used;                         // Whether SIMD was used
    double simd_speedup;                    // Speedup from SIMD optimization
    uint32_t vectorized_operations;        // Number of vectorized operations
} performance_stats_t;

// Performance context
typedef struct performance_context performance_context_t;

// Context management
int32_t perf_create_context(const performance_config_t* config, performance_context_t** context);
void perf_destroy_context(performance_context_t* context);
performance_config_t perf_get_default_config(void);

// Performance monitoring
void perf_start_timing(performance_context_t* context, const char* operation);
void perf_end_timing(performance_context_t* context, const char* operation);
performance_stats_t perf_get_statistics(performance_context_t* context);
void perf_reset_statistics(performance_context_t* context);

#ifdef __cplusplus
}

// C++ Interface for Performance Optimization
namespace PerformanceOptimization {

// Thread pool for parallel processing
class ThreadPool {
public:
    ThreadPool(uint32_t num_threads = 0);
    ~ThreadPool();
    
    // Submit work to thread pool
    template<class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;;
    
    // Wait for all tasks to complete
    void wait();
    
    // Get number of worker threads
    uint32_t getNumThreads() const { return num_threads_; }
    
    // Get thread utilization
    double getUtilization() const;
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    uint32_t num_threads_;
    std::atomic<uint32_t> active_threads_;
    std::atomic<uint64_t> completed_tasks_;
};

// Memory pool for efficient buffer management
template<typename T>
class MemoryPool {
public:
    MemoryPool(size_t buffer_size, size_t pool_size = 16);
    ~MemoryPool();
    
    // Get buffer from pool
    std::unique_ptr<std::vector<T>> getBuffer();
    
    // Return buffer to pool
    void returnBuffer(std::unique_ptr<std::vector<T>> buffer);
    
    // Pool statistics
    size_t getPoolSize() const { return pool_size_; }
    size_t getAvailableBuffers() const;
    
private:
    std::queue<std::unique_ptr<std::vector<T>>> available_buffers_;
    std::mutex pool_mutex_;
    size_t buffer_size_;
    size_t pool_size_;
};

// FFT result cache for performance optimization
class FFTCache {
public:
    FFTCache(size_t max_cache_size = 100);
    ~FFTCache();
    
    // Cache operations
    bool get(const std::vector<float>& input, std::vector<std::complex<float>>& output);
    void put(const std::vector<float>& input, const std::vector<std::complex<float>>& output);
    
    // Cache management
    void clear();
    size_t getSize() const;
    double getHitRatio() const;
    
private:
    struct CacheEntry {
        std::vector<float> input_hash;
        std::vector<std::complex<float>> output;
        uint64_t timestamp;
        uint32_t access_count;
    };
    
    std::unordered_map<std::string, CacheEntry> cache_;
    std::mutex cache_mutex_;
    size_t max_cache_size_;
    std::atomic<uint32_t> cache_hits_;
    std::atomic<uint32_t> cache_misses_;
    
    std::string computeHash(const std::vector<float>& input);
    void evictOldEntries();
};

// SIMD-optimized operations
class SIMDOperations {
public:
    // Check SIMD availability
    static bool isNEONAvailable();
    static bool isAVXAvailable();
    
    // Vectorized audio operations
    static void vectorizedAdd(const float* a, const float* b, float* result, size_t size);
    static void vectorizedMultiply(const float* a, const float* b, float* result, size_t size);
    static void vectorizedScale(const float* input, float scale, float* output, size_t size);
    static float vectorizedDotProduct(const float* a, const float* b, size_t size);
    
    // Vectorized correlation
    static void vectorizedCorrelation(
        const std::vector<std::complex<float>>& a,
        const std::vector<std::complex<float>>& b,
        std::vector<std::complex<float>>& result
    );
    
    // Vectorized DTW distance computation
    static float vectorizedEuclideanDistance(const float* a, const float* b, size_t size);
    static float vectorizedManhattanDistance(const float* a, const float* b, size_t size);
    static float vectorizedCosineDistance(const float* a, const float* b, size_t size);
    
private:
    static bool neon_available_;
    static bool avx_available_;
    static void initializeSIMDSupport();
};

// Parallel processing coordinator
class ParallelProcessor {
public:
    ParallelProcessor(const performance_config_t& config);
    ~ParallelProcessor();
    
    // Parallel FFT processing
    std::vector<std::vector<std::complex<float>>> processParallelFFT(
        const std::vector<std::vector<float>>& inputs
    );
    
    // Parallel DTW distance matrix computation
    std::vector<std::vector<float>> computeParallelDTWMatrix(
        const std::vector<std::vector<float>>& ref_features,
        const std::vector<std::vector<float>>& target_features,
        uint32_t num_threads = 0
    );
    
    // Parallel RANSAC iterations
    template<typename ModelType>
    ModelType performParallelRANSAC(
        const std::vector<typename ModelType::PointType>& points,
        uint32_t iterations,
        typename ModelType::FitFunction fit_func,
        typename ModelType::EvaluateFunction eval_func
    );
    
    // Concurrent audio pair processing
    std::vector<std::future<int32_t>> processAudioPairsConcurrently(
        const std::vector<std::pair<std::vector<float>, std::vector<float>>>& audio_pairs,
        double sample_rate
    );
    
    // Performance monitoring
    performance_stats_t getPerformanceStats() const { return stats_; }
    void resetPerformanceStats();
    
private:
    performance_config_t config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<MemoryPool<float>> float_pool_;
    std::unique_ptr<MemoryPool<std::complex<float>>> complex_pool_;
    std::unique_ptr<FFTCache> fft_cache_;
    mutable performance_stats_t stats_;
    mutable std::mutex stats_mutex_;
    
    void updateStats(const std::string& operation, double time_ms);
    uint32_t getOptimalThreadCount() const;
};

// Streaming processor for large audio files
class StreamingProcessor {
public:
    StreamingProcessor(const performance_config_t& config);
    ~StreamingProcessor();
    
    // Process large audio files in chunks
    template<typename ProcessorType>
    auto processAudioStream(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        ProcessorType processor
    ) -> decltype(processor(std::vector<float>(), std::vector<float>(), sample_rate));
    
    // Streaming GCC-PHAT
    std::vector<double> streamingGCCPHAT(
        const std::vector<float>& reference,
        const std::vector<float>& target,
        double sample_rate,
        uint32_t chunk_size = 0
    );
    
    // Memory usage monitoring
    uint64_t getCurrentMemoryUsage() const;
    uint64_t getPeakMemoryUsage() const { return peak_memory_usage_; }
    
private:
    performance_config_t config_;
    std::unique_ptr<MemoryPool<float>> chunk_pool_;
    mutable uint64_t peak_memory_usage_;
    mutable std::mutex memory_mutex_;
    
    uint32_t getOptimalChunkSize(size_t total_size) const;
    void updateMemoryUsage();
};

// Performance profiler
class PerformanceProfiler {
public:
    PerformanceProfiler();
    ~PerformanceProfiler();
    
    // Timing operations
    void startTimer(const std::string& operation);
    void endTimer(const std::string& operation);
    
    // Scoped timer for RAII timing
    class ScopedTimer {
    public:
        ScopedTimer(PerformanceProfiler& profiler, const std::string& operation);
        ~ScopedTimer();
    private:
        PerformanceProfiler& profiler_;
        std::string operation_;
    };
    
    // Performance reporting
    performance_stats_t generateReport() const;
    void printReport() const;
    void reset();
    
    // Memory tracking
    void recordMemoryUsage(uint64_t bytes);
    void recordCacheAccess(bool hit);
    
private:
    struct TimingData {
        std::chrono::high_resolution_clock::time_point start_time;
        double total_time_ms;
        uint32_t call_count;
    };
    
    std::unordered_map<std::string, TimingData> timers_;
    mutable std::mutex timing_mutex_;
    std::atomic<uint64_t> peak_memory_;
    std::atomic<uint32_t> cache_hits_;
    std::atomic<uint32_t> cache_misses_;
};

} // namespace PerformanceOptimization

#endif
