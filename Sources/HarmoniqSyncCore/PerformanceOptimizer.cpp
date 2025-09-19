#include "include/PerformanceOptimizer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <Accelerate/Accelerate.h>
#endif

// Internal context structure
struct performance_context {
    performance_config_t config;
    std::unique_ptr<PerformanceOptimization::PerformanceProfiler> profiler;
    std::unique_ptr<PerformanceOptimization::ParallelProcessor> parallel_processor;
    std::unique_ptr<PerformanceOptimization::StreamingProcessor> streaming_processor;
    bool is_initialized;
    
    performance_context() : is_initialized(false) {}
};

namespace {
    // Get system thread count
    uint32_t get_system_thread_count() {
        uint32_t thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) {
            thread_count = 4; // Fallback to 4 threads
        }
        return thread_count;
    }
    
    // Get system memory size
    uint64_t get_system_memory_size() {
#ifdef __APPLE__
        int mib[2];
        mib[0] = CTL_HW;
        mib[1] = HW_MEMSIZE;
        uint64_t size = 0;
        size_t len = sizeof(size);
        
        if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) {
            return size;
        }
#endif
        return 8ULL * 1024 * 1024 * 1024; // Fallback to 8GB
    }
}

// C Interface Implementation
int32_t perf_create_context(const performance_config_t* config, performance_context_t** context) {
    if (!config || !context) {
        return -1;
    }
    
    auto* ctx = new performance_context();
    if (!ctx) {
        return -2;
    }
    
    try {
        ctx->config = *config;
        
        // Create performance components
        ctx->profiler = std::make_unique<PerformanceOptimization::PerformanceProfiler>();
        ctx->parallel_processor = std::make_unique<PerformanceOptimization::ParallelProcessor>(ctx->config);
        ctx->streaming_processor = std::make_unique<PerformanceOptimization::StreamingProcessor>(ctx->config);
        
        ctx->is_initialized = true;
        *context = ctx;
        
        return 0;
        
    } catch (...) {
        delete ctx;
        return -3;
    }
}

void perf_destroy_context(performance_context_t* context) {
    if (context) {
        delete context;
    }
}

performance_config_t perf_get_default_config(void) {
    performance_config_t config = {};
    
    // Threading options
    config.num_threads = get_system_thread_count();
    config.enable_parallel_fft = true;
    config.enable_parallel_dtw = true;
    config.enable_parallel_ransac = true;
    config.enable_concurrent_pairs = true;
    
    // SIMD options
    config.enable_neon_optimization = true;
    config.enable_vectorized_audio_ops = true;
    config.enable_vectorized_correlation = true;
    config.enable_vectorized_dtw_distance = true;
    
    // Memory optimization
    config.enable_streaming_processing = true;
    config.chunk_size_samples = 1024 * 1024; // 1M samples
    config.buffer_pool_size = 16;
    config.enable_fft_caching = true;
    config.fft_cache_size = 100;
    
    // Performance tuning
    config.dtw_parallel_threshold = 1000;
    config.ransac_parallel_threshold = 100;
    config.memory_limit_gb = get_system_memory_size() / (1024.0 * 1024.0 * 1024.0) * 0.8; // 80% of system memory
    config.enable_early_termination = true;
    
    return config;
}

// C++ Implementation
namespace PerformanceOptimization {

// ThreadPool implementation
ThreadPool::ThreadPool(uint32_t num_threads) 
    : stop_(false), active_threads_(0), completed_tasks_(0) {
    
    if (num_threads == 0) {
        num_threads = get_system_thread_count();
    }
    
    num_threads_ = num_threads;
    
    for (uint32_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                active_threads_++;
                task();
                active_threads_--;
                completed_tasks_++;
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread &worker : workers_) {
        worker.join();
    }
}

template<class F, class... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_) {
            throw std::runtime_error("submit on stopped ThreadPool");
        }
        
        tasks_.emplace([task](){ (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty() && active_threads_ == 0; });
}

double ThreadPool::getUtilization() const {
    if (num_threads_ == 0) return 0.0;
    return static_cast<double>(active_threads_) / num_threads_;
}

// MemoryPool implementation
template<typename T>
MemoryPool<T>::MemoryPool(size_t buffer_size, size_t pool_size)
    : buffer_size_(buffer_size), pool_size_(pool_size) {
    
    for (size_t i = 0; i < pool_size_; ++i) {
        auto buffer = std::make_unique<std::vector<T>>();
        buffer->reserve(buffer_size_);
        available_buffers_.push(std::move(buffer));
    }
}

template<typename T>
MemoryPool<T>::~MemoryPool() = default;

template<typename T>
std::unique_ptr<std::vector<T>> MemoryPool<T>::getBuffer() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (!available_buffers_.empty()) {
        auto buffer = std::move(available_buffers_.front());
        available_buffers_.pop();
        buffer->clear();
        return buffer;
    }
    
    // Create new buffer if pool is empty
    auto buffer = std::make_unique<std::vector<T>>();
    buffer->reserve(buffer_size_);
    return buffer;
}

template<typename T>
void MemoryPool<T>::returnBuffer(std::unique_ptr<std::vector<T>> buffer) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (available_buffers_.size() < pool_size_) {
        buffer->clear();
        available_buffers_.push(std::move(buffer));
    }
    // If pool is full, just let the buffer be destroyed
}

template<typename T>
size_t MemoryPool<T>::getAvailableBuffers() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(pool_mutex_));
    return available_buffers_.size();
}

// Explicit template instantiations
template class MemoryPool<float>;
template class MemoryPool<std::complex<float>>;

// FFTCache implementation
FFTCache::FFTCache(size_t max_cache_size) 
    : max_cache_size_(max_cache_size), cache_hits_(0), cache_misses_(0) {}

FFTCache::~FFTCache() = default;

bool FFTCache::get(const std::vector<float>& input, std::vector<std::complex<float>>& output) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::string hash = computeHash(input);
    auto it = cache_.find(hash);
    
    if (it != cache_.end()) {
        output = it->second.output;
        it->second.access_count++;
        it->second.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        cache_hits_++;
        return true;
    }
    
    cache_misses_++;
    return false;
}

void FFTCache::put(const std::vector<float>& input, const std::vector<std::complex<float>>& output) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (cache_.size() >= max_cache_size_) {
        evictOldEntries();
    }
    
    std::string hash = computeHash(input);
    CacheEntry entry;
    entry.input_hash = input;
    entry.output = output;
    entry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    entry.access_count = 1;
    
    cache_[hash] = std::move(entry);
}

void FFTCache::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

size_t FFTCache::getSize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache_mutex_));
    return cache_.size();
}

double FFTCache::getHitRatio() const {
    uint32_t hits = cache_hits_;
    uint32_t misses = cache_misses_;
    uint32_t total = hits + misses;
    
    return total > 0 ? static_cast<double>(hits) / total : 0.0;
}

std::string FFTCache::computeHash(const std::vector<float>& input) {
    // Simple hash based on input size and first few samples
    std::ostringstream oss;
    oss << input.size();
    
    size_t sample_count = std::min(input.size(), size_t(16));
    for (size_t i = 0; i < sample_count; i += input.size() / sample_count) {
        oss << "_" << std::fixed << std::setprecision(6) << input[i];
    }
    
    return oss.str();
}

void FFTCache::evictOldEntries() {
    // Remove 25% of oldest entries
    size_t remove_count = cache_.size() / 4;
    
    std::vector<std::pair<uint64_t, std::string>> entries_by_time;
    for (const auto& entry : cache_) {
        entries_by_time.emplace_back(entry.second.timestamp, entry.first);
    }
    
    std::sort(entries_by_time.begin(), entries_by_time.end());
    
    for (size_t i = 0; i < remove_count && !entries_by_time.empty(); ++i) {
        cache_.erase(entries_by_time[i].second);
    }
}

// SIMDOperations implementation
bool SIMDOperations::neon_available_ = false;
bool SIMDOperations::avx_available_ = false;

bool SIMDOperations::isNEONAvailable() {
#ifdef __APPLE__
    return true; // ARM64 Macs always have NEON
#else
    return false;
#endif
}

bool SIMDOperations::isAVXAvailable() {
    // For now, focus on NEON for Apple Silicon
    return false;
}

void SIMDOperations::vectorizedAdd(const float* a, const float* b, float* result, size_t size) {
#ifdef __APPLE__
    if (isNEONAvailable()) {
        vDSP_vadd(a, 1, b, 1, result, 1, size);
        return;
    }
#endif
    
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOperations::vectorizedMultiply(const float* a, const float* b, float* result, size_t size) {
#ifdef __APPLE__
    if (isNEONAvailable()) {
        vDSP_vmul(a, 1, b, 1, result, 1, size);
        return;
    }
#endif
    
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOperations::vectorizedScale(const float* input, float scale, float* output, size_t size) {
#ifdef __APPLE__
    if (isNEONAvailable()) {
        vDSP_vsmul(input, 1, &scale, output, 1, size);
        return;
    }
#endif
    
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

float SIMDOperations::vectorizedDotProduct(const float* a, const float* b, size_t size) {
#ifdef __APPLE__
    if (isNEONAvailable()) {
        float result;
        vDSP_dotpr(a, 1, b, 1, &result, size);
        return result;
    }
#endif
    
    // Fallback to scalar implementation
    float result = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float SIMDOperations::vectorizedEuclideanDistance(const float* a, const float* b, size_t size) {
#ifdef __APPLE__
    if (isNEONAvailable()) {
        float result;
        vDSP_distancesq(a, 1, b, 1, &result, size);
        return std::sqrt(result);
    }
#endif
    
    // Fallback to scalar implementation
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float SIMDOperations::vectorizedManhattanDistance(const float* a, const float* b, size_t size) {
    // No direct vDSP equivalent, use optimized loop
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

float SIMDOperations::vectorizedCosineDistance(const float* a, const float* b, size_t size) {
    float dot_product = vectorizedDotProduct(a, b, size);
    float norm_a = vectorizedDotProduct(a, a, size);
    float norm_b = vectorizedDotProduct(b, b, size);
    
    if (norm_a > 0.0f && norm_b > 0.0f) {
        float cosine_sim = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
        return 1.0f - cosine_sim;
    }
    
    return 1.0f; // Maximum distance
}

// ParallelProcessor implementation
ParallelProcessor::ParallelProcessor(const performance_config_t& config) 
    : config_(config), stats_{} {
    
    thread_pool_ = std::make_unique<ThreadPool>(config_.num_threads);
    float_pool_ = std::make_unique<MemoryPool<float>>(config_.chunk_size_samples, config_.buffer_pool_size);
    complex_pool_ = std::make_unique<MemoryPool<std::complex<float>>>(config_.chunk_size_samples, config_.buffer_pool_size);
    
    if (config_.enable_fft_caching) {
        fft_cache_ = std::make_unique<FFTCache>(config_.fft_cache_size);
    }
}

ParallelProcessor::~ParallelProcessor() = default;

void ParallelProcessor::updateStats(const std::string& operation, double time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (operation == "fft") {
        stats_.fft_time_ms += time_ms;
    } else if (operation == "gcc_phat") {
        stats_.gcc_phat_time_ms += time_ms;
    } else if (operation == "dtw") {
        stats_.dtw_time_ms += time_ms;
    } else if (operation == "ransac") {
        stats_.ransac_time_ms += time_ms;
    }
    
    stats_.total_processing_time_ms += time_ms;
    stats_.threads_used = thread_pool_->getNumThreads();
    stats_.thread_efficiency = thread_pool_->getUtilization();
    
    if (fft_cache_) {
        stats_.cache_hit_ratio = fft_cache_->getHitRatio();
    }
}

uint32_t ParallelProcessor::getOptimalThreadCount() const {
    return std::min(config_.num_threads, get_system_thread_count());
}

// PerformanceProfiler implementation
PerformanceProfiler::PerformanceProfiler() : peak_memory_(0), cache_hits_(0), cache_misses_(0) {}

PerformanceProfiler::~PerformanceProfiler() = default;

void PerformanceProfiler::startTimer(const std::string& operation) {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    timers_[operation].start_time = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::endTimer(const std::string& operation) {
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(timing_mutex_);
    auto& timer = timers_[operation];
    
    if (timer.start_time.time_since_epoch().count() > 0) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - timer.start_time);
        timer.total_time_ms += duration.count() / 1000.0;
        timer.call_count++;
        timer.start_time = std::chrono::high_resolution_clock::time_point{};
    }
}

PerformanceProfiler::ScopedTimer::ScopedTimer(PerformanceProfiler& profiler, const std::string& operation)
    : profiler_(profiler), operation_(operation) {
    profiler_.startTimer(operation_);
}

PerformanceProfiler::ScopedTimer::~ScopedTimer() {
    profiler_.endTimer(operation_);
}

performance_stats_t PerformanceProfiler::generateReport() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(timing_mutex_));
    
    performance_stats_t stats = {};
    
    for (const auto& timer : timers_) {
        if (timer.first == "fft") {
            stats.fft_time_ms = timer.second.total_time_ms;
        } else if (timer.first == "gcc_phat") {
            stats.gcc_phat_time_ms = timer.second.total_time_ms;
        } else if (timer.first == "dtw") {
            stats.dtw_time_ms = timer.second.total_time_ms;
        } else if (timer.first == "ransac") {
            stats.ransac_time_ms = timer.second.total_time_ms;
        }
        
        stats.total_processing_time_ms += timer.second.total_time_ms;
    }
    
    stats.peak_memory_bytes = peak_memory_;
    stats.cache_hits = cache_hits_;
    stats.cache_misses = cache_misses_;
    
    uint32_t total_cache_access = stats.cache_hits + stats.cache_misses;
    stats.cache_hit_ratio = total_cache_access > 0 ? 
        static_cast<double>(stats.cache_hits) / total_cache_access : 0.0;
    
    stats.simd_used = SIMDOperations::isNEONAvailable();
    
    return stats;
}

void PerformanceProfiler::recordMemoryUsage(uint64_t bytes) {
    uint64_t current = peak_memory_.load();
    while (bytes > current && !peak_memory_.compare_exchange_weak(current, bytes)) {
        // Retry until successful or bytes <= current peak
    }
}

void PerformanceProfiler::recordCacheAccess(bool hit) {
    if (hit) {
        cache_hits_++;
    } else {
        cache_misses_++;
    }
}

// StreamingProcessor implementation
StreamingProcessor::StreamingProcessor(const performance_config_t& config) 
    : config_(config), peak_memory_usage_(0) {
    
    chunk_pool_ = std::make_unique<MemoryPool<float>>(config_.chunk_size_samples, config_.buffer_pool_size);
}

StreamingProcessor::~StreamingProcessor() = default;

std::vector<double> StreamingProcessor::streamingGCCPHAT(
    const std::vector<float>& reference,
    const std::vector<float>& target,
    double sample_rate,
    uint32_t chunk_size) {
    
    if (chunk_size == 0) {
        chunk_size = getOptimalChunkSize(std::max(reference.size(), target.size()));
    }
    
    std::vector<double> offsets;
    
    for (size_t i = 0; i + chunk_size < std::min(reference.size(), target.size()); i += chunk_size / 2) {
        // Extract chunks
        std::vector<float> ref_chunk(reference.begin() + i, reference.begin() + i + chunk_size);
        std::vector<float> target_chunk(target.begin() + i, target.begin() + i + chunk_size);
        
        // Simple correlation-based offset estimation (placeholder)
        double local_offset = 0.0; // Would implement actual GCC-PHAT here
        offsets.push_back(local_offset);
        
        updateMemoryUsage();
    }
    
    return offsets;
}

uint64_t StreamingProcessor::getCurrentMemoryUsage() const {
    // Simple estimation based on current allocations
    return chunk_pool_->getAvailableBuffers() * config_.chunk_size_samples * sizeof(float);
}

uint32_t StreamingProcessor::getOptimalChunkSize(size_t total_size) const {
    // Calculate optimal chunk size based on memory constraints
    uint32_t max_chunk = config_.chunk_size_samples;
    uint32_t min_chunk = 1024; // Minimum 1K samples
    
    if (total_size < max_chunk) {
        return std::max(min_chunk, static_cast<uint32_t>(total_size));
    }
    
    return max_chunk;
}

void StreamingProcessor::updateMemoryUsage() {
    uint64_t current_usage = getCurrentMemoryUsage();
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    if (current_usage > peak_memory_usage_) {
        peak_memory_usage_ = current_usage;
    }
}

} // namespace PerformanceOptimization