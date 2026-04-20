/**
 * lsl_connector.cpp — khaos-core Lab Streaming Layer Connector
 *
 * Provides LSLConnector, a thread-safe adapter that:
 *   • Discovers an EEG LSL stream by name/type
 *   • Pulls samples in a real-time background thread
 *   • Writes them into a SignalProcessor pinned ring buffer (zero-copy DMA path)
 *   • Measures and reports per-sample jitter using dual timestamps:
 *       - LSL remote timestamp (µs clock from amplifier / LSL clock)
 *       - Local CLOCK_MONOTONIC_RAW timestamp (hardware clock, immune to NTP)
 *   • Falls back to a synthetic sinusoidal generator if no LSL stream is found
 *     within the discovery timeout (configurable, default 5 s)
 *
 * Jitter statistics (ring-buffered, continuously updated):
 *   - mean_jitter_us    : running mean of |LSL_ts - local_ts| per sample
 *   - max_jitter_us     : worst-case observed jitter
 *   - sigma_jitter_us   : running standard deviation (Welford online algorithm)
 *   - late_count        : samples where |jitter| > JITTER_ALERT_THRESHOLD_US
 *   - drop_count        : missed expected samples (gap > 1.5 × expected_period)
 *
 * Integration with SignalProcessor:
 *   The SignalProcessor exposes acquire_write_slot() + commit_write_slot() for
 *   lock-free single-producer single-consumer ring buffer access.
 *   The LSL pull thread is the sole producer; the CUDA pipeline is the sole consumer.
 *
 * Build guard: KHAOS_LSL_ENABLED (set by CMakeLists.txt when liblsl is found).
 * When disabled, the file compiles to a stub that always uses the synthetic generator.
 *
 * Dependencies: liblsl (optional), POSIX clocks (CLOCK_MONOTONIC_RAW)
 */

#ifndef ETHICS_COMPLIANT
#  error "lsl_connector.cpp requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "../../include/lsl_connector.h"   // EEGFrameSlot (shared C API type)
#include "../../include/khaos_bridge.h"   // N_CHANNELS, NeuralPhaseVector
#include "../../include/safety_constants.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// POSIX
#include <pthread.h>
#include <sched.h>
#include <time.h>

#ifdef KHAOS_LSL_ENABLED
#  include <lsl_cpp.h>
#endif

// =============================================================================
// Jitter statistics (Welford online algorithm, lock-free read path)
// =============================================================================

struct JitterStats {
    std::atomic<uint64_t> n_samples    {0};   // total samples pulled
    std::atomic<uint64_t> late_count   {0};   // samples with |jitter| > threshold
    std::atomic<uint64_t> drop_count   {0};   // detected sample gaps

    // Welford state — updated only by pull thread, read by external callers.
    // The read path accepts a stale snapshot (no lock needed for monitoring).
    double mean_us  = 0.0;
    double M2_us    = 0.0;   // sum of squared deviations (for variance)
    double max_us   = 0.0;

    double sigma_us() const {
        uint64_t n = n_samples.load(std::memory_order_relaxed);
        return (n > 1) ? std::sqrt(M2_us / (double)(n - 1)) : 0.0;
    }

    void update(double jitter_us) {
        uint64_t n = n_samples.fetch_add(1, std::memory_order_relaxed) + 1;
        double delta  = jitter_us - mean_us;
        mean_us      += delta / (double)n;
        double delta2 = jitter_us - mean_us;
        M2_us        += delta * delta2;
        if (jitter_us > max_us) max_us = jitter_us;
    }

    void print() const {
        uint64_t n = n_samples.load(std::memory_order_relaxed);
        fprintf(stderr,
            "[lsl] Jitter  n=%llu  mean=%.1f µs  σ=%.1f µs  max=%.1f µs"
            "  late=%llu  drops=%llu\n",
            (unsigned long long)n,
            mean_us, sigma_us(), max_us,
            (unsigned long long)late_count.load(),
            (unsigned long long)drop_count.load());
    }
};

// ── LSL / timing constants ────────────────────────────────────────────────────

static constexpr int    SAMPLE_RATE_HZ           = 1000;
static constexpr double EXPECTED_PERIOD_US        = 1000000.0 / SAMPLE_RATE_HZ; // 1000 µs
static constexpr double JITTER_ALERT_THRESHOLD_US = 500.0;    // 0.5 ms — loud alert
static constexpr double DROP_THRESHOLD_RATIO      = 1.5;      // > 1.5× period = drop
static constexpr int    LSL_DISCOVERY_TIMEOUT_S   = 5;
static constexpr int    LSL_PULL_TIMEOUT_S        = 2;        // max wait per sample

// ── Synthetic fallback parameters ────────────────────────────────────────────
// Simulates a resting-state EEG: 10 Hz mu + 25 Hz beta + 1/f background noise
static constexpr float SYN_MU_AMP   = 5.0f;    // µV
static constexpr float SYN_BETA_AMP = 3.0f;    // µV
static constexpr float SYN_NOISE_AMP= 2.0f;    // µV

// =============================================================================
// Ring buffer  (EEGFrameSlot is defined in include/lsl_connector.h)
// =============================================================================

// ── Ring buffer (fixed-size, matches RING_FRAMES in signal_processor.cu) ─────
static constexpr int RING_FRAMES = 8;

struct EEGRingBuffer {
    EEGFrameSlot           frames[RING_FRAMES];
    std::atomic<int>       write_head{0};
    std::atomic<int>       read_head{0};
};

// =============================================================================
// Platform clock helper
// =============================================================================

static inline uint64_t monotonic_ns()
{
#if defined(_WIN32)
    // Windows: QueryPerformanceCounter
    // Not used in WSL2 (which runs Linux), but included for completeness.
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    struct timespec ts{};
#  if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#  else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#  endif
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// =============================================================================
// LSLConnector
// =============================================================================

class LSLConnector {
public:
    /**
     * @param stream_name   LSL stream name to search for (e.g. "Emotiv EEG")
     * @param stream_type   LSL stream type (e.g. "EEG")
     * @param ring          Pointer to the SignalProcessor's pinned ring buffer.
     *                      Must remain valid for the lifetime of LSLConnector.
     */
    LSLConnector(const std::string& stream_name,
                 const std::string& stream_type,
                 EEGRingBuffer*     ring)
        : stream_name_(stream_name)
        , stream_type_(stream_type)
        , ring_(ring)
    {}

    ~LSLConnector() { stop(); }

    /**
     * Start the acquisition thread.
     *
     * Attempts to find an LSL stream within LSL_DISCOVERY_TIMEOUT_S seconds.
     * If no stream is found, the synthetic generator is used automatically.
     *
     * @param use_synthetic  Force synthetic mode regardless of LSL availability.
     */
    void start(bool use_synthetic = false)
    {
        if (running_.load()) return;

        if (!use_synthetic) {
            use_synthetic = !try_connect_lsl();
        }
        synthetic_ = use_synthetic;

        if (synthetic_) {
            fprintf(stderr, "[lsl] No LSL stream found — using synthetic generator"
                            " (10 Hz μ + 25 Hz β + noise)\n");
        } else {
            fprintf(stderr, "[lsl] Connected to stream '%s' type '%s'\n",
                    stream_name_.c_str(), stream_type_.c_str());
        }

        running_.store(true);
        thread_ = std::thread(&LSLConnector::pull_loop, this);
    }

    /** Signal the pull thread to stop and join it.  Safe to call multiple times. */
    void stop()
    {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
        stats_.print();
    }

    /** Read-only access to live jitter statistics. */
    const JitterStats& stats() const { return stats_; }

    /** True when using synthetic data (no LSL stream). */
    bool is_synthetic() const { return synthetic_; }

    /** Observed LSL-reported sample rate (0 if not yet connected or synthetic). */
    double lsl_sample_rate() const { return lsl_sample_rate_; }

    /**
     * Configure real-time scheduling for the pull thread.
     * Must be called before start().  Settings are applied inside pull_loop()
     * via pthread_setaffinity_np + pthread_setschedparam(SCHED_FIFO).
     *
     * @param cpu_core       Core index (≥0) to pin the thread to.  Pass -1 to skip.
     * @param sched_priority SCHED_FIFO priority [1,99].  Pass -1 to skip.
     */
    void set_realtime(int cpu_core, int sched_priority)
    {
        rt_cpu_core_    = cpu_core;
        rt_sched_prio_  = sched_priority;
    }

private:
    // ── LSL discovery ─────────────────────────────────────────────────────────

    bool try_connect_lsl()
    {
#ifndef KHAOS_LSL_ENABLED
        (void)stream_name_; (void)stream_type_;
        return false;   // liblsl not compiled in
#else
        try {
            std::vector<lsl::stream_info> results =
                lsl::resolve_stream("name", stream_name_.c_str(),
                                    /*minimum=*/1, LSL_DISCOVERY_TIMEOUT_S);
            if (results.empty()) {
                // Try by type as fallback
                results = lsl::resolve_stream("type", stream_type_.c_str(),
                                              1, 2 /*shorter second search*/);
            }
            if (results.empty()) return false;

            inlet_       = std::make_unique<lsl::stream_inlet>(results[0]);
            lsl_sample_rate_ = results[0].nominal_srate();
            lsl_n_ch_    = results[0].channel_count();
            lsl_buf_.resize(lsl_n_ch_);

            // Open the inlet (blocks briefly for handshake)
            inlet_->open_stream(5.0 /*timeout*/);
            return true;
        } catch (const std::exception& ex) {
            fprintf(stderr, "[lsl] Discovery error: %s\n", ex.what());
            return false;
        }
#endif
    }

    // ── Pull loop (runs in background thread) ─────────────────────────────────

    void pull_loop()
    {
        // ── Real-time thread setup (applied before the acquisition loop) ───────
        pthread_t self = pthread_self();

        if (rt_cpu_core_ >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET((size_t)rt_cpu_core_, &cpuset);
            int rc = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
            if (rc != 0)
                fprintf(stderr, "[lsl] WARNING: pthread_setaffinity_np(core=%d) "
                                "failed (rc=%d)\n", rt_cpu_core_, rc);
            else
                fprintf(stderr, "[lsl] Pull thread pinned to Core %d\n",
                        rt_cpu_core_);
        }

        if (rt_sched_prio_ >= 1) {
            struct sched_param sp{};
            sp.sched_priority = rt_sched_prio_;
            int rc = pthread_setschedparam(self, SCHED_FIFO, &sp);
            if (rc != 0)
                fprintf(stderr, "[lsl] WARNING: SCHED_FIFO prio=%d failed "
                                "(rc=%d) — running SCHED_OTHER\n",
                        rt_sched_prio_, rc);
            else
                fprintf(stderr, "[lsl] Pull thread SCHED_FIFO prio=%d\n",
                        rt_sched_prio_);
        }

        uint64_t last_ts_ns    = 0;
        uint32_t frame_index   = 0;
        double   lsl_offset_us = compute_lsl_offset();  // LSL ↔ local clock offset

        fprintf(stderr, "[lsl] Pull thread started (synthetic=%d)\n", (int)synthetic_);

        while (running_.load(std::memory_order_relaxed)) {
            // --- Acquire ring buffer write slot ---
            EEGFrameSlot* slot = acquire_slot();
            if (!slot) {
                // Ring full — CUDA pipeline is lagging.  Spin briefly then retry.
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            // --- Fill the slot ---
            uint64_t local_ts_ns = monotonic_ns();

            if (synthetic_) {
                fill_synthetic(slot->samples, frame_index);
                slot->timestamp_ns = local_ts_ns;
            } else {
                bool ok = fill_lsl(slot->samples, local_ts_ns,
                                   lsl_offset_us, slot->timestamp_ns);
                if (!ok) {
                    release_slot_without_commit();
                    continue;
                }
            }

            slot->frame_index = frame_index++;

            // --- Jitter analysis ---
            if (last_ts_ns > 0) {
                double actual_period_us = (double)(local_ts_ns - last_ts_ns) / 1000.0;
                double jitter_us = std::abs(actual_period_us - EXPECTED_PERIOD_US);
                stats_.update(jitter_us);

                if (jitter_us > JITTER_ALERT_THRESHOLD_US)
                    stats_.late_count.fetch_add(1, std::memory_order_relaxed);

                if (actual_period_us > DROP_THRESHOLD_RATIO * EXPECTED_PERIOD_US)
                    stats_.drop_count.fetch_add(1, std::memory_order_relaxed);
            }
            last_ts_ns = local_ts_ns;

            // --- Commit ---
            commit_slot();

            // --- Periodic jitter report (every 10 000 samples = 10 s) ---
            if (frame_index % 10000 == 0) stats_.print();

            // --- Synthetic rate pacing (LSL path is blocked by pull_sample) ---
            if (synthetic_) {
                // Target 1000 Hz — sleep for the remainder of 1 ms period
                uint64_t elapsed_ns = monotonic_ns() - local_ts_ns;
                int64_t  sleep_ns   = 1000000LL - (int64_t)elapsed_ns;
                if (sleep_ns > 50000LL) {
                    struct timespec ts{ 0, sleep_ns };
                    nanosleep(&ts, nullptr);
                }
            }
        }
        fprintf(stderr, "[lsl] Pull thread stopped after %llu samples\n",
                (unsigned long long)stats_.n_samples.load());
    }

    // ── Ring buffer helpers ───────────────────────────────────────────────────

    EEGFrameSlot* acquire_slot()
    {
        int w = ring_->write_head.load(std::memory_order_relaxed);
        int r = ring_->read_head.load(std::memory_order_acquire);
        if (w - r >= RING_FRAMES) return nullptr;   // full
        return &ring_->frames[w % RING_FRAMES];
    }

    void commit_slot() {
        ring_->write_head.fetch_add(1, std::memory_order_release);
    }

    void release_slot_without_commit() { /* slot simply not committed */ }

    // ── Synthetic generator ───────────────────────────────────────────────────

    void fill_synthetic(float* samples, uint32_t frame_index) const
    {
        const float t    = (float)frame_index / (float)SAMPLE_RATE_HZ;
        const float mu   = SYN_MU_AMP   * sinf(2.0f * 3.14159f * 10.0f * t);
        const float beta = SYN_BETA_AMP * sinf(2.0f * 3.14159f * 25.0f * t);

        // Simple xorshift32 noise (deterministic, low overhead)
        uint32_t xr = 0x12345678u ^ frame_index;
        for (int c = 0; c < N_CHANNELS; ++c) {
            xr ^= xr << 13; xr ^= xr >> 17; xr ^= xr << 5;
            float noise = SYN_NOISE_AMP * ((float)(xr & 0xFFFF) / 32768.0f - 1.0f);
            // Slight inter-channel phase variation (hemisphere model)
            float phase_offset = (float)c * 0.15f;
            samples[c] = mu * cosf(phase_offset) + beta * sinf(phase_offset) + noise;
        }
    }

    // ── LSL sample pull ───────────────────────────────────────────────────────

#ifdef KHAOS_LSL_ENABLED
    bool fill_lsl(float* samples, uint64_t local_ts_ns,
                  double lsl_offset_us, uint64_t& out_ts_ns)
    {
        double lsl_ts = inlet_->pull_sample(lsl_buf_,
                                             (double)LSL_PULL_TIMEOUT_S);
        if (lsl_ts == 0.0) return false;   // timeout

        // Convert LSL timestamp (seconds from LSL epoch) to ns
        // lsl_offset_us bridges the gap between LSL epoch and CLOCK_MONOTONIC_RAW
        double lsl_local_us = lsl_ts * 1e6 + lsl_offset_us;
        out_ts_ns = (uint64_t)(lsl_local_us * 1000.0);

        // Map LSL channels to khaos N_CHANNELS
        int n_copy = std::min((int)lsl_n_ch_, N_CHANNELS);
        for (int c = 0; c < n_copy; ++c)
            samples[c] = (float)lsl_buf_[c];
        // Zero-pad if LSL stream has fewer channels than N_CHANNELS
        for (int c = n_copy; c < N_CHANNELS; ++c)
            samples[c] = 0.0f;

        return true;
    }
#else
    bool fill_lsl(float*, uint64_t, double, uint64_t&) { return false; }
#endif

    // ── LSL ↔ local clock offset (computed once at startup) ──────────────────

    double compute_lsl_offset() const
    {
#ifdef KHAOS_LSL_ENABLED
        if (!inlet_) return 0.0;
        // Measure round-trip time to the LSL clock and estimate offset.
        // Pull one sample, record local time before and after, use midpoint.
        // This is the standard LSL time-correction approach.
        const int N_PING = 10;
        double sum = 0.0;
        for (int i = 0; i < N_PING; ++i) {
            uint64_t t0 = monotonic_ns();
            double lsl_now = lsl::local_clock();
            uint64_t t1 = monotonic_ns();
            double local_mid_us = (double)(t0 + t1) / 2.0 / 1000.0;
            sum += (local_mid_us - lsl_now * 1e6);
        }
        return sum / N_PING;
#else
        return 0.0;
#endif
    }

    // ── Members ───────────────────────────────────────────────────────────────
    std::string       stream_name_;
    std::string       stream_type_;
    EEGRingBuffer*    ring_;

    std::atomic<bool> running_    {false};
    bool              synthetic_  {true};
    std::thread       thread_;

    JitterStats       stats_;
    double            lsl_sample_rate_ = 0.0;
    int               lsl_n_ch_        = 0;

    // Real-time scheduling settings (applied inside pull_loop() at startup)
    int               rt_cpu_core_    = -1;   // -1 = no affinity pinning
    int               rt_sched_prio_  = -1;   // -1 = no SCHED_FIFO

#ifdef KHAOS_LSL_ENABLED
    std::unique_ptr<lsl::stream_inlet> inlet_;
    std::vector<double>                lsl_buf_;
#endif
};

// =============================================================================
// Jitter report helper (for main.cpp shutdown summary)
// =============================================================================

/**
 * Print a one-line jitter summary suitable for the sovereignty audit log.
 */
static void print_lsl_jitter_report(const JitterStats& s)
{
    fprintf(stderr,
        "════════════════════════════════════════\n"
        "  LSL Jitter Report\n"
        "  Samples  : %llu\n"
        "  Mean     : %.2f µs\n"
        "  Sigma    : %.2f µs\n"
        "  Max      : %.2f µs\n"
        "  Late (>%.0f µs) : %llu\n"
        "  Drops    : %llu\n"
        "════════════════════════════════════════\n",
        (unsigned long long)s.n_samples.load(),
        s.mean_us,
        s.sigma_us(),
        s.max_us,
        JITTER_ALERT_THRESHOLD_US,
        (unsigned long long)s.late_count.load(),
        (unsigned long long)s.drop_count.load());
}

// =============================================================================
// C API (extern "C") — used by main.cpp which is compiled by g++, not nvcc
// =============================================================================
//
// LSLHandle owns both the ring buffer and the LSLConnector.
// The connector's pull thread is the sole ring-buffer producer;
// lsl_try_pop() is the sole consumer (called from the main loop).

struct LSLHandle {
    EEGRingBuffer                ring;
    std::unique_ptr<LSLConnector> connector;

    LSLHandle(const char* name, const char* type)
        : connector(std::make_unique<LSLConnector>(
              std::string(name), std::string(type), &ring))
    {}
};

extern "C" {

LSLHandle* lsl_create(const char* stream_name, const char* stream_type)
{
    return new LSLHandle(stream_name, stream_type);
}

void lsl_start(LSLHandle* h, int use_synthetic)
{
    h->connector->start(use_synthetic != 0);
}

int lsl_try_pop(LSLHandle* h, EEGFrameSlot* out)
{
    // Single-consumer: only the main loop calls this.
    // Relaxed load of write_head is fine — the producer used release on commit.
    int r = h->ring.read_head.load(std::memory_order_relaxed);
    int w = h->ring.write_head.load(std::memory_order_acquire);
    if (r >= w) return 0;
    *out = h->ring.frames[r % RING_FRAMES];
    h->ring.read_head.store(r + 1, std::memory_order_release);
    return 1;
}

void lsl_print_stats(const LSLHandle* h)
{
    print_lsl_jitter_report(h->connector->stats());
}

int lsl_is_synthetic(const LSLHandle* h)
{
    return h->connector->is_synthetic() ? 1 : 0;
}

void lsl_set_realtime(LSLHandle* h, int cpu_core, int sched_priority)
{
    h->connector->set_realtime(cpu_core, sched_priority);
}

void lsl_stop(LSLHandle* h)
{
    h->connector->stop();
}

void lsl_destroy(LSLHandle* h)
{
    delete h;
}

} // extern "C"
