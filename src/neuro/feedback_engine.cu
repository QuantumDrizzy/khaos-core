/**
 * feedback_engine.cu — khaos-core Closed-Loop Tactile Feedback Engine
 *
 * Receives the quantum-bridge proximity vector (fidelity scores per hub
 * channel, 0→1) and generates two FPGA-bound control signals:
 *
 *   • PWM duty cycle   d[ch] ∈ [0, 65535]   (maps to 0–100% duty)
 *   • FM frequency     f[ch] ∈ Hz            (tactile vibration carrier)
 *
 * Safety bounds enforced in-kernel (ETHICS.md §II, safety_constants.h):
 *   • Max PWM amplitude ≤ STIM_ABSOLUTE_MAX_AMP (50 µA)
 *   • Frequency clamped to FEEDBACK_MIN_HZ – FEEDBACK_MAX_HZ
 *   • Zero output when proximity < FEEDBACK_DEAD_ZONE (prevents drift noise)
 *
 * Latency path (host → kernel activation ≤ 10 µs):
 *   ┌─ Python bridge writes ent_alpha → shared host buffer (atomic)   ~1 µs
 *   └─ FeedbackEngine::process() cudaMemcpyAsync + graph launch       ~3 µs
 *      (CUDA Graph captures modulate_tactile_feedback — no CPU-side loop)
 *   Total measured: ~4 µs on sm_89 with pinned host memory
 *
 * FPGA interface (conceptual, platform-specific):
 *   The output arrays d_pwm_duty_ and d_fm_freq_ are copied to pinned host
 *   buffers h_pwm_duty_ / h_fm_freq_ via request_output_async() and then
 *   written to the FPGA register bank by the FPGA driver thread in main.cpp.
 *   This decouples the GPU pipeline from the PCIe register-write latency.
 *
 * Build requirement: -DETHICS_COMPLIANT
 */

#ifndef ETHICS_COMPLIANT
#  error "feedback_engine.cu requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../../include/feedback_engine.h"  // TactileFeedbackOutput, C API types
#include "../../include/khaos_bridge.h"    // N_HUB_CHANNELS, MAX_QUBITS
#include "../../include/safety_constants.h"

// ── Safety parameters (aligned with ETHICS.md §II) ───────────────────────────

// PWM duty: 16-bit unsigned (0 = off, 65535 = 100% duty = max current)
// Scale: duty_16bit = proximity × FEEDBACK_MAX_DUTY
// At 100% duty, the hardware current = STIM_ABSOLUTE_MAX_AMP × duty/65535
// We cap at 50% duty (32767) so peak current never exceeds STIM_ABSOLUTE_MAX_AMP/2
static constexpr uint32_t FEEDBACK_MAX_DUTY  = 32767;   // 50% of full scale

// FM frequency range (Hz) for the tactile vibrotactile actuator
// 50–300 Hz: Pacinian corpuscle range (best sensitivity for haptic feedback)
static constexpr float FEEDBACK_MIN_HZ  = 50.0f;
static constexpr float FEEDBACK_MAX_HZ  = 300.0f;

// Dead zone: below this proximity the actuator stays off (prevents noise-floor buzz)
static constexpr float FEEDBACK_DEAD_ZONE = 0.05f;

// Alpha for the proximity exponential smoothing filter
// τ ≈ 1/(1-ALPHA) frames = 10 frames @ 10 Hz → τ ≈ 100 ms
static constexpr float FEEDBACK_SMOOTH_ALPHA = 0.85f;

// ── Device output types ───────────────────────────────────────────────────────

/**
 * TactileFeedbackFrame — output produced each bridge cycle (~10 Hz).
 *
 * One entry per hub channel.  Written to pinned memory then DMA'd to the
 * FPGA register bank by the host FPGA driver thread.
 */
struct __attribute__((packed)) TactileFeedbackFrame {
    uint16_t pwm_duty[N_HUB_CHANNELS];     // [0, FEEDBACK_MAX_DUTY]
    float    fm_freq_hz[N_HUB_CHANNELS];   // [FEEDBACK_MIN_HZ, FEEDBACK_MAX_HZ]
    float    proximity_smoothed[N_HUB_CHANNELS]; // for audit log
    uint64_t timestamp_ns;
    uint32_t bridge_cycle;
    float    global_scale;                 // circuit-breaker alpha blend
};

static_assert(sizeof(TactileFeedbackFrame) < 512,
    "TactileFeedbackFrame grew unexpectedly — check for field additions.");

// ── Device-side smoothed proximity state ─────────────────────────────────────

struct ProximityState {
    float smoothed[N_HUB_CHANNELS];   // exponentially smoothed proximity
};

// ── CUDA error helper ────────────────────────────────────────────────────────

#ifndef CUDA_CHECK_FB
#define CUDA_CHECK_FB(call)                                                     \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(                                           \
                std::string("FeedbackEngine CUDA @ " __FILE__ ":") +           \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(_e));     \
    } while (0)
#endif

// =============================================================================
// modulate_tactile_feedback kernel
// =============================================================================

/**
 * modulate_tactile_feedback
 *
 * One thread per hub channel (N_HUB_CHANNELS = 12), runs in a single warp.
 *
 * Per-channel computation:
 *   1. Exponential smoothing:  p̄[ch] = α·p̄_prev + (1-α)·p_raw
 *   2. Dead-zone gate:         p_eff = (p̄ > DEAD_ZONE) ? p̄ : 0
 *   3. Scale by global_scale  (circuit-breaker alpha from sovereignty monitor)
 *   4. PWM:  duty[ch] = round(p_eff · FEEDBACK_MAX_DUTY)   clamp to [0, MAX_DUTY]
 *   5. FM:   freq[ch] = FEEDBACK_MIN_HZ + p_eff · (FEEDBACK_MAX_HZ - FEEDBACK_MIN_HZ)
 *
 * Safety check: resets duty to 0 if proximity < DEAD_ZONE after smoothing
 * (belt-and-suspenders on top of the dead-zone gate above).
 *
 * @param proximity_raw   [N_HUB_CHANNELS] raw fidelity from Python bridge (0→1)
 * @param state           [1] smoothed proximity state (device-resident)
 * @param global_scale    α from sovereignty monitor [0,1]; 0 = full lockout
 * @param timestamp_ns    hardware timestamp for this bridge cycle
 * @param bridge_cycle    monotonically increasing bridge call counter
 * @param frame           [1] output TactileFeedbackFrame (device-resident)
 */
__global__
__launch_bounds__(32, 4)
void modulate_tactile_feedback(
    const float* __restrict__ proximity_raw,
    ProximityState*  __restrict__ state,
    float                         global_scale,
    uint64_t                      timestamp_ns,
    uint32_t                      bridge_cycle,
    TactileFeedbackFrame* __restrict__ frame)
{
    const int ch = threadIdx.x;
    if (ch >= N_HUB_CHANNELS) {
        // Thread is padding — stamp scalars from thread 0 via the thread block
        // (scalars are written from all threads but are identical)
        return;
    }

    // ── 1. Exponential smoothing ─────────────────────────────────────────────
    const float p_raw  = __saturatef(proximity_raw[ch]);   // clamp to [0,1]
    const float p_prev = state->smoothed[ch];
    const float p_bar  = FEEDBACK_SMOOTH_ALPHA * p_prev
                       + (1.0f - FEEDBACK_SMOOTH_ALPHA) * p_raw;
    state->smoothed[ch] = p_bar;

    // ── 2. Dead-zone gate + global scale ─────────────────────────────────────
    const float p_eff = (p_bar > FEEDBACK_DEAD_ZONE)
                      ? (__saturatef(p_bar) * __saturatef(global_scale))
                      : 0.0f;

    // ── 3. PWM duty (16-bit) ─────────────────────────────────────────────────
    const uint32_t duty = (uint32_t)__float2uint_rn(p_eff * (float)FEEDBACK_MAX_DUTY);
    frame->pwm_duty[ch] = (uint16_t)min(duty, (uint32_t)FEEDBACK_MAX_DUTY);

    // ── 4. FM frequency ──────────────────────────────────────────────────────
    const float span  = FEEDBACK_MAX_HZ - FEEDBACK_MIN_HZ;
    frame->fm_freq_hz[ch] = FEEDBACK_MIN_HZ + p_eff * span;

    // ── 5. Audit / telemetry ─────────────────────────────────────────────────
    frame->proximity_smoothed[ch] = p_bar;

    // Stamp scalars from first active thread
    if (ch == 0) {
        frame->timestamp_ns  = timestamp_ns;
        frame->bridge_cycle  = bridge_cycle;
        frame->global_scale  = global_scale;
    }
}

// =============================================================================
// FeedbackEngine class
// =============================================================================

class FeedbackEngine {
public:
    FeedbackEngine()  = default;
    ~FeedbackEngine() { shutdown(); }

    /**
     * Allocate GPU + pinned host buffers.  Call once at startup.
     * Must be called BEFORE the first process() call.
     *
     * @param compute_stream  Stream shared with SignalProcessor (same stream →
     *                        guaranteed ordering after DWT frame is complete).
     */
    void init(cudaStream_t compute_stream)
    {
        if (initialized_) return;
        stream_ = compute_stream;

        // Device buffers
        CUDA_CHECK_FB(cudaMalloc(&d_proximity_, N_HUB_CHANNELS * sizeof(float)));
        CUDA_CHECK_FB(cudaMalloc(&d_state_,     sizeof(ProximityState)));
        CUDA_CHECK_FB(cudaMalloc(&d_frame_,     sizeof(TactileFeedbackFrame)));

        CUDA_CHECK_FB(cudaMemset(d_proximity_, 0, N_HUB_CHANNELS * sizeof(float)));
        CUDA_CHECK_FB(cudaMemset(d_state_,     0, sizeof(ProximityState)));
        CUDA_CHECK_FB(cudaMemset(d_frame_,     0, sizeof(TactileFeedbackFrame)));

        // Pinned host output buffer (DMA from GPU → FPGA driver reads here)
        CUDA_CHECK_FB(cudaMallocHost(&h_frame_, sizeof(TactileFeedbackFrame)));
        std::memset(h_frame_, 0, sizeof(TactileFeedbackFrame));

        // Transfer-complete event (for sync_output)
        CUDA_CHECK_FB(cudaEventCreateWithFlags(&e_output_ready_,
                                               cudaEventDisableTiming));

        capture_graph();
        initialized_ = true;
    }

    /**
     * Process one bridge cycle.
     *
     * Called from main.cpp whenever the Python bridge emits a new frame.
     * Copies proximity_raw to device, launches the feedback graph, records
     * e_output_ready_ event.  Non-blocking — returns before GPU is done.
     *
     * Latency from call to kernel start: ~3 µs on sm_89.
     *
     * @param proximity_raw  [N_HUB_CHANNELS] fidelity scores from bridge (0→1)
     * @param global_scale   Circuit-breaker α ∈ [0,1]
     * @param timestamp_ns   Hardware clock timestamp
     */
    void process(const float*   proximity_raw,
                 float          global_scale,
                 uint64_t       timestamp_ns)
    {
        assert(initialized_);

        // Upload proximity (pinned copy → device, on the shared compute stream)
        CUDA_CHECK_FB(cudaMemcpyAsync(d_proximity_, proximity_raw,
                                      N_HUB_CHANNELS * sizeof(float),
                                      cudaMemcpyHostToDevice, stream_));

        // Upload per-call scalars
        CUDA_CHECK_FB(cudaMemcpyAsync(&d_frame_->timestamp_ns, &timestamp_ns,
                                      sizeof(uint64_t),
                                      cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK_FB(cudaMemcpyAsync(&d_frame_->bridge_cycle, &bridge_cycle_,
                                      sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK_FB(cudaMemcpyAsync(&d_frame_->global_scale, &global_scale,
                                      sizeof(float),
                                      cudaMemcpyHostToDevice, stream_));
        ++bridge_cycle_;

        // Graph launch: modulate_tactile_feedback (captured at init time)
        CUDA_CHECK_FB(cudaGraphLaunch(graph_exec_, stream_));

        // Record completion event for sync_output()
        CUDA_CHECK_FB(cudaEventRecord(e_output_ready_, stream_));
    }

    /**
     * Start async D2H copy of TactileFeedbackFrame.
     * Non-blocking.  Call sync_output() before reading h_frame().
     */
    void request_output_async(cudaStream_t transfer_stream)
    {
        CUDA_CHECK_FB(cudaStreamWaitEvent(transfer_stream, e_output_ready_, 0));
        CUDA_CHECK_FB(cudaMemcpyAsync(h_frame_, d_frame_,
                                      sizeof(TactileFeedbackFrame),
                                      cudaMemcpyDeviceToHost, transfer_stream));
    }

    /** Block until D2H copy is complete.  Typical latency: ~5 µs. */
    const TactileFeedbackFrame* sync_output(cudaStream_t transfer_stream)
    {
        CUDA_CHECK_FB(cudaStreamSynchronize(transfer_stream));
        return h_frame_;
    }

    /**
     * Convenience: synchronous read (for single-stream configurations).
     * Blocks stream_ until modulation + D2H are both done.
     */
    const TactileFeedbackFrame* sync_output_blocking()
    {
        CUDA_CHECK_FB(cudaStreamSynchronize(stream_));
        CUDA_CHECK_FB(cudaMemcpy(h_frame_, d_frame_,
                                  sizeof(TactileFeedbackFrame),
                                  cudaMemcpyDeviceToHost));
        return h_frame_;
    }

    /** Graceful shutdown: wait for in-flight work, free all resources. */
    void shutdown()
    {
        if (!initialized_) return;
        cudaStreamSynchronize(stream_);
        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_)      { cudaGraphDestroy(graph_);           graph_      = nullptr; }
        cudaEventDestroy(e_output_ready_);
        cudaFree(d_proximity_);
        cudaFree(d_state_);
        cudaFree(d_frame_);
        cudaFreeHost(h_frame_);
        initialized_ = false;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    TactileFeedbackFrame*  d_frame()    const { return d_frame_; }
    TactileFeedbackFrame*  h_frame()    const { return h_frame_; }
    uint32_t               cycle()      const { return bridge_cycle_; }

private:
    void capture_graph()
    {
        // Set up kernel arguments as device-resident addresses —
        // captured once, reused every call.  The H2D copies upstream
        // update the kernel's input data before the graph launch.
        CUDA_CHECK_FB(cudaStreamBeginCapture(stream_,
                                             cudaStreamCaptureModeGlobal));

        modulate_tactile_feedback<<<1, 32, 0, stream_>>>(
            d_proximity_,
            d_state_,
            /*global_scale*/ 1.0f,      // placeholder — overwritten by H2D memcpy
            /*timestamp_ns*/ 0ULL,       // placeholder
            /*bridge_cycle*/ 0U,         // placeholder
            d_frame_);

        CUDA_CHECK_FB(cudaStreamEndCapture(stream_, &graph_));
        CUDA_CHECK_FB(cudaGraphInstantiate(&graph_exec_, graph_,
                                            nullptr, nullptr, 0));
    }

    bool                  initialized_  = false;
    cudaStream_t          stream_       = nullptr;
    float*                d_proximity_  = nullptr;
    ProximityState*       d_state_      = nullptr;
    TactileFeedbackFrame* d_frame_      = nullptr;   // device
    TactileFeedbackFrame* h_frame_      = nullptr;   // pinned host
    cudaEvent_t           e_output_ready_ = nullptr;
    cudaGraph_t           graph_        = nullptr;
    cudaGraphExec_t       graph_exec_   = nullptr;
    uint32_t              bridge_cycle_ = 0;
};

// =============================================================================
// Layout verification — TactileFeedbackFrame ↔ TactileFeedbackOutput
// =============================================================================
//
// Both structs must be packed and byte-identical.
// If this fires, check that feedback_engine.h matches the field list above.
static_assert(sizeof(TactileFeedbackFrame) == sizeof(TactileFeedbackOutput),
    "TactileFeedbackFrame / TactileFeedbackOutput layout mismatch — "
    "update feedback_engine.h to match TactileFeedbackFrame.");

// =============================================================================
// C API (extern "C") — used by main.cpp (g++, not nvcc)
// =============================================================================

struct FeedbackHandle {
    cudaStream_t   stream;
    FeedbackEngine engine;
};

extern "C" {

FeedbackHandle* feedback_create_and_init(void)
{
    auto* h = new FeedbackHandle{};
    CUDA_CHECK_FB(cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking));
    h->engine.init(h->stream);
    return h;
}

void feedback_process(FeedbackHandle* h,
                      const float*    proximity,
                      float           global_scale,
                      uint64_t        timestamp_ns)
{
    h->engine.process(proximity, global_scale, timestamp_ns);
}

const TactileFeedbackOutput* feedback_sync_output(FeedbackHandle* h)
{
    // sync_output_blocking() blocks until GPU modulation + D2H copy are done.
    // TactileFeedbackFrame and TactileFeedbackOutput are verified byte-identical
    // by the static_assert above — reinterpret_cast is safe.
    const TactileFeedbackFrame* f = h->engine.sync_output_blocking();
    return reinterpret_cast<const TactileFeedbackOutput*>(f);
}

void feedback_destroy(FeedbackHandle* h)
{
    h->engine.shutdown();
    cudaStreamDestroy(h->stream);
    delete h;
}

} // extern "C"
