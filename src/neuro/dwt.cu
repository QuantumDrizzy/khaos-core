/**
 * dwt.cu
 * KHAOS — Daubechies D4 Feature Extractor
 *
 * Implements a streaming à-trous (undecimated) Daubechies D4 wavelet
 * analysis filter bank for real-time mu/beta band power extraction.
 *
 * Pipeline position:
 *   signal_processor.cu → [d_filt_samples] → THIS FILE → [NeuralPhaseVector.theta]
 *
 * Why à-trous (undecimated)?
 *   Standard DWT decimates by 2 at each level, producing one coefficient every
 *   2^j samples at level j. At 1000 Hz, level-6 detail gives one coefficient
 *   per 64 ms — too slow for real-time BCI. The à-trous DWT inserts zeros
 *   between filter taps at each level (upsampled kernels) and produces one
 *   output per input sample at every level. Translation-invariant and
 *   compatible with single-sample streaming.
 *
 * Frequency bands (at 1000 Hz sample rate):
 *   Level 5 detail:  15.6 – 31.25 Hz  ≈ beta  (target: 13–30 Hz)
 *   Level 6 detail:   7.8 – 15.6  Hz  ≈ mu    (target:  8–13 Hz)
 *
 * Theta angle mapping:
 *   For each of the N_HUB_CHANNELS ICA neural hub channels, we compute:
 *     - P_mu[ch]   = short-window RMS of level-6 detail coefficients
 *     - P_beta[ch] = short-window RMS of level-5 detail coefficients
 *     - theta[ch]  = 2 * arcsin( sqrt( P_beta / (P_mu + P_beta + ε) ) )
 *
 *   When P_beta >> P_mu  → theta ≈ π   (active/motor state → gate near |1⟩)
 *   When P_mu  >> P_beta → theta ≈ 0   (rest/idle state    → gate near |0⟩)
 *   Equal power           → theta ≈ π/2 (neutral superposition)
 *
 * CUDA parallelism:
 *   One thread per hub channel. All 12 channels run in parallel within
 *   a single 32-thread warp (padded to warp width). No inter-thread sync needed.
 *
 * Build requirement: ETHICS_COMPLIANT
 */

#ifndef ETHICS_COMPLIANT
#  error "dwt.cu requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>   // CUDART_PI_F, etc.
#include <cassert>
#include <stdexcept>
#include <string>
#include "../../include/khaos_bridge.h"

// ── CUDA error helper ─────────────────────────────────────────────────────────
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":")                      \
                + std::to_string(__LINE__) + " — "                              \
                + cudaGetErrorString(_e));                                      \
        }                                                                       \
    } while (0)
#endif

// =============================================================================
// D4 wavelet coefficients
// =============================================================================

/**
 * Daubechies D4 scaling function (low-pass analysis filter):
 *   h[k], k = 0..3
 *
 * Orthonormal, compact support, 2 vanishing moments.
 * Values: (1 ± √3) / (4√2)
 */
#define D4_H0  0.48296291314453416f   //  (1 + √3) / (4√2)
#define D4_H1  0.83651630373780772f   //  (3 + √3) / (4√2)
#define D4_H2  0.22414386804201339f   //  (3 - √3) / (4√2)
#define D4_H3 -0.12940952255126034f   //  (1 - √3) / (4√2)

/**
 * D4 wavelet function (high-pass analysis filter, QMF relationship):
 *   g[k] = (-1)^k * h[3-k]
 */
#define D4_G0  D4_H3                  //  (1 - √3) / (4√2)
#define D4_G1 (-D4_H2)               // -(3 - √3) / (4√2)
#define D4_G2  D4_H1                  //  (3 + √3) / (4√2)
#define D4_G3 (-D4_H0)               // -(1 + √3) / (4√2)

// Number of DWT levels computed per frame
#define N_DWT_LEVELS  7               // level 0..6; we use detail[5] and detail[6]
#define N_D4_TAPS     4               // D4 kernel length

// Level indices for the target bands
#define LEVEL_BETA  5                 // ~15.6 – 31.25 Hz ≈ beta
#define LEVEL_MU    6                 // ~7.8  – 15.6  Hz ≈ mu

// Short-window RMS: number of past detail coefficients to accumulate
// 32 samples = 32 ms at 1000 Hz — responsive but stable
#define POWER_WINDOW  32

// =============================================================================
// Per-channel DWT state
// =============================================================================

/**
 * Streaming à-trous DWT state for one source channel.
 *
 * For the à-trous transform at level j, the effective filter uses
 * taps spaced 2^j samples apart. We emulate this by maintaining
 * a delay line of length (N_D4_TAPS - 1) * 2^j + 1 per level.
 *
 * To keep state compact, we store the minimum required delay samples
 * per level using a fixed-size circular buffer. The buffer at level j
 * must hold at least (N_D4_TAPS - 1) * 2^(j-1) + 1 samples.
 *
 * Max delay required: level 6 → (4-1) * 2^5 = 96 samples.
 * We allocate 128 per level (next power of 2) to simplify indexing.
 */
#define DWT_DELAY_SAMPLES  128        // per level, per channel
#define DWT_DELAY_MASK     (DWT_DELAY_SAMPLES - 1)

struct DWTChannelState {
    // Approximation coefficients (low-pass output) per level
    float approx[N_DWT_LEVELS][DWT_DELAY_SAMPLES];

    // Power accumulator for detail coefficients at target levels
    // Ring buffer of squared detail coefficients
    float power_window_beta[POWER_WINDOW];   // level LEVEL_BETA detail^2
    float power_window_mu[POWER_WINDOW];     // level LEVEL_MU   detail^2

    // Write positions in the power windows
    int  pw_beta_pos;
    int  pw_mu_pos;

    // Write position in the delay buffers (shared across levels)
    int  delay_pos[N_DWT_LEVELS];

    // Running sums for O(1) window power update
    float power_sum_beta;
    float power_sum_mu;
};

// =============================================================================
// Device-side à-trous DWT step
// =============================================================================

/**
 * Compute one level of the à-trous DWT for a single new input sample.
 *
 * The à-trous filter at level j applies the D4 kernel with taps spaced
 * stride = 2^(j-1) apart (stride = 1 at level 1).
 *
 * This is called once per level per incoming sample.
 *
 * @param state     Channel state (delay lines, power accumulators)
 * @param x_in      Input sample (approx output of previous level, or raw ICA source)
 * @param level     Current DWT level (1-indexed)
 * @param detail_out  Output: detail coefficient (high-pass) at this level
 * @returns           Approximation coefficient (low-pass) for the next level
 */
__device__ float atrous_step(DWTChannelState* state, float x_in,
                               int level, float* detail_out)
{
    int stride = 1 << (level - 1);   // 2^(level-1)
    int lv     = level - 1;          // 0-indexed level

    // Write new input into the circular delay buffer
    int pos = state->delay_pos[lv];
    state->approx[lv][pos & DWT_DELAY_MASK] = x_in;
    state->delay_pos[lv] = pos + 1;

    // Read taps spaced 'stride' apart from the delay buffer
    // tap[k] = sample from 'stride * k' steps ago
    auto read_tap = [&](int k) -> float {
        int idx = (pos - k * stride) & DWT_DELAY_MASK;
        return state->approx[lv][idx];
    };

    float t0 = read_tap(0);
    float t1 = read_tap(1);
    float t2 = read_tap(2);
    float t3 = read_tap(3);

    // Low-pass output (approximation) — D4 scaling function
    float approx  = D4_H0 * t0 + D4_H1 * t1 + D4_H2 * t2 + D4_H3 * t3;

    // High-pass output (detail) — D4 wavelet function
    float detail   = D4_G0 * t0 + D4_G1 * t1 + D4_G2 * t2 + D4_G3 * t3;

    *detail_out = detail;
    return approx;
}

/**
 * Update a short-window power accumulator with a new detail coefficient.
 * Uses O(1) update: subtract oldest sample, add new sample.
 *
 * @param window  Circular buffer of squared detail values
 * @param sum     Running sum (maintained for O(1) mean power)
 * @param pos     Current write position (incremented by caller)
 * @param d       New detail coefficient
 * @returns       Updated mean power ∈ [0, ∞)
 */
__device__ float update_power(float* window, float* sum, int pos, float d)
{
    float d_sq = d * d;
    int idx    = pos & (POWER_WINDOW - 1);
    *sum      -= window[idx];
    *sum      += d_sq;
    window[idx] = d_sq;
    return *sum / POWER_WINDOW;
}

// =============================================================================
// Main DWT kernel
// =============================================================================

/**
 * dwt_extract_theta_kernel
 *
 * Processes one new sample per hub channel, runs the à-trous DWT through
 * all required levels, extracts mu/beta band power, and maps to theta angles.
 *
 * Launch configuration:
 *   grid(1), block(32)   — 12 active threads, padded to warp width
 *
 * @param ica_sources     [N_HUB_CHANNELS] — one ICA source sample per channel
 *                         (output of the ICA unmixing step, NOT raw EEG)
 * @param state           [N_HUB_CHANNELS] — persistent per-channel DWT state
 * @param frame           Output NeuralPhaseVector (theta angles updated in-place)
 */
__global__ void dwt_extract_theta_kernel(
    const float*        __restrict__ ica_sources,
    DWTChannelState*    __restrict__ state,
    NeuralPhaseVector*  __restrict__ frame)
{
    int ch = threadIdx.x;
    if (ch >= N_HUB_CHANNELS) return;   // pad threads do nothing

    DWTChannelState* s = &state[ch];
    float x = ica_sources[ch];

    // Run à-trous DWT levels 1 through N_DWT_LEVELS
    float detail;

    for (int lv = 1; lv <= N_DWT_LEVELS; ++lv) {
        x = atrous_step(s, x, lv, &detail);

        if (lv == LEVEL_BETA) {
            float p = update_power(s->power_window_beta, &s->power_sum_beta,
                                   s->pw_beta_pos++, detail);
            // Store for theta computation below (reuse register 'detail')
            detail = p;   // repurpose variable for clarity
        }

        if (lv == LEVEL_MU) {
            float p_mu   = update_power(s->power_window_mu, &s->power_sum_mu,
                                        s->pw_mu_pos++, detail);
            // Retrieve beta power (stored in approx slot — see above)
            float p_beta = s->power_sum_beta / POWER_WINDOW;

            // ── Theta angle mapping ──────────────────────────────────────────
            //
            // theta = 2 * arcsin( sqrt( P_beta / (P_beta + P_mu + ε) ) )
            //
            // This maps the beta/(mu+beta) power ratio to a Ry gate angle via
            // arcsin-squared, which gives a uniform distribution of state
            // populations for uniformly distributed power ratios. The factor
            // of 2 gives full [0, π] range.
            //
            // Boundary behaviour:
            //   P_beta = 0, P_mu > 0  →  theta = 0          → Ry(0) = |0⟩
            //   P_beta = P_mu         →  theta = π/2        → Ry(π/2) = |+⟩
            //   P_beta > 0, P_mu = 0  →  theta = π          → Ry(π) = |1⟩

            const float eps    = 1e-8f;
            float total        = p_beta + p_mu + eps;
            float ratio        = p_beta / total;
            float theta        = 2.0f * asinf(sqrtf(__saturatef(ratio)));

            frame->theta[ch] = theta;
        }
    }
}

// =============================================================================
// ICA unmixing kernel (precedes the DWT — applies the learned W matrix)
// =============================================================================

/**
 * ica_apply_kernel
 *
 * Projects the 64-channel filtered EEG into 12 independent components
 * (neural hubs) using the current ICA unmixing matrix W.
 *
 * sources[i] = Σ_j  W[i][j] * filtered[j]    for i in [0, N_HUB_CHANNELS)
 *
 * Launch config:
 *   grid(1), block(N_HUB_CHANNELS)
 *
 * @param filtered   [N_CHANNELS] IIR-filtered EEG samples (µV)
 * @param W          [N_HUB_CHANNELS][N_CHANNELS] unmixing matrix (row-major)
 * @param sources    [N_HUB_CHANNELS] output — independent source amplitudes
 */
__global__ void ica_apply_kernel(
    const float* __restrict__ filtered,
    const float* __restrict__ W,         // row-major: W[hub][channel]
    float*       __restrict__ sources)
{
    int hub = threadIdx.x;
    if (hub >= N_HUB_CHANNELS) return;

    const float* row = W + hub * N_CHANNELS;
    float acc = 0.0f;

    // Unrolled dot product: W_row · filtered
    // N_CHANNELS = 64 — fits in registers for sm_89
    #pragma unroll 8
    for (int ch = 0; ch < N_CHANNELS; ++ch) {
        acc += row[ch] * filtered[ch];
    }

    sources[hub] = acc;
}

// =============================================================================
// DWTExtractor class — manages device state and kernel launches
// =============================================================================

class DWTExtractor {
public:
    DWTExtractor() = default;
    ~DWTExtractor() { shutdown(); }

    /**
     * Allocate device memory for DWT state and ICA buffers.
     * Call once at startup, before the first process_frame().
     */
    void init() {
        if (initialized_) return;

        CUDA_CHECK(cudaMalloc(&d_state_,   N_HUB_CHANNELS * sizeof(DWTChannelState)));
        CUDA_CHECK(cudaMalloc(&d_sources_, N_HUB_CHANNELS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_,       N_HUB_CHANNELS * N_CHANNELS * sizeof(float)));

        // Zero-initialize DWT state (zero initial conditions)
        CUDA_CHECK(cudaMemset(d_state_, 0, N_HUB_CHANNELS * sizeof(DWTChannelState)));

        // W starts as identity-like — first N_HUB_CHANNELS rows of identity
        // Overwritten immediately after first ICA convergence
        float W_init[N_HUB_CHANNELS * N_CHANNELS] = {};
        for (int i = 0; i < N_HUB_CHANNELS; ++i)
            W_init[i * N_CHANNELS + i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_W_, W_init,
                              N_HUB_CHANNELS * N_CHANNELS * sizeof(float),
                              cudaMemcpyHostToDevice));

        initialized_ = true;
    }

    /**
     * Update the ICA unmixing matrix W from the background ICA stream.
     * Safe to call between frames — the kernel reads W atomically by
     * value (no partial reads possible for a full row at this launch size).
     *
     * @param W_new  Row-major [N_HUB_CHANNELS × N_CHANNELS] matrix on host.
     *               Copied asynchronously on stream_ica (does not stall stream_compute).
     */
    void update_W_async(const float* W_new, cudaStream_t stream_ica) {
        CUDA_CHECK(cudaMemcpyAsync(d_W_, W_new,
                                   N_HUB_CHANNELS * N_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_ica));
    }

    /**
     * Process one frame:
     *   1. Apply ICA unmixing (filtered EEG → 12 sources)
     *   2. Run streaming à-trous DWT
     *   3. Map mu/beta power to theta angles in the NeuralPhaseVector
     *
     * Both kernels are launched on stream_compute and will be captured
     * into a CUDA Graph by the caller (SignalProcessor) for deterministic dispatch.
     *
     * @param d_filtered    Device ptr: [N_CHANNELS] IIR-filtered samples
     * @param d_frame       Device ptr: NeuralPhaseVector to update
     * @param stream        CUDA stream (stream_compute from SignalProcessor)
     */
    void process_frame(const float* d_filtered,
                       NeuralPhaseVector* d_frame,
                       cudaStream_t stream)
    {
        assert(initialized_);

        // Step 1: ICA unmixing — 64 channels → 12 sources
        ica_apply_kernel<<<1, N_HUB_CHANNELS, 0, stream>>>(
            d_filtered, d_W_, d_sources_);

        // Step 2: à-trous DWT + theta extraction — 12 sources → 12 theta angles
        // Block size = 32 (one warp), 12 active threads
        dwt_extract_theta_kernel<<<1, 32, 0, stream>>>(
            d_sources_, d_state_, d_frame);
    }

    void shutdown() {
        if (!initialized_) return;
        cudaFree(d_state_);
        cudaFree(d_sources_);
        cudaFree(d_W_);
        initialized_ = false;
    }

    // ── CUDA Graph capture helpers ────────────────────────────────────────────

    /**
     * Returns the device pointers needed by the graph capture in
     * SignalProcessor. The graph captures kernel launches, not device
     * pointer values — so these must remain stable for the session lifetime.
     */
    float*           d_W()       const { return d_W_;       }
    float*           d_sources() const { return d_sources_; }
    DWTChannelState* d_state()   const { return d_state_;   }

private:
    bool              initialized_ = false;
    DWTChannelState*  d_state_     = nullptr;
    float*            d_sources_   = nullptr;
    float*            d_W_         = nullptr;
};

// =============================================================================
// Unit-test hook — compile only when requested
// =============================================================================

#ifdef KHAOS_BUILD_DWT_TEST
/**
 * Host-side DWT test: generates a synthetic 250 Hz sine wave (beta band)
 * and a 10 Hz sine wave (mu band), processes 512 samples through the DWT,
 * and verifies that the extracted theta angle reflects beta dominance (theta > π/2).
 *
 * Expected result:
 *   After warm-up (~96 samples to fill delay lines), theta should converge to
 *   a value > π/2 for a pure beta signal, and < π/2 for a pure mu signal.
 */
#include <cmath>
#include <cstdio>
#include <vector>

void test_dwt_frequency_discrimination() {
    DWTExtractor extractor;
    extractor.init();

    // Allocate minimal host/device buffers for testing
    float  h_filtered[N_CHANNELS]     = {};
    float* d_filtered                 = nullptr;
    NeuralPhaseVector  h_frame        = {};
    NeuralPhaseVector* d_frame        = nullptr;
    cudaMalloc(&d_filtered, N_CHANNELS * sizeof(float));
    cudaMalloc(&d_frame,    sizeof(NeuralPhaseVector));

    // Identity-like W: hub 0 → channel 0, rest inactive
    // This makes hub-0 theta track channel-0 frequency
    // (Other hubs receive silence and will have theta ≈ π/2 due to ε domination)

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int N_SAMPLES = 512;
    const float FS      = 1000.0f;

    // Test 1: pure beta (25 Hz)
    printf("DWT test: pure 25 Hz (beta band) — expected theta > π/2\n");
    CUDA_CHECK(cudaMemset(d_frame, 0, sizeof(NeuralPhaseVector)));
    for (int n = 0; n < N_SAMPLES; ++n) {
        float beta_sample = sinf(2.0f * CUDART_PI_F * 25.0f * n / FS);
        h_filtered[0] = beta_sample;
        CUDA_CHECK(cudaMemcpy(d_filtered, h_filtered,
                               N_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
        extractor.process_frame(d_filtered, d_frame, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&h_frame, d_frame, sizeof(NeuralPhaseVector),
                           cudaMemcpyDeviceToHost));
    printf("  theta[0] = %.4f rad (π/2 = %.4f, π = %.4f)\n",
           h_frame.theta[0], CUDART_PI_F / 2, CUDART_PI_F);
    bool beta_ok = h_frame.theta[0] > CUDART_PI_F / 2;
    printf("  %s\n", beta_ok ? "PASS" : "FAIL — theta should be > π/2 for beta");

    // Test 2: pure mu (10 Hz)
    printf("DWT test: pure 10 Hz (mu band) — expected theta < π/2\n");
    // Re-init state for clean test
    CUDA_CHECK(cudaMemset(extractor.d_state(), 0,
                           N_HUB_CHANNELS * sizeof(DWTChannelState)));
    CUDA_CHECK(cudaMemset(d_frame, 0, sizeof(NeuralPhaseVector)));
    for (int n = 0; n < N_SAMPLES; ++n) {
        float mu_sample = sinf(2.0f * CUDART_PI_F * 10.0f * n / FS);
        h_filtered[0] = mu_sample;
        CUDA_CHECK(cudaMemcpy(d_filtered, h_filtered,
                               N_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
        extractor.process_frame(d_filtered, d_frame, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&h_frame, d_frame, sizeof(NeuralPhaseVector),
                           cudaMemcpyDeviceToHost));
    printf("  theta[0] = %.4f rad\n", h_frame.theta[0]);
    bool mu_ok = h_frame.theta[0] < CUDART_PI_F / 2;
    printf("  %s\n", mu_ok ? "PASS" : "FAIL — theta should be < π/2 for mu");

    cudaStreamDestroy(stream);
    cudaFree(d_filtered);
    cudaFree(d_frame);
    extractor.shutdown();

    if (!beta_ok || !mu_ok) {
        fprintf(stderr, "DWT frequency discrimination test FAILED\n");
        exit(1);
    }
    printf("All DWT tests passed.\n");
}
#endif // KHAOS_BUILD_DWT_TEST
