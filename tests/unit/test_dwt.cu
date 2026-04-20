/**
 * test_dwt.cu — khaos-core DWT Feature Extractor Unit Test
 *
 * Standalone CUDA unit test.  Includes dwt.cu directly
 * (not linked against khaos_core).
 *
 * The DWTExtractor's default W matrix is identity-like:
 *   hub[i] ← channel[i]   for i ∈ [0, N_HUB_CHANNELS)
 * So feeding a sinusoid on channel 0 drives only theta[0]; all other
 * hub channels receive zero and converge to theta ≈ 0.
 *
 * à-trous D4 frequency bands at fs = 1000 Hz:
 *   Level 5 detail:  15.6 – 31.25 Hz  ≈ β (beta)
 *   Level 6 detail:   7.8 – 15.6  Hz  ≈ μ (mu)
 *
 * Theta mapping:
 *   θ = 2·arcsin( √(P_β / (P_β + P_μ + ε)) )
 *   β dominant  →  θ > π/2  (gate towards |1⟩)
 *   μ dominant  →  θ < π/2  (gate towards |0⟩)
 *
 * Test suite:
 *   #1  10 Hz pure sinusoid   → theta[0] < π/2   (μ-band dominant)
 *   #2  20 Hz pure sinusoid   → theta[0] > π/2   (β-band dominant)
 *   #3  Alternating burst 10→20 Hz: theta follows frequency switch
 *   #4  Quantum fidelity: mu-state and beta-state are distinguishable
 *         F = |⟨ψ_μ|ψ_β⟩|² = ∏_q cos²((θ_μ[q]−θ_β[q])/2) < 0.95
 *
 * Warm-up rationale:
 *   Level-6 à-trous delay = (N_D4_TAPS−1)·2^5 = 96 samples.
 *   Power window = 32 samples.
 *   Total warm-up = 800 samples ≫ max( 96, 32 ).
 *
 * Build (standalone):
 *   nvcc -O2 -DETHICS_COMPLIANT=1 -I../../include test_dwt.cu \
 *        -o test_dwt
 */

#define ETHICS_COMPLIANT 1
#include "../../src/neuro/dwt.cu"

#include <cmath>
#include <cstdio>
#include <cstring>

// ── compile-time constants ────────────────────────────────────────────────────
static constexpr float K_PI   = 3.14159265358979323846f;
static constexpr float K_PI_2 = K_PI / 2.0f;
static constexpr float K_FS   = 1000.0f;   // sample rate Hz

// ── helpers ───────────────────────────────────────────────────────────────────

/**
 * Feed n_frames of a unit-amplitude sine at freq_hz on channel 0
 * through the DWTExtractor (all other channels silent).
 *
 * @param extractor  Initialised DWTExtractor.
 * @param freq_hz    Sinusoid frequency.
 * @param n_frames   Number of samples to push.
 * @param n_offset   Phase offset — use the caller's frame counter so
 *                   consecutive calls produce a continuous waveform.
 * @param d_filt     Device buffer [N_CHANNELS] (scratch, reused per frame).
 * @param d_frame    Device buffer [NeuralPhaseVector] (output).
 * @param stream     CUDA stream.
 * @param reset      If true, zero DWT state before running.
 */
static void run_burst(DWTExtractor&      extractor,
                      float              freq_hz,
                      int                n_frames,
                      int                n_offset,
                      float*             d_filt,
                      NeuralPhaseVector* d_frame,
                      cudaStream_t       stream,
                      bool               reset = false)
{
    if (reset) {
        cudaMemset(extractor.d_state(), 0,
                   N_HUB_CHANNELS * sizeof(DWTChannelState));
        cudaMemset(d_frame, 0, sizeof(NeuralPhaseVector));
    }

    float h_filt[N_CHANNELS];
    const float omega = 2.0f * K_PI * freq_hz / K_FS;

    for (int i = 0; i < n_frames; ++i) {
        memset(h_filt, 0, sizeof(h_filt));
        h_filt[0] = sinf(omega * (float)(n_offset + i));

        cudaMemcpy(d_filt, h_filt,
                   N_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
        extractor.process_frame(d_filt, d_frame, stream);
    }
    cudaStreamSynchronize(stream);
}

/** Copy NeuralPhaseVector from device and return it. */
static NeuralPhaseVector read_frame(NeuralPhaseVector* d_frame)
{
    NeuralPhaseVector h = {};
    cudaMemcpy(&h, d_frame, sizeof(NeuralPhaseVector), cudaMemcpyDeviceToHost);
    return h;
}

/**
 * Compute quantum fidelity between two product states:
 *
 *   |ψ_a⟩ = ⊗_q [cos(θ_a[q]/2)|0⟩ + sin(θ_a[q]/2)|1⟩]
 *
 *   F = |⟨ψ_a|ψ_b⟩|²
 *     = ( ∏_q cos((θ_a[q] − θ_b[q])/2) )²
 *
 * This is the per-qubit inner product squared over all N_HUB_CHANNELS qubits.
 */
static float quantum_fidelity(const float* theta_a, const float* theta_b,
                               int n_qubits)
{
    double inner = 1.0;
    for (int q = 0; q < n_qubits; ++q) {
        inner *= (double)cosf((theta_a[q] - theta_b[q]) * 0.5f);
    }
    return (float)(inner * inner);
}

// ── test runner ───────────────────────────────────────────────────────────────

int main(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║    khaos-core DWT Feature Extractor Unit Test                   ║\n");
    printf("║    à-trous Daubechies D4, 7 levels, fs = 1000 Hz               ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  Frequency bands:  β = Level 5 (15.6–31.25 Hz)\n");
    printf("                    μ = Level 6  (7.8–15.6  Hz)\n");
    printf("  Warm-up: 800 samples;  Power window: %d samples\n\n",
           POWER_WINDOW);

    // ── device resources ─────────────────────────────────────────────────────
    DWTExtractor extractor;
    extractor.init();

    float*             d_filt  = nullptr;
    NeuralPhaseVector* d_frame = nullptr;
    cudaStream_t       stream  = nullptr;

    cudaMalloc(&d_filt,  N_CHANNELS * sizeof(float));
    cudaMalloc(&d_frame, sizeof(NeuralPhaseVector));
    cudaStreamCreate(&stream);

    const int N_WARMUP   = 800;
    const int N_ALT_HALF = 500;    // each half of the alternating burst

    int n_pass = 0, n_fail = 0;
    NeuralPhaseVector frame;

    // ── Test 1: 10 Hz (μ-band) → theta[0] < π/2 ─────────────────────────────
    {
        run_burst(extractor, 10.0f, N_WARMUP, 0, d_filt, d_frame, stream,
                  /*reset=*/true);
        frame = read_frame(d_frame);

        bool ok = (frame.theta[0] < K_PI_2);
        printf("  Test 1 — 10 Hz μ-band\n");
        printf("    theta[0] = %.4f rad  (expect < π/2 = %.4f)\n",
               frame.theta[0], K_PI_2);
        printf("    %s\n\n", ok ? "PASS" : "*** FAIL ***");
        ok ? ++n_pass : ++n_fail;
    }
    // Save theta vector for fidelity test
    float theta_mu[MAX_QUBITS];
    for (int q = 0; q < MAX_QUBITS; ++q) theta_mu[q] = frame.theta[q];

    // ── Test 2: 20 Hz (β-band) → theta[0] > π/2 ─────────────────────────────
    {
        run_burst(extractor, 20.0f, N_WARMUP, 0, d_filt, d_frame, stream,
                  /*reset=*/true);
        frame = read_frame(d_frame);

        bool ok = (frame.theta[0] > K_PI_2);
        printf("  Test 2 — 20 Hz β-band\n");
        printf("    theta[0] = %.4f rad  (expect > π/2 = %.4f)\n",
               frame.theta[0], K_PI_2);
        printf("    %s\n\n", ok ? "PASS" : "*** FAIL ***");
        ok ? ++n_pass : ++n_fail;
    }
    // Save theta vector for fidelity test
    float theta_beta[MAX_QUBITS];
    for (int q = 0; q < MAX_QUBITS; ++q) theta_beta[q] = frame.theta[q];

    // ── Test 3: Alternating burst — 10 Hz → 20 Hz ────────────────────────────
    {
        // Phase A: 500 ms at 10 Hz (μ dominant)
        run_burst(extractor, 10.0f, N_ALT_HALF, 0,
                  d_filt, d_frame, stream, /*reset=*/true);
        float theta_A = read_frame(d_frame).theta[0];

        // Phase B: 500 ms at 20 Hz (β dominant), state NOT reset
        // n_offset = N_ALT_HALF → phase-continuous waveform
        run_burst(extractor, 20.0f, N_ALT_HALF, N_ALT_HALF,
                  d_filt, d_frame, stream, /*reset=*/false);
        float theta_B = read_frame(d_frame).theta[0];

        bool ok_A = (theta_A < K_PI_2);
        bool ok_B = (theta_B > K_PI_2);
        bool ok   = ok_A && ok_B;

        printf("  Test 3 — Alternating burst 10 Hz → 20 Hz\n");
        printf("    Phase A (10 Hz): theta[0] = %.4f rad  (expect < π/2)  %s\n",
               theta_A, ok_A ? "✓" : "✗");
        printf("    Phase B (20 Hz): theta[0] = %.4f rad  (expect > π/2)  %s\n",
               theta_B, ok_B ? "✓" : "✗");
        printf("    %s\n\n", ok ? "PASS" : "*** FAIL ***");
        ok ? ++n_pass : ++n_fail;
    }

    // ── Test 4: Quantum fidelity — mu and beta states are distinguishable ─────
    {
        // F < 0.95 means the quantum states carry discriminable information
        float F = quantum_fidelity(theta_mu, theta_beta, MAX_QUBITS);
        bool  ok = (F < 0.95f);

        printf("  Test 4 — Quantum fidelity F = |⟨ψ_μ|ψ_β⟩|²\n");
        printf("    theta_μ[0]  = %.4f rad\n", theta_mu[0]);
        printf("    theta_β[0]  = %.4f rad\n", theta_beta[0]);
        printf("    F           = %.4f  (expect < 0.95)\n", F);
        printf("    %s\n\n", ok ? "PASS" : "*** FAIL ***");
        ok ? ++n_pass : ++n_fail;
    }

    // ── cleanup ───────────────────────────────────────────────────────────────
    cudaStreamDestroy(stream);
    cudaFree(d_filt);
    cudaFree(d_frame);
    extractor.shutdown();

    // ── summary ───────────────────────────────────────────────────────────────
    const int n_total = 4;
    printf("  %d / %d tests passed.\n", n_pass, n_total);

    if (n_fail > 0) {
        printf("\n  FAILURE: DWT frequency discrimination or fidelity check failed.\n");
        printf("  Possible causes:\n");
        printf("    • à-trous delay lines not fully flushed (increase N_WARMUP)\n");
        printf("    • POWER_WINDOW too small for frequency at 1000 Hz\n");
        printf("    • D4 coefficient precision loss (check float vs double)\n");
        printf("\n");
        return 1;
    }

    printf("  All assertions satisfied — DWT within specification.\n\n");
    return 0;
}
