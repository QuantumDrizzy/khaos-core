/**
 * test_iir_filter.cu — khaos-core IIR Filter SNR + Phase Coherence Test
 *
 * Standalone CUDA unit test.  Includes signal_processor.cu directly
 * (not linked against khaos_core) so it can inspect d_filt_samples().
 *
 * Filter under test:
 *   10th-order Butterworth bandpass 8–30 Hz, fs=1000 Hz (10 SOS sections)
 *   Stopband specification: ≥ -60 dB at 50 Hz (powerline fundamental)
 *   Measured:               -63.04 dB at 50 Hz, -82.0 dB at 60 Hz
 *
 * ── Test suite ────────────────────────────────────────────────────────────
 *
 *  #1  Single-tone gain — 10 Hz (μ passband edge)    : gain ≥ 0.30
 *  #2  Single-tone gain — 25 Hz (β passband centre)  : gain ≥ 0.50
 *  #3  Single-tone gain — 50 Hz (powerline stopband) : gain ≤ 0.05
 *
 *  #4  Composite signal -60 dB attenuation
 *      Input : x[n] = ⅓·sin(2π·10n/fs) + ⅓·sin(2π·25n/fs) + ⅓·sin(2π·50n/fs)
 *      BH-windowed Goertzel at 50 Hz → effective gain ≤ 0.001  (-60 dB w.r.t. ⅓ amplitude)
 *      BH-windowed Goertzel at 10 Hz → gain ≥ 0.25             (passband preserved)
 *      BH-windowed Goertzel at 25 Hz → gain ≥ 0.45             (passband preserved)
 *      NOTE: rectangular Goertzel fails this test (spectral leakage gives -44 dB);
 *            Blackman-Harris window reduces sidelobes to -92 dB → -63.03 dB result.
 *
 *  #5  Phase coherence at 10 Hz
 *      Two consecutive steady-state windows of N_COH samples each.
 *      Phase advance error: |Δφ_measured − Δφ_expected| ≤ 5°
 *      Amplitude stability: |A₂/A₁ − 1| ≤ 1%
 *
 *  #6  Phase coherence at 25 Hz (same protocol)
 *
 * ── Measurement method ────────────────────────────────────────────────────
 *
 *  Goertzel algorithm: O(N) DFT at a single frequency.
 *  For N samples, k = round(f·N/fs), the Goertzel gives:
 *     X(f) = Σ_{n=0}^{N-1} x[n]·e^{-j2πkn/N}
 *  Amplitude = |X(f)| / (N/2)   (single-sided normalisation)
 *
 *  Windowed Goertzel (Blackman-Harris, -92 dB sidelobes):
 *  Used for composite-signal measurements where non-integer bin positions
 *  would cause spectral leakage from the passband into the stopband bin.
 *  w[n] = 0.35875 − 0.48829·cos(2πn/(N-1)) + 0.14128·cos(4πn/(N-1))
 *                  − 0.01168·cos(6πn/(N-1))
 *  Amplitude = 2·|X_windowed| / Σ w[n]   (coherent gain normalisation)
 *
 * Build (standalone):
 *   nvcc -O2 -DETHICS_COMPLIANT=1 -I../../include test_iir_filter.cu \
 *        -o test_iir_filter
 */

#define ETHICS_COMPLIANT 1
#include "../../src/neuro/signal_processor.cu"

#include <cmath>
#include <cstdio>
#include <cstring>

// ── constants ─────────────────────────────────────────────────────────────────
static constexpr float K_FS       = 1000.0f;
static constexpr float K_PI       = 3.14159265358979323846f;
static constexpr int   N_WARMUP   = 3000;    // >> group delay at all poles
static constexpr int   N_MEASURE  = 1000;    // single-tone RMS window
static constexpr int   N_GOERTZEL = 2048;    // composite-signal DFT window
static constexpr int   N_COH      = 500;     // coherence window (500 ms)

// ── helpers ───────────────────────────────────────────────────────────────────

/** Sync compute stream and read channel-0 filtered sample. */
static float read_ch0(SignalProcessor& sp)
{
    cudaStreamSynchronize(sp.stream_compute());
    float y = 0.0f;
    cudaMemcpy(&y, sp.d_filt_samples(), sizeof(float), cudaMemcpyDeviceToHost);
    return y;
}

/**
 * Goertzel DFT at a single frequency bin.
 *
 * Computes the complex DFT coefficient X[k] at the bin nearest to freq_hz
 * for the N-sample sequence x[0..N-1].
 *
 * Returns: (real, imag) of X[k], and the corresponding single-sided amplitude
 *   amplitude = sqrt(re^2+im^2) / (N/2)
 *
 * Reference: Lyons, "Understanding Digital Signal Processing", §13.19
 */
static void goertzel(const float* x, int N, float freq_hz,
                     float* out_re, float* out_im, float* out_amp)
{
    const float k     = freq_hz * (float)N / K_FS;
    const float omega = 2.0f * K_PI * k / (float)N;
    const float coeff = 2.0f * cosf(omega);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;

    for (int n = 0; n < N; ++n) {
        s0 = x[n] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    *out_re = s1 - s2 * cosf(omega);
    *out_im =      s2 * sinf(omega);
    *out_amp = sqrtf(*out_re * *out_re + *out_im * *out_im) / ((float)N / 2.0f);
}

/**
 * Blackman-Harris windowed Goertzel DFT at a single frequency bin.
 *
 * Equivalent to goertzel() but pre-multiplies each sample by a 4-term
 * Blackman-Harris window before accumulating.  This reduces spectral
 * sidelobes to -92 dB, preventing passband energy from leaking into
 * stopband measurements when the target frequency is not on an integer bin.
 *
 * Normalisation:  amplitude = 2·|X_windowed| / window_sum
 * (coherent gain normalisation, matches rectangular Goertzel within ~0.01 dB
 *  at bin-centred frequencies).
 */
static void goertzel_bh(const float* x, int N, float freq_hz,
                        float* out_re, float* out_im, float* out_amp)
{
    const float k     = freq_hz * (float)N / K_FS;
    const float omega = 2.0f * K_PI * k / (float)N;
    const float coeff = 2.0f * cosf(omega);

    // Blackman-Harris coefficients (4-term, -92 dB sidelobe level)
    static constexpr float BH_A0 = 0.35875f;
    static constexpr float BH_A1 = 0.48829f;
    static constexpr float BH_A2 = 0.14128f;
    static constexpr float BH_A3 = 0.01168f;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    float wsum = 0.0f;

    for (int n = 0; n < N; ++n) {
        const float phase = 2.0f * K_PI * (float)n / (float)(N - 1);
        const float w = BH_A0
                      - BH_A1 * cosf(phase)
                      + BH_A2 * cosf(2.0f * phase)
                      - BH_A3 * cosf(3.0f * phase);
        wsum += w;
        s0 = x[n] * w + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    *out_re  = s1 - s2 * cosf(omega);
    *out_im  =      s2 * sinf(omega);
    *out_amp = 2.0f * sqrtf(*out_re * *out_re + *out_im * *out_im) / wsum;
}

/**
 * Feed n_frames of the supplied input pattern through SignalProcessor and
 * collect output into h_out[0..n_frames-1] from channel 0.
 * Returns number of samples actually collected.
 */
static int run_and_collect(SignalProcessor& sp,
                            const float* h_in,   // N_CHANNELS per frame
                            int           n_frames,
                            float*        h_out)
{
    for (int i = 0; i < n_frames; ++i) {
        sp.process_frame(h_in + (long long)i * N_CHANNELS,
                         (uint64_t)i * 1000000ULL, 1.0f);
        h_out[i] = read_ch0(sp);
    }
    return n_frames;
}

// ── Test 1-3: single-tone gain ─────────────────────────────────────────────────

struct GainCase {
    float freq_hz;
    float min_gain;
    float max_gain;
    const char* label;
};

static const GainCase k_gain_cases[] = {
    { 10.0f, 0.30f, 2.0f,  "μ passband edge  ( 8–12 Hz)" },
    { 25.0f, 0.50f, 2.0f,  "β passband centre(15–30 Hz)" },
    { 50.0f, 0.0f,  0.05f, "powerline stopband (50 Hz) " },
};

static float measure_gain_single(float freq_hz)
{
    SignalProcessor sp;
    sp.init();

    // Build a 3000-warmup + 1000-measure block of input
    const int N_TOTAL = N_WARMUP + N_MEASURE;
    const float omega = 2.0f * K_PI * freq_hz / K_FS;

    // Allocate and build input matrix [N_TOTAL × N_CHANNELS]
    float* h_in  = new float[(long long)N_TOTAL * N_CHANNELS]();
    float* h_out = new float[N_TOTAL]();

    for (int n = 0; n < N_TOTAL; ++n) {
        float x = sinf(omega * (float)n);
        for (int c = 0; c < N_CHANNELS; ++c) h_in[(long long)n * N_CHANNELS + c] = x;
    }

    run_and_collect(sp, h_in, N_TOTAL, h_out);
    sp.shutdown();

    // RMS over measurement window only
    double pow_out = 0.0;
    for (int i = N_WARMUP; i < N_TOTAL; ++i) {
        pow_out += (double)h_out[i] * h_out[i];
    }
    delete[] h_in;
    delete[] h_out;

    const double rms_out = sqrt(pow_out / N_MEASURE);
    const double rms_in  = 1.0 / sqrt(2.0);  // unit-amplitude sine
    return (float)(rms_out / rms_in);
}

// ── Test 4: composite signal, Goertzel -60 dB ─────────────────────────────────

struct GoertzelCase {
    float freq_hz;
    float min_amp;
    float max_amp;
    float in_amp;       // amplitude of that component in the composite input
    const char* label;
};

static const GoertzelCase k_goertzel_cases[] = {
    // Passband components must pass through
    { 10.0f, 0.25f, 2.0f, 1.0f/3.0f, "10 Hz passband (composite)" },
    { 25.0f, 0.45f, 2.0f, 1.0f/3.0f, "25 Hz passband (composite)" },
    // Powerline component must be crushed to ≤ -60 dB
    { 50.0f, 0.0f,  0.001f,  1.0f/3.0f, "50 Hz stopband -60 dB     " },
};

static void measure_composite(float* g10, float* g25, float* g50)
{
    SignalProcessor sp;
    sp.init();

    const int N_TOTAL = N_WARMUP + N_GOERTZEL;
    const float w10 = 2.0f * K_PI * 10.0f / K_FS;
    const float w25 = 2.0f * K_PI * 25.0f / K_FS;
    const float w50 = 2.0f * K_PI * 50.0f / K_FS;
    const float A   = 1.0f / 3.0f;

    float* h_in  = new float[(long long)N_TOTAL * N_CHANNELS]();
    float* h_out = new float[N_TOTAL]();

    for (int n = 0; n < N_TOTAL; ++n) {
        float x = A * (sinf(w10 * n) + sinf(w25 * n) + sinf(w50 * n));
        for (int c = 0; c < N_CHANNELS; ++c) h_in[(long long)n * N_CHANNELS + c] = x;
    }

    run_and_collect(sp, h_in, N_TOTAL, h_out);
    sp.shutdown();

    // Windowed Goertzel (Blackman-Harris) on the measurement window.
    // BH window (-92 dB sidelobes) prevents passband energy from leaking
    // into the 50 Hz stopband bin when 10 Hz / 25 Hz / 50 Hz all fall on
    // non-integer bins for N=2048, fs=1000 Hz.
    float re, im, amp;
    goertzel_bh(h_out + N_WARMUP, N_GOERTZEL, 10.0f, &re, &im, &amp);
    *g10 = amp / A;   // normalise by component amplitude → effective gain

    goertzel_bh(h_out + N_WARMUP, N_GOERTZEL, 25.0f, &re, &im, &amp);
    *g25 = amp / A;

    goertzel_bh(h_out + N_WARMUP, N_GOERTZEL, 50.0f, &re, &im, &amp);
    *g50 = amp / A;

    delete[] h_in;
    delete[] h_out;
}

// ── Test 5-6: phase coherence ─────────────────────────────────────────────────

struct CohResult {
    float phase_error_deg;   // |Δφ_measured − Δφ_expected| in degrees
    float amp_stability;     // |A2/A1 − 1| (fraction)
};

static CohResult measure_coherence(float freq_hz)
{
    SignalProcessor sp;
    sp.init();

    const int N_TOTAL = N_WARMUP + 2 * N_COH;
    const float omega = 2.0f * K_PI * freq_hz / K_FS;

    float* h_in  = new float[(long long)N_TOTAL * N_CHANNELS]();
    float* h_out = new float[N_TOTAL]();

    for (int n = 0; n < N_TOTAL; ++n) {
        float x = sinf(omega * (float)n);
        for (int c = 0; c < N_CHANNELS; ++c) h_in[(long long)n * N_CHANNELS + c] = x;
    }

    run_and_collect(sp, h_in, N_TOTAL, h_out);
    sp.shutdown();

    // Window 1: samples [N_WARMUP, N_WARMUP+N_COH)
    float re1, im1, a1;
    goertzel(h_out + N_WARMUP, N_COH, freq_hz, &re1, &im1, &a1);
    float phi1 = atan2f(im1, re1);

    // Window 2: samples [N_WARMUP+N_COH, N_WARMUP+2*N_COH)
    float re2, im2, a2;
    goertzel(h_out + N_WARMUP + N_COH, N_COH, freq_hz, &re2, &im2, &a2);
    float phi2 = atan2f(im2, re2);

    delete[] h_in;
    delete[] h_out;

    // Expected phase advance over N_COH samples at freq_hz
    // Δφ_expected = 2π · freq_hz · N_COH / fs  (mod 2π)
    const float dphi_expected = fmodf(
        2.0f * K_PI * freq_hz * (float)N_COH / K_FS,
        2.0f * K_PI);

    // Wrap measured advance to [-π, +π]
    float dphi_measured = phi2 - phi1;
    while (dphi_measured >  K_PI) dphi_measured -= 2.0f * K_PI;
    while (dphi_measured < -K_PI) dphi_measured += 2.0f * K_PI;

    float phase_error_rad = fabsf(dphi_measured - dphi_expected);
    // Also check the complementary wrap
    float alt = fabsf(dphi_measured - dphi_expected + 2.0f * K_PI);
    if (alt < phase_error_rad) phase_error_rad = alt;
    alt = fabsf(dphi_measured - dphi_expected - 2.0f * K_PI);
    if (alt < phase_error_rad) phase_error_rad = alt;

    CohResult r;
    r.phase_error_deg = phase_error_rad * 180.0f / K_PI;
    r.amp_stability   = (a1 > 1e-6f) ? fabsf(a2 / a1 - 1.0f) : 1.0f;
    return r;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║    khaos-core IIR Filter SNR + Phase Coherence Test             ║\n");
    printf("║    10th-order Butterworth bandpass 8–30 Hz, fs = 1000 Hz        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    int n_pass = 0, n_fail = 0;

    // ── Tests 1–3: single-tone gain ─────────────────────────────────────────
    printf("  ── Single-tone gain (warmup=%d, measure=%d) ──\n",
           N_WARMUP, N_MEASURE);
    for (int i = 0; i < 3; ++i) {
        const GainCase& tc = k_gain_cases[i];
        printf("    %5.1f Hz  ", tc.freq_hz); fflush(stdout);
        const float g  = measure_gain_single(tc.freq_hz);
        const float db = 20.0f * log10f(g > 1e-9f ? g : 1e-9f);
        const bool ok  = (g >= tc.min_gain) && (g <= tc.max_gain);
        ok ? ++n_pass : ++n_fail;
        printf("gain=%8.5f  %+7.2f dB  [%.3f,%.3f]  %-30s  %s\n",
               g, db, tc.min_gain, tc.max_gain, tc.label,
               ok ? "PASS" : "*** FAIL ***");
    }

    // ── Test 4: composite -60 dB ────────────────────────────────────────────
    printf("\n  ── Composite signal (Goertzel, N=%d) ──\n", N_GOERTZEL);
    printf("    Input: x = ⅓·sin(10Hz) + ⅓·sin(25Hz) + ⅓·sin(50Hz)\n");

    float g10, g25, g50;
    printf("    Computing … "); fflush(stdout);
    measure_composite(&g10, &g25, &g50);
    printf("done\n");

    const float db50 = 20.0f * log10f(g50 > 1e-9f ? g50 : 1e-9f);
    const GoertzelCase gc[] = {
        { 10.0f, 0.25f, 2.0f, 1.0f/3.0f, "10 Hz passband (composite)" },
        { 25.0f, 0.45f, 2.0f, 1.0f/3.0f, "25 Hz passband (composite)" },
        { 50.0f, 0.0f,  0.001f, 1.0f/3.0f, "50 Hz -60 dB spec        " },
    };
    const float gains[3] = {g10, g25, g50};
    for (int i = 0; i < 3; ++i) {
        const float g  = gains[i];
        const float db = 20.0f * log10f(g > 1e-9f ? g : 1e-9f);
        const bool ok  = (g >= gc[i].min_amp) && (g <= gc[i].max_amp);
        ok ? ++n_pass : ++n_fail;
        printf("    %5.1f Hz  gain=%8.5f  %+7.2f dB  %-30s  %s\n",
               gc[i].freq_hz, g, db, gc[i].label,
               ok ? "PASS" : "*** FAIL ***");
    }
    printf("    50 Hz powerline rejection: %+.2f dB  (specification: ≤ -60 dB)\n", db50);

    // ── Tests 5–6: phase coherence ──────────────────────────────────────────
    printf("\n  ── Phase coherence (two windows of %d samples) ──\n", N_COH);
    const float coh_freqs[] = { 10.0f, 25.0f };
    for (float f : coh_freqs) {
        printf("    %5.1f Hz  ", f); fflush(stdout);
        CohResult r = measure_coherence(f);
        const bool ok_phase = (r.phase_error_deg <= 5.0f);
        const bool ok_amp   = (r.amp_stability   <= 0.01f);
        const bool ok       = ok_phase && ok_amp;
        ok ? ++n_pass : ++n_fail;
        printf("phase_err=%6.3f°  amp_dev=%7.4f%%  [≤5°, ≤1%%]  %s\n",
               r.phase_error_deg, r.amp_stability * 100.0f,
               ok ? "PASS" : "*** FAIL ***");
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    const int n_total = 3 + 3 + 2;
    printf("\n  %d / %d tests passed.\n", n_pass, n_total);

    if (n_fail > 0) {
        printf("\n  FAILURE:\n");
        if (g50 > 0.001f) {
            printf("    50 Hz attenuation %.1f dB — specification requires ≤ -60 dB\n",
                   20.0f * log10f(g50));
            printf("    → Check N_SOS_SECTIONS (must be 10) and __constant__ arrays\n");
        }
        printf("\n");
        return 1;
    }

    printf("  All assertions satisfied.\n");
    printf("  IIR filter meets SNR, -60 dB attenuation, and phase coherence specs.\n\n");
    return 0;
}
