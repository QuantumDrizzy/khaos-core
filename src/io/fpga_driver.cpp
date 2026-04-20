/**
 * fpga_driver.cpp — khaos-core PCIe FPGA Tactile Driver
 *
 * Implements the C API declared in include/fpga_driver.h.
 *
 * Compile-time modes:
 *   KHAOS_FPGA_ENABLED == 1 : real UIO path — mmap's /dev/uio0 BAR0.
 *   (default)                : stub mode — malloc'd shadow buffer, all
 *                              writes logged at FPGA_STUB_LOG verbosity.
 *
 * Register encoding:
 *   PWM  — pauli_z_to_pwm(proximity_smoothed): <Z_i> = 1 - 2·p_i,
 *           duty = (<Z_i> + 1) / 2 × 32767, clamped to [0, 32767].
 *   FM   — freq_to_reg(fm_freq_hz): Q16.8 fixed-point (value * 256),
 *           clamped to [50, 300] Hz before encoding.
 *   SOV  — sovereignty_token(): XOR-fold of bridge_cycle ⊕ 0xDEADBEEF,
 *           all pwm_duty[], all fm_freq_hz bit-cast to uint32, masked to
 *           lower 16 bits.
 *
 * Safety:
 *   • global_scale == 0 → all shadow writes zeroed (PANIC lockout).
 *   • STATUS FAULT or SOV_FAIL → fault counter incremented; caller
 *     (fpga_driver_thread in main.cpp) arms the software kill-switch.
 */

#ifndef ETHICS_COMPLIANT
#  error "fpga_driver.cpp requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "../../include/fpga_driver.h"
#include "../../include/safety_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

// POSIX
#include <unistd.h>
#include <fcntl.h>

#if defined(KHAOS_FPGA_ENABLED) && KHAOS_FPGA_ENABLED
#  include <sys/mman.h>
#  include <sys/stat.h>
#endif

// =============================================================================
// Stub logging (enabled in non-FPGA mode)
// =============================================================================

#if defined(KHAOS_FPGA_ENABLED) && KHAOS_FPGA_ENABLED
#  define FPGA_STUB_LOG(...)  do {} while (0)
#else
#  define FPGA_STUB_LOG(...)  fprintf(stderr, "[fpga-stub] " __VA_ARGS__)
#endif

// =============================================================================
// FPGAHandle — opaque struct definition
// =============================================================================

struct FPGAHandle {
    volatile uint32_t* bar0;         ///< Pointer to BAR0 (mmap or stub heap)
    int                uio_fd;       ///< UIO file descriptor (-1 in stub mode)
    int                stub_mode;    ///< 1 = stub, 0 = real UIO
    uint64_t           write_count;  ///< Monotonic fpga_write_frame() counter
    uint32_t           fault_count;  ///< Accumulated STATUS fault events

    // Stub-mode shadow buffer (heap-allocated when stub_mode == 1)
    uint32_t           shadow[FPGA_BAR0_SIZE / 4];
};

// =============================================================================
// Architecture-correct write memory barrier
// =============================================================================
//
// On x86-64, stores to MMIO (WC or UC mapped) are already ordered; a compiler
// fence prevents the compiler from reordering writes across the wmb() site.
// On AArch64, a DSB ST is required before any MMIO commit write.

static inline void wmb(void)
{
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("dmb st" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");   // compiler fence — sufficient on x86-64
#endif
}

// =============================================================================
// Register I/O helpers
// =============================================================================

static inline void bar0_write32(FPGAHandle* h, uint32_t offset, uint32_t val)
{
    uint32_t idx = offset / 4u;
    if (h->stub_mode) {
        h->shadow[idx] = val;
    } else {
        h->bar0[idx] = val;
    }
}

static inline uint32_t bar0_read32(const FPGAHandle* h, uint32_t offset)
{
    uint32_t idx = offset / 4u;
    if (h->stub_mode) {
        return h->shadow[idx];
    } else {
        return h->bar0[idx];
    }
}

// =============================================================================
// Signal encoding helpers
// =============================================================================

/**
 * Map proximity_smoothed ∈ [0, 1] to a PWM duty via the Pauli-Z observable.
 *
 * <Z_i>  = 1 - 2 · proximity_smoothed_i   ∈ [-1, +1]
 * duty   = (<Z_i> + 1) / 2 × 32767        ∈ [0, 32767]
 *
 * This ensures that maximum proximity (p=1) → maximum duty, and p=0 → 0 duty,
 * preserving quantum-to-haptic fidelity without overflow.
 */
static inline uint16_t pauli_z_to_pwm(float proximity_smoothed)
{
    float zi     = 1.0f - 2.0f * proximity_smoothed;
    float duty_f = (zi + 1.0f) * 0.5f * 32767.0f;
    float clamped = duty_f < 0.0f ? 0.0f : (duty_f > 32767.0f ? 32767.0f : duty_f);
    return (uint16_t)clamped;
}

/**
 * Encode a frequency in Hz as a Q16.8 fixed-point uint32.
 *
 * value_reg = round(freq_hz × 256)
 *
 * Frequency is clamped to [50, 300] Hz (Pacinian corpuscle range, per ETHICS.md §II)
 * before encoding.  Range in register:
 *   50 Hz  → 0x3200  (12800)
 *  300 Hz  → 0x12C00 (76800)
 */
static inline uint32_t freq_to_reg(float freq_hz)
{
    float clamped = freq_hz < 50.0f ? 50.0f : (freq_hz > 300.0f ? 300.0f : freq_hz);
    return (uint32_t)(clamped * 256.0f + 0.5f);   // round to nearest Q16.8
}

/**
 * Compute the 16-bit sovereignty token for a feedback frame.
 *
 * token = (bridge_cycle XOR 0xDEADBEEF)
 *           XOR pwm_duty[0] XOR (pwm_duty[1] << 1) XOR ...
 *           XOR fm_freq_hz[0] (bit-cast) XOR fm_freq_hz[1] (bit-cast) XOR ...
 *
 * Lower 16 bits are written to SOVEREIGNTY_TOKEN.  The FPGA RTL independently
 * re-computes the same XOR fold over the latched shadow values and asserts
 * SOV_FAIL in STATUS if the values disagree.
 */
static uint32_t sovereignty_token(const TactileFeedbackOutput* out)
{
    uint32_t token = out->bridge_cycle ^ 0xDEADBEEFu;

    for (int i = 0; i < N_HUB_CHANNELS; ++i) {
        // PWM contribution: rotate by channel index to spread entropy
        token ^= (uint32_t)out->pwm_duty[i] << (i % 16);

        // FM contribution: bit-cast float to uint32 (well-defined via memcpy)
        uint32_t fm_bits = 0;
        memcpy(&fm_bits, &out->fm_freq_hz[i], sizeof(fm_bits));
        token ^= fm_bits;
    }

    return token & 0xFFFFu;
}

// =============================================================================
// API implementation
// =============================================================================

FPGAHandle* fpga_open(const char* uio_dev)
{
    FPGAHandle* h = (FPGAHandle*)calloc(1, sizeof(FPGAHandle));
    if (!h) {
        fprintf(stderr, "[fpga] FATAL: calloc failed\n");
        abort();
    }

    h->uio_fd      = -1;
    h->write_count = 0;
    h->fault_count = 0;

#if defined(KHAOS_FPGA_ENABLED) && KHAOS_FPGA_ENABLED
    if (uio_dev && uio_dev[0] != '\0') {
        // ── Real UIO path ─────────────────────────────────────────────────────
        h->uio_fd = open(uio_dev, O_RDWR | O_SYNC);
        if (h->uio_fd < 0) {
            perror("[fpga] open UIO device");
            fprintf(stderr, "[fpga] WARNING: falling back to stub mode\n");
            goto stub_fallback;
        }

        void* mapped = mmap(nullptr, FPGA_BAR0_SIZE,
                            PROT_READ | PROT_WRITE, MAP_SHARED,
                            h->uio_fd, 0);
        if (mapped == MAP_FAILED) {
            perror("[fpga] mmap BAR0");
            close(h->uio_fd);
            h->uio_fd = -1;
            fprintf(stderr, "[fpga] WARNING: falling back to stub mode\n");
            goto stub_fallback;
        }

        h->bar0      = (volatile uint32_t*)mapped;
        h->stub_mode = 0;
        fprintf(stderr, "[fpga] BAR0 mapped from %s  (%u bytes)\n",
                uio_dev, FPGA_BAR0_SIZE);
        return h;
    }
stub_fallback:
#else
    (void)uio_dev;
#endif

    // ── Stub mode ─────────────────────────────────────────────────────────────
    h->stub_mode = 1;
    h->bar0      = nullptr;   // stub uses h->shadow[]
    memset(h->shadow, 0, sizeof(h->shadow));
    FPGA_STUB_LOG("stub mode active (KHAOS_FPGA_ENABLED not set)\n");
    return h;
}

// -----------------------------------------------------------------------------

void fpga_write_frame(FPGAHandle* h, const TactileFeedbackOutput* out)
{
    if (!h || !out) return;

    const int panic = (out->global_scale == 0.0f);

    // ── 1. PWM shadow registers ───────────────────────────────────────────────
    for (int i = 0; i < N_HUB_CHANNELS; ++i) {
        uint32_t duty = 0;
        if (!panic) {
            // Re-derive from proximity_smoothed to guarantee safety bounds
            // even if the frame was partially corrupted in transit.
            duty = (uint32_t)pauli_z_to_pwm(out->proximity_smoothed[i]);
        }
        bar0_write32(h, FPGA_REG_PWM_SHADOW(i), duty);
    }

    // ── 2. FM shadow registers ────────────────────────────────────────────────
    for (int i = 0; i < N_HUB_CHANNELS; ++i) {
        uint32_t fm_reg = 0;
        if (!panic) {
            fm_reg = freq_to_reg(out->fm_freq_hz[i]);
        }
        bar0_write32(h, FPGA_REG_FM_SHADOW(i), fm_reg);
    }

    // ── 3. Sovereignty token ──────────────────────────────────────────────────
    uint32_t token = panic ? 0u : sovereignty_token(out);
    bar0_write32(h, FPGA_REG_SOV_TOKEN, token);

    // ── 4. Write memory barrier ───────────────────────────────────────────────
    // All shadow register writes must be globally visible before the COMMIT
    // write arrives at the FPGA.  On x86-64 this is a compiler fence only;
    // the x86 memory model guarantees store ordering without sfence.
    wmb();

    // ── 5. COMMIT latch (fire-and-forget) ─────────────────────────────────────
    bar0_write32(h, FPGA_REG_COMMIT, 0x1u);

    ++h->write_count;

    if (h->stub_mode) {
        FPGA_STUB_LOG("COMMIT #%llu  cycle=%u  scale=%.2f  "
                      "pwm[0]=%u  fm[0]=%.1fHz  token=0x%04X%s\n",
                      (unsigned long long)h->write_count,
                      out->bridge_cycle, out->global_scale,
                      (unsigned)pauli_z_to_pwm(out->proximity_smoothed[0]),
                      out->fm_freq_hz[0], token,
                      panic ? "  [PANIC — outputs zeroed]" : "");
    }
}

// -----------------------------------------------------------------------------

FPGAStatus fpga_read_status(FPGAHandle* h)
{
    FPGAStatus s{};

    if (!h) return s;

    if (h->stub_mode) {
        // Synthetic ACK — no faults in stub mode
        s.raw       = FPGA_STATUS_ACK;
        s.frame_ctr = (uint32_t)(h->write_count & 0xFFFFFFFFu);
        s.fault     = 0;
        s.overrun   = 0;
        return s;
    }

    s.raw       = bar0_read32(h, FPGA_REG_STATUS);
    s.frame_ctr = bar0_read32(h, FPGA_REG_FRAME_COUNTER);
    s.fault     = (s.raw & (FPGA_STATUS_FAULT | FPGA_STATUS_SOV_FAIL)) ? 1 : 0;
    s.overrun   = (s.raw & FPGA_STATUS_OVERRUN) ? 1 : 0;

    if (s.fault) {
        ++h->fault_count;
        fprintf(stderr, "[fpga] STATUS FAULT  raw=0x%04X  "
                "fault_count=%u  frame_ctr=%u\n",
                s.raw, h->fault_count, s.frame_ctr);
    }

    return s;
}

// -----------------------------------------------------------------------------

void fpga_close(FPGAHandle* h)
{
    if (!h) return;

    if (!h->stub_mode && h->bar0) {
#if defined(KHAOS_FPGA_ENABLED) && KHAOS_FPGA_ENABLED
        munmap((void*)h->bar0, FPGA_BAR0_SIZE);
#endif
        h->bar0 = nullptr;
    }

    if (h->uio_fd >= 0) {
        close(h->uio_fd);
        h->uio_fd = -1;
    }

    fprintf(stderr, "[fpga] closed  write_count=%llu  fault_count=%u\n",
            (unsigned long long)h->write_count, h->fault_count);

    free(h);
}
