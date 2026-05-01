#pragma once
/**
 * fpga_driver.h — C API for the KHAOS PCIe FPGA Tactile Driver.
 *
 * PCIe BAR0 register map (112 bytes, 0x000–0x070):
 *
 *   Offset   Width  R/W  Description
 *   ──────   ─────  ───  ─────────────────────────────────────────────────────
 *   0x000    4 B    W    PWM_SHADOW[0]   lower 16 bits = duty [0, 32767]
 *   0x004    4 B    W    PWM_SHADOW[1]
 *   ...
 *   0x02C    4 B    W    PWM_SHADOW[11]
 *   0x030    4 B    W    FM_SHADOW[0]    Q16.8 fixed-point Hz [50, 300]
 *   0x034    4 B    W    FM_SHADOW[1]
 *   ...
 *   0x05C    4 B    W    FM_SHADOW[11]
 *   0x060    4 B    W    SOVEREIGNTY_TOKEN   lower 16 bits = XOR-fold hash
 *   0x064    4 B    W    COMMIT              write 0x1 → atomic latch shadow→DAC
 *   0x068    4 B    R    STATUS              ACK | FAULT | SOV_FAIL | OVERRUN
 *   0x06C    4 B    R    FRAME_COUNTER       monotonic DAC cycle counter
 *
 * Write protocol (per bridge cycle, ~10 Hz):
 *   1. Compute sovereignty token (XOR-fold of bridge_cycle, pwm_duty[], fm_freq_hz[]).
 *   2. Write PWM_SHADOW[0..N_HUB_CHANNELS-1].
 *   3. Write FM_SHADOW[0..N_HUB_CHANNELS-1].
 *   4. Write SOVEREIGNTY_TOKEN.
 *   5. wmb() — store-store barrier.
 *   6. Write 0x1 to COMMIT — FPGA atomically latches shadow→DAC in one RTL cycle.
 *
 * Fire-and-forget: fpga_write_frame() does NOT wait for ACK.  Call
 * fpga_read_status() on the NEXT cycle (~100 ms later) to check STATUS.
 *
 * Stub mode (default — KHAOS_FPGA_ENABLED not defined):
 *   fpga_open() allocates a heap shadow buffer instead of mmap'ing BAR0.
 *   All register writes are logged to stderr (FPGA_STUB_LOG).
 *   fpga_read_status() returns synthetic ACK with no fault bits.
 *   This mode is always active in --dry-run.
 *
 * Real mode (KHAOS_FPGA_ENABLED == 1):
 *   fpga_open() opens the UIO character device and mmap's BAR0.
 *   The kernel UIO driver exposes the physical BAR0 range via /dev/uio0.
 *
 * Safety invariants enforced in fpga_write_frame():
 *   • PWM duty is re-clamped to [0, 32767] after pauli_z_to_pwm().
 *   • FM freq is re-clamped to [50, 300] Hz before Q16.8 encoding.
 *   • When out->global_scale == 0 (PANIC), all shadow registers are zeroed
 *     regardless of frame content — hardware lockout at register level.
 *   • STATUS FAULT or SOV_FAIL detected on the next cycle → arms kill-switch.
 *
 * Thread model:
 *   Producer: closed-loop EEG thread — calls fpga_write_frame() indirectly by
 *             signalling eventfd after feedback_sync_output().
 *   Consumer: fpga_driver_thread (Core 2, SCHED_FIFO 80) — epoll_wait →
 *             fpga_write_frame → fpga_read_status (next cycle).
 *
 * See also: ETHICS.md §II (stimulation safety bounds).
 */

#ifndef ETHICS_COMPLIANT
#  error "fpga_driver.h requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "safety_constants.h"   // N_HUB_CHANNELS, STIM_ABSOLUTE_MAX_AMP
#include "feedback_engine.h"    // TactileFeedbackOutput
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// PCIe BAR0 register offsets
// =============================================================================

/** PWM shadow register for channel @p ch (0 ≤ ch < N_HUB_CHANNELS). */
#define FPGA_REG_PWM_SHADOW(ch)   (0x000u + (uint32_t)(ch) * 4u)

/** FM shadow register for channel @p ch (0 ≤ ch < N_HUB_CHANNELS). */
#define FPGA_REG_FM_SHADOW(ch)    (0x030u + (uint32_t)(ch) * 4u)

/** Sovereignty token register: lower 16 bits = XOR-fold of frame hash. */
#define FPGA_REG_SOV_TOKEN        0x060u

/**
 * COMMIT register.
 * Write 0x1 to atomically latch all shadow registers into the DAC in one
 * RTL clock cycle.  The FPGA verifies SOVEREIGNTY_TOKEN before applying.
 */
#define FPGA_REG_COMMIT           0x064u

/** STATUS register (read-only).  See FPGA_STATUS_* bit masks below. */
#define FPGA_REG_STATUS           0x068u

/** FRAME_COUNTER register (read-only monotonic DAC cycle counter). */
#define FPGA_REG_FRAME_COUNTER    0x06Cu

/** Total BAR0 footprint in bytes. */
#define FPGA_BAR0_SIZE            0x070u

// =============================================================================
// STATUS register bit masks
// =============================================================================

/** Previous COMMIT was accepted and DAC cycle completed. */
#define FPGA_STATUS_ACK           0x0001u

/** Hardware fault: overcurrent, overtemperature, or rail failure. */
#define FPGA_STATUS_FAULT         0x0002u

/** Sovereignty token mismatch: FPGA rejected the most recent frame. */
#define FPGA_STATUS_SOV_FAIL      0x0004u

/** COMMIT written before the previous DAC cycle finished. */
#define FPGA_STATUS_OVERRUN       0x0008u

// =============================================================================
// FPGAStatus — decoded result returned by fpga_read_status()
// =============================================================================

typedef struct FPGAStatus {
    uint32_t raw;        ///< Raw STATUS register value
    uint32_t frame_ctr;  ///< FRAME_COUNTER value at read time
    int      fault;      ///< Non-zero if FAULT or SOV_FAIL bit is set
    int      overrun;    ///< Non-zero if OVERRUN bit is set
} FPGAStatus;

/** Opaque handle returned by fpga_open(). */
typedef struct FPGAHandle FPGAHandle;

// =============================================================================
// C API
// =============================================================================

/**
 * Open the UIO device and memory-map BAR0.
 *
 * In stub mode (KHAOS_FPGA_ENABLED not defined), @p uio_dev is ignored and a
 * heap-allocated shadow buffer is used.  Pass NULL to make stub mode explicit.
 *
 * @param uio_dev  Path to UIO char device (e.g. "/dev/uio0"), or NULL for stub.
 * @return  New handle.  Caller owns; must pass to fpga_close() at shutdown.
 *          Aborts on allocation or mmap failure.
 */
FPGAHandle* fpga_open(const char* uio_dev);

/**
 * Write one TactileFeedbackOutput frame to the FPGA and fire the COMMIT latch.
 *
 * Fire-and-forget — returns before the DAC cycle completes.  Call
 * fpga_read_status() on the next cycle to verify STATUS.
 *
 * When @p out->global_scale == 0.0 (circuit-breaker PANIC), all shadow
 * registers are zeroed before COMMIT regardless of frame contents.
 *
 * @param h    Handle from fpga_open().
 * @param out  Feedback frame from feedback_sync_output().
 */
void fpga_write_frame(FPGAHandle* h, const TactileFeedbackOutput* out);

/**
 * Read the STATUS and FRAME_COUNTER registers and decode them.
 *
 * Intended to be called on the cycle following fpga_write_frame() to detect
 * hardware faults without blocking the real-time write path.
 *
 * @param h  Handle from fpga_open().
 * @return   Decoded FPGAStatus.
 */
FPGAStatus fpga_read_status(FPGAHandle* h);

/**
 * Flush pending writes and release all resources (file descriptor + mmap).
 * Safe to call with @p h == NULL.
 */
void fpga_close(FPGAHandle* h);

#ifdef __cplusplus
}
#endif
