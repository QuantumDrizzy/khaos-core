#pragma once
/**
 * feedback_engine.h — C API for the KHAOS Tactile Feedback Engine.
 *
 * Provides opaque C wrappers around FeedbackEngine (feedback_engine.cu) so
 * that main.cpp can drive the kernel without being compiled by nvcc.
 *
 * All functions are implemented in src/neuro/feedback_engine.cu.
 *
 * Data flow:
 *   Python bridge → ent_alpha[N_HUB_CHANNELS]  (quantum entanglement proximity)
 *   feedback_process()  → modulate_tactile_feedback kernel  (~3 µs on sm_89)
 *   feedback_sync_output() → TactileFeedbackOutput  (PWM duty + FM freq)
 *   FPGA driver thread → register bank (DMA, platform-specific)
 *
 * Safety bounds (enforced in-kernel, see ETHICS.md §II):
 *   PWM duty  ≤ 32767  (50% of 65535 → peak current ≤ STIM_ABSOLUTE_MAX_AMP/2)
 *   FM freq   ∈ [50, 300] Hz  (Pacinian corpuscle range)
 *   global_scale = 0  ⟹ all outputs zeroed  (circuit-breaker lockout)
 */

#ifndef ETHICS_COMPLIANT
#  error "feedback_engine.h requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "safety_constants.h"   // N_HUB_CHANNELS, STIM_ABSOLUTE_MAX_AMP
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TactileFeedbackOutput — host-visible feedback frame.
 *
 * Layout is byte-identical to the device-side TactileFeedbackFrame
 * (verified by static_assert in feedback_engine.cu).
 */
typedef struct __attribute__((packed)) TactileFeedbackOutput {
    uint16_t pwm_duty[N_HUB_CHANNELS];          // [0, 32767]  (50% full scale)
    float    fm_freq_hz[N_HUB_CHANNELS];         // [50, 300] Hz
    float    proximity_smoothed[N_HUB_CHANNELS]; // exp-smoothed fidelity (audit)
    uint64_t timestamp_ns;                       // hardware clock at bridge cycle
    uint32_t bridge_cycle;                       // monotonic bridge counter
    float    global_scale;                       // circuit-breaker α used
} TactileFeedbackOutput;

/** Opaque handle to the FeedbackEngine + its CUDA stream. */
typedef struct FeedbackHandle FeedbackHandle;

/**
 * Allocate GPU resources and create a dedicated cudaStreamNonBlocking stream.
 * Call once at startup, before the first feedback_process().
 *
 * @return  New handle.  Caller owns; must call feedback_destroy() at shutdown.
 */
FeedbackHandle* feedback_create_and_init(void);

/**
 * Process one bridge cycle (~10 Hz).  Non-blocking — returns before the
 * GPU modulation kernel completes.
 *
 * @param proximity    [N_HUB_CHANNELS] quantum fidelity/entanglement scores
 *                     from the Python bridge (0 → 1).  Typically ent_alpha[].
 *                     When landmark == "flow" and fidelity is high these
 *                     values rise, driving higher PWM duty and FM frequency.
 * @param global_scale  Circuit-breaker α ∈ [0,1].  0 = full lockout (all
 *                      outputs zeroed in-kernel, no FPGA stimulation).
 * @param timestamp_ns  Hardware clock timestamp (CLOCK_MONOTONIC_RAW ns).
 */
void feedback_process(FeedbackHandle* h,
                      const float*    proximity,
                      float           global_scale,
                      uint64_t        timestamp_ns);

/**
 * Synchronously retrieve the most-recent output frame.
 * Blocks until GPU modulation + D2H copy are both complete (typically ~5 µs).
 *
 * The returned pointer is valid until the next feedback_process() call.
 * Never returns NULL after a successful feedback_create_and_init().
 */
const TactileFeedbackOutput* feedback_sync_output(FeedbackHandle* h);

/** Graceful shutdown: flush GPU streams and free all GPU + host resources. */
void feedback_destroy(FeedbackHandle* h);

#ifdef __cplusplus
}
#endif
