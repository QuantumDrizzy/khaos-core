#pragma once
/**
 * lsl_connector.h — C API for the KHAOS LSL EEG Connector.
 *
 * Exposes EEGFrameSlot and an opaque LSLHandle so that main.cpp can
 * drive EEG acquisition without depending on LSL or C++ class headers.
 *
 * Implementation: src/neuro/lsl_connector.cpp
 *
 * Thread model:
 *   Producer: LSL pull thread (fills internal ring buffer, 1000 Hz)
 *   Consumer: main loop (pops via lsl_try_pop, also 1000 Hz)
 *
 * Ring buffer capacity: RING_FRAMES (8) slots = 8 ms of headroom.
 * Overflow is signalled by the internal drop counter.
 */

#ifndef ETHICS_COMPLIANT
#  error "lsl_connector.h requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "safety_constants.h"   // N_CHANNELS
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * EEGFrameSlot — one acquisition frame from the LSL pull thread.
 *
 * Layout must remain byte-identical to EEGFrame in signal_processor.cu.
 * A static_assert in main.cpp (KHAOS_VERIFY_FRAME_LAYOUT) validates
 * the sizes at compile time.
 */
typedef struct EEGFrameSlot {
    float    samples[N_CHANNELS];   // µV, one per EEG channel
    uint64_t timestamp_ns;          // CLOCK_MONOTONIC_RAW nanoseconds
    uint32_t frame_index;           // monotonic pull counter
    uint8_t  _pad[4];
} EEGFrameSlot;

/** Opaque handle returned by lsl_create(). */
typedef struct LSLHandle LSLHandle;

/**
 * Allocate and configure the LSL connector.
 *
 * @param stream_name  LSL stream name to search for (e.g. "EEG", "Emotiv EEG").
 * @param stream_type  LSL stream type (e.g. "EEG").
 * @return  New handle.  Caller owns; must call lsl_stop() then lsl_destroy().
 */
LSLHandle* lsl_create(const char* stream_name, const char* stream_type);

/**
 * Start the acquisition thread.
 *
 * Attempts LSL discovery for LSL_DISCOVERY_TIMEOUT_S seconds.
 * If no stream is found (or use_synthetic != 0), falls back to the
 * synthetic sinusoidal generator (10 Hz μ + 25 Hz β + noise).
 *
 * @param use_synthetic  Non-zero to force synthetic mode (e.g. --dry-run).
 */
void lsl_start(LSLHandle* h, int use_synthetic);

/**
 * Non-blocking frame pop.
 *
 * Copies the oldest available frame into *out.
 * @return  1 if a frame was available and copied, 0 if the ring is empty.
 */
int lsl_try_pop(LSLHandle* h, EEGFrameSlot* out);

/**
 * Print a jitter statistics summary to stderr.
 * Safe to call at any time — uses atomic reads, no lock.
 */
void lsl_print_stats(const LSLHandle* h);

/** Non-zero if the connector is using the synthetic generator. */
int lsl_is_synthetic(const LSLHandle* h);

/**
 * Pin the LSL pull thread to a CPU core and set SCHED_FIFO scheduling.
 *
 * Must be called BEFORE lsl_start().  The settings are applied inside the
 * pull thread at startup via pthread_setaffinity_np + pthread_setschedparam.
 *
 * Requires CAP_SYS_NICE (or root) for SCHED_FIFO; the call is silently
 * downgraded to SCHED_OTHER if permission is denied.
 *
 * @param cpu_core       CPU core index to pin the thread to (e.g. 1).
 *                       Pass -1 to skip affinity setting.
 * @param sched_priority SCHED_FIFO real-time priority [1, 99].
 *                       Pass -1 to skip scheduling class change.
 */
void lsl_set_realtime(LSLHandle* h, int cpu_core, int sched_priority);

/** Signal the pull thread to stop and join it.  Safe to call multiple times. */
void lsl_stop(LSLHandle* h);

/** Free all resources.  Must call lsl_stop() first. */
void lsl_destroy(LSLHandle* h);

#ifdef __cplusplus
}
#endif
