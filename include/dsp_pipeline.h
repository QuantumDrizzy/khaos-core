#pragma once
/**
 * dsp_pipeline.h — C++ host interface to the CUDA DSP pipeline.
 *
 * Provides opaque C wrappers around SignalProcessor (signal_processor.cu)
 * and DWTExtractor (dwt.cu) so that main.cpp can drive the pipeline without
 * including CUDA headers or being compiled by nvcc.
 *
 * All functions are defined in dsp_pipeline.cu (thin forwarding layer).
 *
 * Usage in main.cpp:
 *   #include "dsp_pipeline.h"
 *   DSPPipeline* dsp = dsp_create();
 *   dsp_init(dsp);
 *   // per-frame (1 ms):
 *   dsp_process_frame(dsp, raw_samples, timestamp_ns, alpha_blend);
 *   dsp_request_async(dsp);
 *   const NeuralPhaseVector* frame = dsp_sync(dsp);
 *   dsp_destroy(dsp);
 */

#include "khaos_bridge.h"   // NeuralPhaseVector, N_CHANNELS, etc.

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to the combined SignalProcessor + DWTExtractor pipeline. */
typedef struct DSPPipeline DSPPipeline;

/** Allocate + initialise GPU resources.  Call once at startup. */
DSPPipeline* dsp_create_and_init(void);

/**
 * Process one EEG frame (1 ms, 64 channels).
 * @param pipeline     Handle returned by dsp_create_and_init().
 * @param raw_samples  Pointer to N_CHANNELS floats (µV).
 * @param timestamp_ns Hardware clock timestamp for this frame.
 * @param alpha_blend  Current circuit-breaker alpha ∈ [0,1].
 */
void dsp_process_frame(DSPPipeline* pipeline,
                        const float* raw_samples,
                        unsigned long long timestamp_ns,
                        float alpha_blend);

/** Start async D2H copy of the θ-Frame. */
void dsp_request_theta_async(DSPPipeline* pipeline);

/**
 * Block until the θ-Frame is on the host.
 * Returns a pointer to the NeuralPhaseVector (valid until the next call).
 */
const NeuralPhaseVector* dsp_sync_theta(DSPPipeline* pipeline);

/** Graceful shutdown: flush GPU streams, free all resources. */
void dsp_destroy(DSPPipeline* pipeline);

#ifdef __cplusplus
}
#endif
