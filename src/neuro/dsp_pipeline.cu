/**
 * dsp_pipeline.cu  —  C wrapper bridging main.cpp ↔ CUDA DSP pipeline
 *
 * Owns one SignalProcessor + one DWTExtractor and exposes them via a plain
 * C API (dsp_pipeline.h) so that main.cpp does not need to be compiled by
 * nvcc or include any CUDA headers.
 *
 * Execution order per frame (all on SignalProcessor::stream_compute()):
 *   1. iir_biquad_frame      (CUDA Graph node)
 *   2. compute_metrics        (CUDA Graph node)
 *   3. bp_leaky_integrator    (standalone kernel, after graph)
 *   4. ica_apply_kernel       (DWTExtractor::process_frame, step 1)
 *   5. dwt_extract_theta_kernel (DWTExtractor::process_frame, step 2)
 *   6. e_frame_ready_ event   (SignalProcessor signals d_frame_ is complete)
 *   7. D2H copy on s_transfer_ (triggered by dsp_request_theta_async)
 *
 * Build requirement: -DETHICS_COMPLIANT
 */

#ifndef ETHICS_COMPLIANT
#  error "dsp_pipeline.cu requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

// Pull in the full class definitions from the source files.
// This is acceptable here because dsp_pipeline.cu IS a CUDA translation unit
// and will be compiled by nvcc.  main.cpp never sees these headers.
#include "signal_processor.cu"   // defines SignalProcessor
#include "dwt.cu"                // defines DWTExtractor (same directory)
#include "../../include/dsp_pipeline.h"

struct DSPPipeline {
    SignalProcessor sp;
    DWTExtractor    dwt;
};

extern "C" {

DSPPipeline* dsp_create_and_init(void)
{
    auto* p = new DSPPipeline();
    p->sp.init();
    p->dwt.init();
    return p;
}

void dsp_process_frame(DSPPipeline*       pipeline,
                        const float*       raw_samples,
                        unsigned long long timestamp_ns,
                        float              alpha_blend)
{
    // IIR filter + metrics (captured CUDA Graph)
    pipeline->sp.process_frame(raw_samples,
                                static_cast<uint64_t>(timestamp_ns),
                                alpha_blend);

    // ICA unmixing + à-trous DWT + theta extraction
    // Runs on the same stream → sees IIR output, writes theta[] into d_frame_
    pipeline->dwt.process_frame(
        pipeline->sp.d_filt_samples(),
        pipeline->sp.d_frame(),
        pipeline->sp.stream_compute());
}

void dsp_request_theta_async(DSPPipeline* pipeline)
{
    pipeline->sp.request_theta_frame_async();
}

const NeuralPhaseVector* dsp_sync_theta(DSPPipeline* pipeline)
{
    return pipeline->sp.sync_theta_frame();
}

void dsp_destroy(DSPPipeline* pipeline)
{
    pipeline->sp.shutdown();
    pipeline->dwt.shutdown();
    delete pipeline;
}

} // extern "C"
