#pragma once
/**
 * khaos_bridge.h — C/C++ <-> CUDA boundary definitions
 *
 * Types and function declarations shared across the C++ sovereignty layer
 * and the CUDA DSP kernels. Include in both .cpp and .cu files.
 */
#include "safety_constants.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NeuralPhaseVector — the θ-Frame
 * Contract between the CUDA DSP pipeline and the CUDA-Q quantum kernel.
 * Must remain exactly 64 bytes (one cache line).
 */
typedef struct __attribute__((aligned(64))) NeuralPhaseVector {
    float    theta[MAX_QUBITS];  // rotation angles [0, 2π], one per qubit
    float    confidence;         // signal quality [0, 1]
    float    entropy_estimate;   // S(ρ) proxy from band-power ratio
    float    bp_index;           // Bereitschaftspotential accumulator [0, 1]
    float    alpha_blend;        // circuit-breaker α (written by host)
    uint64_t timestamp_ns;       // hardware timestamp
    uint8_t  _pad[4];
} NeuralPhaseVector;

/* Layout note: sizeof(NeuralPhaseVector) varies with MAX_QUBITS.
 * Alignment is 64 bytes (one cache line); size is not constrained. */

#ifdef __cplusplus
}
#endif

