# khaos-core — Quantum Mirror BCI Kernel

Real-time EEG → quantum circuit brain-computer interface kernel.  
The **Quantum Mirror** app uses this to run a meditation/biofeedback trainer that reflects your neural state as a quantum circuit.

---

## Architecture

```
EEG Amplifier (64ch @ 1000 Hz)
        │  Lab Streaming Layer (LSL)
        ▼
┌─────────────────────────────────┐
│  SignalProcessor  (CUDA)        │  IIR Biquad SOS — 8–30 Hz bandpass
│  DWTExtractor     (CUDA)        │  à-trous D4 wavelet → θ angles + BP index
└────────────┬────────────────────┘
             │  NeuralPhaseVector (64 bytes, cache-aligned)
             ▼
┌─────────────────────────────────┐
│  SovereigntyMonitor  (C++)      │  SHA-256 chained audit log + kill-switch
└────────────┬────────────────────┘
             │  JSON-line bridge @ 10 Hz
             ▼
┌─────────────────────────────────┐
│  mirror_bridge.py               │
│  ├─ KhaosBackend  (CUDA-Q)      │  12-qubit circuit + SWAP test fidelity
│  └─ DiracEmulator (Python)      │  Graphene Fermi-Dirac forward model
└────────────┬────────────────────┘
             │  ent_alpha[], fidelity, landmark
             ▼
     Haptic / Audio Feedback
```

---

## Stack

| Layer | Technology |
|---|---|
| EEG capture | Lab Streaming Layer (LSL) |
| DSP | CUDA — IIR biquad SOS, à-trous D4 DWT |
| Quantum simulation | NVIDIA CUDA-Q (cudaq) |
| Graphene forward model | Python + NumPy + SciPy |
| Post-quantum crypto | CRYSTALS-Kyber-1024 (liboqs) |
| Audit log | SHA-256 chained log (OpenSSL EVP or builtin) |
| Build | CMake 3.26+, C++17, CUDA 12 |
| Target GPU | RTX 4xxx/5xxx (sm_89, Ada Lovelace) |

---

## Prerequisites

```bash
# Required
CUDA Toolkit 12+      https://developer.nvidia.com/cuda-downloads
CMake 3.26+           https://cmake.org
Python 3.10+          with numpy, scipy

# Optional but recommended
CUDA-Q (cudaq)        https://developer.nvidia.com/cuda-q
liboqs               https://github.com/open-quantum-safe/liboqs
liblsl               https://github.com/sccn/liblsl
OpenSSL              (system package — apt/brew)
```

---

## Build

```bash
cd khaos-core

# 1. Generate IIR filter coefficients (requires scipy)
python3 scripts/gen_coefficients.py

# 2. Configure
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DETHICS_COMPLIANT=ON

# 3. Build
cmake --build build --parallel

# 4. Smoke test (no EEG hardware required)
./build/khaos_mirror --dry-run
```

### Build options

| Flag | Default | Description |
|---|---|---|
| `ETHICS_COMPLIANT` | `ON` | **Required.** Cannot be disabled. See `ETHICS.md`. |
| `CMAKE_CUDA_ARCHITECTURES` | `89` | Set to `86` for Ampere (RTX 3xxx) |
| `KHAOS_SHA256_BUILTIN` | auto | Use portable SHA-256 fallback (no OpenSSL) |

---

## Run

```bash
# Dry-run with synthetic EEG (CI / no hardware)
./build/khaos_mirror --dry-run

# Live with LSL EEG stream
./build/khaos_mirror \
    --stream "EEG" \
    --log    data/audit_logs/session.log \
    --bridge src/quantum/mirror_bridge.py

# Custom bridge script
./build/khaos_mirror --bridge path/to/mirror_bridge.py
```

### Output

```
[MIRROR] state=NOMINAL     fidelity=0.923 landmark=focus       ent_mean=0.671 bp=0.12 conf=0.88 qid=0 ts=4s
```

| Field | Meaning |
|---|---|
| `state` | Circuit breaker: NOMINAL / DEGRADED / PANIC / RECOVERING |
| `fidelity` | Quantum fidelity against held state — RECOVERING→NOMINAL criterion (>0.85) |
| `landmark` | Nearest mental state: `rest`, `focus`, `flow`, `calm` |
| `ent_mean` | Mean entanglement alpha — graphene coupling strength |
| `bp` | Bereitschaftspotential index — pre-cognitive intent signal |
| `conf` | ICA signal confidence [0, 1] |
| `qid` | Quantum Intent Divergence detected |

---

## Ethics & Safety

khaos-core enforces neurorights at the compiler level — the build **will not compile** without `ETHICS_COMPLIANT=ON`.

Four non-negotiable principles (see `ETHICS.md` for full rationale):

1. **Mental Privacy** — raw EEG never leaves the device
2. **Mental Integrity** — no autonomous stimulation; all dissipation is user-initiated
3. **Psychological Continuity** — no reward signals; feedback is informative, not evaluative
4. **Cognitive Sovereignty** — SHA-256 chained audit log + FPGA hardware kill-switch (5 ms timeout)

To amend any principle: edit `ETHICS.md` with rationale and review date, then rebuild.

---

## Project Structure

```
khaos-core/
├── CMakeLists.txt               Master build (ETHICS gate, CUDA/CUDA-Q/liboqs detection)
├── ETHICS.md                    Neurorights manifesto
├── include/
│   ├── khaos_bridge.h           NeuralPhaseVector, shared C/CUDA types
│   ├── safety_constants.h       STIM_ABSOLUTE_MAX_AMP, KILLSWITCH_TIMEOUT_MS
│   └── sha256.h                 SHA-256 interface
├── scripts/
│   ├── gen_coefficients.py      Generates IIR SOS coefficients via scipy
│   └── init_khaos.sh            Project scaffold / dependency checker
├── src/
│   ├── main.cpp                 Pipeline entry point, circuit breaker, Python bridge
│   ├── graphene/
│   │   └── dirac_emulator.py   Fermi-Dirac forward model → ent_alpha[]
│   ├── neuro/
│   │   ├── signal_processor.cu  CUDA IIR biquad SOS + pinned ring buffer
│   │   └── dwt.cu               CUDA à-trous D4 wavelet → theta[], bp_index
│   ├── quantum/
│   │   ├── circuits.py          CUDA-Q 12-qubit circuit + SWAP test + LatentSpaceNavigator
│   │   └── mirror_bridge.py     Python bridge process (JSON-line stdin/stdout)
│   └── security/
│       ├── sha256.cpp           SHA-256 (OpenSSL EVP + portable FIPS 180-4 fallback)
│       └── sovereignty_monitor.cpp  Audit log chain, dissipation gate, kill-switch
└── tests/
    ├── bench/bench_latency.cu   End-to-end latency benchmark
    └── unit/                    Unit tests (sovereignty, IIR filter, DWT)
```

---

## Neural Phase Vector

The 64-byte struct that flows through the entire pipeline:

```cpp
struct NeuralPhaseVector {
    float    theta[12];       // Ry rotation angles [0, 2π] per qubit
    float    confidence;      // ICA signal quality [0, 1]
    float    entropy_estimate;// Normalised Shannon entropy
    float    bp_index;        // Bereitschaftspotential accumulator
    float    alpha_blend;     // Circuit-breaker blend factor
    uint64_t timestamp_ns;
    uint8_t  _pad[4];
};
```

---

## Circuit Breaker

State machine running at 1000 Hz on the host:

```
NOMINAL ──(conf<0.5 for 200ms)──► DEGRADED
DEGRADED ──(conf<0.3 for 100ms OR bp>0.8)──► PANIC
PANIC ──(user dissipation OR timeout 5s)──► RECOVERING
RECOVERING ──(fidelity>0.85)──► NOMINAL
```

All transitions are logged to the sovereignty audit chain.

---

## Quantum Intent Divergence (QID)

When the system detects that the user's mental state has changed during a PANIC hold:
- Fidelity drop against held state
- High SNR (signal is clean, not noisy)
- Stable theta attractor (user is focused, just on something different)
- Rising bp_index (pre-cognitive shift)

On QID: hot-swap the held state to the new attractor rather than forcing RECOVERING.

---

## License

Research prototype. Not for clinical or medical use.  
All stimulation outputs are subject to the hard limits in `include/safety_constants.h` (50 µA max).
