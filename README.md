# KĦAOS-CORE

> Neurorights at the hardware level — not as a feature, as a constraint.

Closed-loop BCI kernel with embedded sovereignty architecture.  
The Python validation stack runs at 256 Hz · 4–64 channels · 12-qubit feature extraction.  
The C++/CUDA production kernel targets < 250 µs end-to-end at 1000 Hz · 64 channels.

**Status:** Python stack complete and instrumented. FPGA register logic complete (heap stub without hardware).  
ASIC graphene model: physics implemented, not validated. Real QPU: phase 3.0.

---

## what exists (not vaporware)

| Component | State | Evidence |
|-----------|-------|----------|
| EthicsGate + SHA-256 chained audit | **Complete** | Forensic ledger, not logging. Every state transition hashed and chained. |
| SovereigntyMonitor (C++) | **Complete** | Kill-switch as registered event. `#ifndef ETHICS_COMPLIANT` → fatal compile error. |
| 12-qubit invariant representation | **Complete** | 4-channel Muse 2 or 64-channel ADS1299 → identical 240-element theta vector. |
| DSP pipeline (CUDA) | **Complete** | IIR Butterworth order 10, à-trous D4 DWT. < 100 µs measured in kernels. |
| 50 µA stimulation cap | **Complete** | Triple independent layer: Python runtime + C++ `static_assert` (compile-time) + FPGA watchdog. |
| Muse 2 adapter | **Complete** | Live LSL/BlueMuse + synthetic fallback for CI. |
| FPGA driver (BAR0) | **Architecture ready** | Register map correct. Without hardware, writes route to heap buffer. |
| Kyber-1024 (PQC) | **Stub** | CMake infrastructure present, not active in pipeline. |
| Graphene ASIC forward model | **Stub** | Fermi-Dirac physics implemented, not validated against silicon. |
| Real QPU backend | **Roadmap** | Phase 3.0. CUDA-Q simulation only. |

---

## the three things that do not exist elsewhere

Every public BCI repo has an ethics section in the documentation.  
KĦAOS-CORE has ethics in the **binary**:

1. **Compiler-level enforcement.** `ETHICS_COMPLIANT` is not optional. The build fails without it.
2. **Forensic audit chain.** SHA-256 chained ledger of every sovereignty state transition — tamper-evident by construction.
3. **Electrode-density invariant quantum representation.** 4 channels or 64, the 12-qubit decode vector is the same. This is architectural, not a compression artifact.

---

## architecture

```
EEG (4–64ch) → LSL → SignalProcessor (CUDA, IIR SOS) → DWTExtractor (CUDA)
                                                    ↓
                                        NeuralPhaseVector (64 bytes)
                                                    ↓
                              ┌─────────────────────┴─────────────────────┐
                              ↓                                           ↓
                    SovereigntyMonitor (C++)                    mirror_bridge.py
                    SHA-256 chain + kill-switch                 CUDA-Q 12-qubit circuit
                              ↓                                           ↓
                         FPGA driver                          FeedbackEngine (CUDA)
                         (BAR0 stub)                          PWM + FM synthesis
```

**Latency budget:** DSP path (LSL → IIR → DWT → metrics) < 100 µs at 1000 Hz.  
Quantum circuit runs asynchronously (~40 ms) without blocking the real-time cadence.

---

## safety architecture

```
Layer 1 (Python):    runtime bounds checking, stimulation gate
Layer 2 (C++):       static_assert(STIM_ABSOLUTE_MAX_AMP <= 50.0f) — compile-time
Layer 3 (FPGA):      hardware watchdog, 5 ms timeout, independent of software
```

If `STATUS FAULT | SOV_FAIL` → `g_running = false` in one poll cycle.  
If `global_scale = 0` (PANIC) → all FPGA shadow registers zeroed before COMMIT.

---

## stack

| Layer | Technology |
|-------|-----------|
| EEG acquisition | Lab Streaming Layer (LSL) / SyntheticMuse2Adapter |
| DSP | CUDA — IIR biquad SOS (10 sections), à-trous D4 DWT |
| Quantum simulation | NVIDIA CUDA-Q (12-qubit circuit + SWAP test fidelity) |
| Security | SHA-256 chained audit (OpenSSL EVP + portable fallback) |
| Post-quantum crypto | CRYSTALS-Kyber-1024 (CMake stub, not active) |
| Feedback | CUDA modulation kernel → PCIe FPGA (PWM + FM, stub) |
| Build | CMake 3.26+, C++17, CUDA 12 |
| Target GPU | RTX 4xxx/5xxx (sm_89, Ada Lovelace / Blackwell) |

---

## build

```bash
git clone https://github.com/QuantumDrizzy/khaos-core.git
cd khaos-core

# Generate IIR coefficients
python3 scripts/gen_coefficients.py

# Configure (stub FPGA mode — no hardware required)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DETHICS_COMPLIANT=ON

# Build
cmake --build build --parallel

# Smoke test (synthetic EEG, no hardware)
./build/khaos_mirror --dry-run
```

`ETHICS_COMPLIANT` cannot be disabled. See `ETHICS.md` for amendment protocol.

---

## run

```bash
# Live with Muse 2
./build/khaos_mirror --stream "EEG" --bridge src/quantum/mirror_bridge.py

# Synthetic (CI / no hardware)
./build/khaos_mirror --dry-run
```

Console output at 10 Hz:
```
[MIRROR] NOMINAL  fid=0.923  lm=flow  conf=0.88  bp=0.12  pwm=72.4%  fm=218.3Hz  α=1.00
```

---

## for researchers

The 12-qubit representation is invariant to spatial sensor density.  
This is a deliberate architectural property, not a limitation.

**[→ Technical Paper v1.1](docs/KHAOS_CORE_Technical_Paper_v1.1.pdf)** — Dual-stack architecture, signal processing pipeline, 12-qubit feature map, Dirac-LPAS coupling (sec. 5.4), sovereignty architecture, and joint research proposal (sec. 9.2).

---

## license

Research prototype. Not for clinical or medical use.  
All stimulation outputs are subject to the hard limits in `include/safety_constants.h`.

MIT
