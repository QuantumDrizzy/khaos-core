# khaos-core — Quantum Mirror BCI Kernel

Real-time EEG → quantum circuit brain-computer interface kernel.  
The **Quantum Mirror** app uses this to run a meditation/biofeedback trainer that reflects your neural state as a quantum circuit and maps it to tactile haptic feedback via a sovereign PCIe FPGA.

---

## Architecture

```
EEG Amplifier (64ch @ 1000 Hz)
        │  Lab Streaming Layer (LSL)
        │  Core 1 — SCHED_FIFO 90
        ▼
┌─────────────────────────────────┐
│  LSLConnector  (C++)            │  Async ring buffer (8 slots, 8 ms headroom)
│  SignalProcessor  (CUDA)        │  IIR Biquad SOS — 8–30 Hz bandpass
│  DWTExtractor     (CUDA)        │  à-trous D4 wavelet → θ angles + BP index
└────────────┬────────────────────┘
             │  NeuralPhaseVector (64 bytes, cache-aligned)
             │  Core 0 — main EEG loop
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
             │  ent_alpha[12], fidelity, landmark
             ▼
┌─────────────────────────────────┐
│  FeedbackEngine  (CUDA)         │  PWM duty (pauli-Z → [0, 32767])
│                                 │  FM freq (Pacinian range [50, 300] Hz)
└────────────┬────────────────────┘
             │  TactileFeedbackOutput — eventfd signal
             │  Core 2 — SCHED_FIFO 80
             ▼
┌─────────────────────────────────┐
│  FPGA Driver  (C++ / UIO)       │  PCIe BAR0 shadow registers
│  PWM_SHADOW[0..11]              │  Pauli-Z mapped duty per qubit channel
│  FM_SHADOW[0..11]               │  Q16.8 fixed-point Hz
│  SOVEREIGNTY_TOKEN              │  XOR-fold frame hash (FPGA verifies in RTL)
│  COMMIT latch                   │  Atomic shadow→DAC in one RTL cycle
└─────────────────────────────────┘
             │  Core 3 — watchdog
             ▼
     SovereigntyMonitor kill-switch
     (STATUS FAULT / SOV_FAIL → g_running = false)
```

### Thread isolation map

| Core | Thread | Policy | Priority |
|---|---|---|---|
| 0 | Main EEG loop | SCHED_OTHER | — |
| 1 | LSL pull | SCHED_FIFO | 90 |
| 2 | FPGA driver | SCHED_FIFO | 80 |
| 3 | Watchdog | SCHED_OTHER | — |

---

## Stack

| Layer | Technology |
|---|---|
| EEG capture | Lab Streaming Layer (LSL) |
| DSP | CUDA — IIR biquad SOS (10 sections), à-trous D4 DWT |
| Quantum simulation | NVIDIA CUDA-Q (cudaq) |
| Graphene forward model | Python + NumPy + SciPy |
| Tactile feedback | CUDA modulation kernel → PCIe FPGA (PWM + FM) |
| Post-quantum crypto | CRYSTALS-Kyber-1024 (liboqs) |
| Audit log | SHA-256 chained log (OpenSSL EVP or builtin) |
| Build | CMake 3.26+, C++17, CUDA 12 |
| Target GPU | RTX 4xxx/5xxx (sm_89, Ada Lovelace / Blackwell) |

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

# For real FPGA output
UIO kernel module     modprobe uio_pci_generic
/dev/uio0             PCIe BAR0 exposed via Linux UIO driver
```

---

## Build

```bash
cd khaos-core

# 1. Generate IIR filter coefficients (requires scipy)
python3 scripts/gen_coefficients.py

# 2. Configure (stub FPGA mode — no hardware required)
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
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Override: `89` = Ada, `86` = Ampere |
| `KHAOS_FPGA_ENABLED` | `0` | `1` = real UIO/mmap BAR0; `0` = heap stub (default) |
| `KHAOS_SHA256_BUILTIN` | auto | Portable SHA-256 fallback (no OpenSSL) |

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

### Console output (10 Hz)

```
[MIRROR] NOMINAL     fid=0.923  lm=flow          conf=0.88  bp=0.12  pwm= 72.4%  fm=218.3Hz  α=1.00  qid=0
```

| Field | Meaning |
|---|---|
| `state` | Circuit breaker: NOMINAL / DEGRADED / PANIC / RECOVERING |
| `fid` | Quantum fidelity against held state — RECOVERING→NOMINAL criterion (>0.85) |
| `lm` | Active landmark: `rest`, `focus`, `flow`, `calm` |
| `conf` | ICA signal confidence [0, 1] |
| `bp` | Bereitschaftspotential index — pre-cognitive intent signal |
| `pwm` | Mean PWM duty across 12 hub channels (% of 32767) |
| `fm` | Mean FM frequency across 12 hub channels (Hz) |
| `α` | Circuit-breaker global scale (0 = PANIC lockout) |
| `qid` | Quantum Intent Divergence flag |

---

## FPGA Register Map (PCIe BAR0)

```
Offset   Width  R/W  Register
0x000    4 B    W    PWM_SHADOW[0]   lower 16 bits = duty [0, 32767]
0x004    4 B    W    PWM_SHADOW[1]
...
0x02C    4 B    W    PWM_SHADOW[11]
0x030    4 B    W    FM_SHADOW[0]    Q16.8 fixed-point Hz
...
0x05C    4 B    W    FM_SHADOW[11]
0x060    4 B    W    SOVEREIGNTY_TOKEN   lower 16 bits = XOR-fold frame hash
0x064    4 B    W    COMMIT              write 0x1 → atomic latch shadow→DAC
0x068    4 B    R    STATUS              ACK | FAULT | SOV_FAIL | OVERRUN
0x06C    4 B    R    FRAME_COUNTER       monotonic DAC cycle counter
```

Signal encoding:
- **PWM**: `<Z_i> = 1 − 2·proximity_smoothed_i`, `duty = (<Z_i>+1)/2 × 32767`
- **FM**: `reg = round(freq_hz × 256)` (Q16.8)
- **Sovereignty token**: `XOR-fold(bridge_cycle ⊕ 0xDEADBEEF, pwm_duty[], fm_freq_hz[]) & 0xFFFF`

In stub mode (`KHAOS_FPGA_ENABLED=0`), all writes go to a heap buffer and are logged to stderr. Activate real hardware with `-DKHAOS_FPGA_ENABLED=1` once the FPGA is on the PCIe bus.

---

## Ethics & Safety

khaos-core enforces neurorights at the compiler level — the build **will not compile** without `ETHICS_COMPLIANT=ON`.

Four non-negotiable principles (see `ETHICS.md` for full rationale):

1. **Mental Privacy** — raw EEG never leaves the device
2. **Mental Integrity** — no autonomous stimulation; all dissipation is user-initiated
3. **Psychological Continuity** — no reward signals; feedback is informative, not evaluative
4. **Cognitive Sovereignty** — SHA-256 chained audit log + FPGA hardware kill-switch (5 ms timeout)

Safety bounds enforced in-kernel and at the register layer:
- PWM duty ≤ 32767 (50% of 65535 → peak current ≤ `STIM_ABSOLUTE_MAX_AMP/2`)
- FM freq ∈ [50, 300] Hz (Pacinian corpuscle range)
- `global_scale = 0` (PANIC) → all FPGA shadow registers zeroed before COMMIT
- `STATUS FAULT | SOV_FAIL` → software kill-switch armed within one STATUS poll cycle

To amend any principle: edit `ETHICS.md` with rationale and review date, then rebuild.

---

## Project Structure

```
khaos-core/
├── CMakeLists.txt               Master build (ETHICS gate, CUDA/CUDA-Q/liboqs detection)
├── ETHICS.md                    Neurorights manifesto
├── include/
│   ├── khaos_bridge.h           NeuralPhaseVector, shared C/CUDA types
│   ├── safety_constants.h       STIM_ABSOLUTE_MAX_AMP, N_HUB_CHANNELS, …
│   ├── dsp_pipeline.h           DSPPipeline C API
│   ├── lsl_connector.h          EEGFrameSlot, LSLHandle C API
│   ├── feedback_engine.h        TactileFeedbackOutput, FeedbackHandle C API
│   ├── fpga_driver.h            BAR0 register map, FPGAHandle C API
│   └── sha256.h                 SHA-256 interface
├── scripts/
│   ├── gen_coefficients.py      Generates IIR SOS coefficients via scipy
│   └── init_khaos.sh            Project scaffold / dependency checker
├── src/
│   ├── main.cpp                 Closed-loop orchestrator: EEG → bridge → FPGA
│   ├── graphene/
│   │   └── dirac_emulator.py   Fermi-Dirac forward model → ent_alpha[]
│   ├── io/
│   │   └── fpga_driver.cpp      UIO/mmap PCIe BAR0 driver (stub by default)
│   ├── neuro/
│   │   ├── signal_processor.cu  CUDA IIR biquad SOS + pinned ring buffer
│   │   ├── dwt.cu               CUDA à-trous D4 wavelet → theta[], bp_index
│   │   ├── dsp_pipeline.cu      CUDA pipeline driver (includes signal_processor + dwt)
│   │   ├── feedback_engine.cu   CUDA PWM + FM modulation kernel
│   │   └── lsl_connector.cpp    LSL acquisition thread + synthetic fallback
│   ├── quantum/
│   │   ├── circuits.py          CUDA-Q 12-qubit circuit + SWAP test + LatentSpaceNavigator
│   │   └── mirror_bridge.py     Python bridge process (JSON-line stdin/stdout)
│   └── security/
│       ├── sha256.cpp           SHA-256 (OpenSSL EVP + portable FIPS 180-4 fallback)
│       └── sovereignty_monitor.cpp  Audit log chain, dissipation gate, kill-switch
└── tests/
    ├── bench/bench_latency.cu   End-to-end latency benchmark
    └── unit/
        ├── test_iir_filter.cu   IIR filter SNR + composite signal tests (8/8)
        └── test_dwt.cu          DWT reconstruction fidelity tests
```

---

## Neural Phase Vector

The 64-byte struct that flows through the entire pipeline:

```cpp
struct NeuralPhaseVector {
    float    theta[12];        // Ry rotation angles [0, 2π] per qubit
    float    confidence;       // ICA signal quality [0, 1]
    float    entropy_estimate; // Normalised Shannon entropy
    float    bp_index;         // Bereitschaftspotential accumulator
    float    alpha_blend;      // Circuit-breaker blend factor
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

`global_scale` output per state: NOMINAL=1.0, DEGRADED=0.6, PANIC=0.0, RECOVERING=0.3.  
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
