#!/usr/bin/env bash
# =============================================================================
# init_khaos.sh — khaos-core project scaffold
#
# Usage:
#   chmod +x scripts/init_khaos.sh
#   ./scripts/init_khaos.sh [--force]
#
# Creates the full directory structure, stubs for missing headers,
# and verifies that required tools are present on PATH.
# Safe to re-run — existing files are never overwritten unless --force is given.
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[khaos]${RESET} $*"; }
success() { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error()   { echo -e "${RED}[error]${RESET} $*" >&2; }
die()     { error "$*"; exit 1; }

FORCE=0
for arg in "$@"; do [[ "$arg" == "--force" ]] && FORCE=1; done

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
info "Repository root: ${BOLD}$ROOT${RESET}"
cd "$ROOT"

# =============================================================================
# 1. Tool checks
# =============================================================================
info "Checking required tools..."

check_tool() {
    local tool="$1"; local hint="$2"
    if command -v "$tool" &>/dev/null; then
        success "$tool ($(command -v "$tool"))"
    else
        warn "$tool not found — $hint"
    fi
}

check_tool cmake    "Install cmake >= 3.26 (https://cmake.org)"
check_tool nvcc     "Install CUDA Toolkit >= 12.0 (https://developer.nvidia.com/cuda-downloads)"
check_tool python3  "Install Python >= 3.10"
check_tool git      "Install git"

# CUDA-Q (cudaq) — optional at scaffold time, required at build time
if python3 -c "import cudaq" 2>/dev/null; then
    success "cudaq Python package found"
else
    warn "cudaq not importable — install with: pip install cudaq"
    warn "See https://developer.nvidia.com/cuda-q for CUDA-Q setup"
fi

# liboqs — optional at scaffold time
if pkg-config --exists liboqs 2>/dev/null; then
    success "liboqs found ($(pkg-config --modversion liboqs))"
else
    warn "liboqs not found via pkg-config"
    warn "Install from: https://github.com/open-quantum-safe/liboqs"
    warn "  cmake -DBUILD_SHARED_LIBS=ON . && make && sudo make install"
fi

echo ""

# =============================================================================
# 2. Directory structure
# =============================================================================
info "Creating directory structure..."

dirs=(
    # Source modules
    "src/neuro"         # EEG capture, IIR filter, DWT
    "src/quantum"       # CUDA-Q circuits, kernels
    "src/graphene"      # Dirac emulator / forward model
    "src/security"      # Sovereignty monitor, fuzzy extractor

    # Headers
    "include"

    # Documentation
    "docs"

    # Python tooling
    "scripts"

    # Build artifacts (gitignored)
    "build"

    # Tests
    "tests/unit"
    "tests/bench"
    "tests/integration"

    # Calibration data (gitignored — user-specific)
    "data/calibration"
    "data/audit_logs"

    # Generated SOS / DWT coefficients
    "coefficients"
)

for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
        success "  $d (exists)"
    else
        mkdir -p "$d"
        success "  $d (created)"
    fi
done

echo ""

# =============================================================================
# 3. Gitignore
# =============================================================================
info "Writing .gitignore..."
GITIGNORE="$ROOT/.gitignore"

write_if_absent() {
    local file="$1"; local content="$2"
    if [[ -f "$file" && "$FORCE" -eq 0 ]]; then
        warn "  $file exists, skipping (--force to overwrite)"
    else
        printf '%s\n' "$content" > "$file"
        success "  $file"
    fi
}

write_if_absent "$GITIGNORE" "# Build artifacts
build/
*.o
*.a
*.so
*.cubin
*.ptx

# Calibration data — never commit; user-specific and privacy-sensitive
data/calibration/
data/audit_logs/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Editor
.vscode/
.idea/
*.swp
*.swo

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile

# Generated coefficient files (regenerate with scripts/gen_coefficients.py)
coefficients/*.json
coefficients/*.npy

# Secrets — must never be committed
*.jks
*.keystore
*.pem
*.key
user_private_key*"

echo ""

# =============================================================================
# 4. Header stubs (created only if absent)
# =============================================================================
info "Creating header stubs..."

write_stub() {
    local path="$1"; local content="$2"
    if [[ -f "$path" && "$FORCE" -eq 0 ]]; then
        success "  $path (exists)"
    else
        mkdir -p "$(dirname "$path")"
        printf '%s\n' "$content" > "$path"
        success "  $path (created)"
    fi
}

write_stub "include/safety_constants.h" \
'#pragma once
/**
 * safety_constants.h — khaos-core
 * Hardware safety limits. Values here are enforced by static_assert()
 * in sovereignty_monitor.cpp. Do not increase STIM_ABSOLUTE_MAX_AMP
 * without IRB-level review and ETHICS.md amendment.
 */

// Maximum stimulation output — microamperes (µA)
// 50 µA is the conservative safe limit for transcutaneous stimulation.
// Sub-sensory EEG biofeedback applications stay below 1 µA.
constexpr float STIM_ABSOLUTE_MAX_AMP = 50.0f;

// Hardware watchdog timeout — milliseconds
// If the GPU pipeline does not write a heartbeat within this window,
// the FPGA firmware cuts all DAC outputs autonomously.
constexpr int KILLSWITCH_TIMEOUT_MS = 5;

// NeuralPhaseVector parameters
constexpr int MAX_QUBITS     = 12;
constexpr int N_HUB_CHANNELS = 12;   // ICA neural hubs mapped to qubits
constexpr int N_CHANNELS     = 64;   // raw EEG channels
constexpr int N_SOS_SECTIONS = 4;    // IIR filter order / 2
'

write_stub "include/khaos_bridge.h" \
'#pragma once
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

_Static_assert(sizeof(NeuralPhaseVector) == 64,
    "NeuralPhaseVector must be exactly 64 bytes.");

#ifdef __cplusplus
}
#endif
'

write_stub "include/sha256.h" \
'#pragma once
/**
 * sha256.h — minimal SHA-256 interface
 *
 * In production, link against libcrypto (OpenSSL) or liboqs hash primitives.
 * This header provides the declaration expected by sovereignty_monitor.cpp.
 *
 * Replace the stub implementation in src/security/sha256.cpp with a real one.
 */
#include <array>
#include <cstdint>
#include <cstddef>

std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len);
'

echo ""

# =============================================================================
# 5. Coefficient generation script
# =============================================================================
info "Writing coefficient generator script..."

write_stub "scripts/gen_coefficients.py" \
'#!/usr/bin/env python3
"""
gen_coefficients.py — khaos-core
Generates IIR SOS filter coefficients and writes them as C headers.

Usage:
    python3 scripts/gen_coefficients.py

Requires: scipy, numpy
    pip install scipy numpy
"""

import numpy as np
from scipy.signal import iirfilter, zpk2sos
import json, pathlib

FS       = 1000.0   # sample rate (Hz)
BANDPASS = (8, 30)  # mu + beta combined pass band
ORDER    = 8        # Butterworth order (= 4 SOS sections)

z, p, k = iirfilter(ORDER, [b/FS*2 for b in BANDPASS],
                    btype="bandpass", ftype="butter", output="zpk")
sos = zpk2sos(z, p, k)  # shape: (ORDER/2, 6)

print(f"Generated {len(sos)}-section SOS filter for {BANDPASS[0]}–{BANDPASS[1]} Hz @ {FS} Hz")

# Write as JSON for inspection
out = pathlib.Path("coefficients/sos_8_30hz.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"sos": sos.tolist(), "fs": FS, "band": BANDPASS}, indent=2))
print(f"Saved: {out}")

# Write as C header
header = "// Auto-generated by scripts/gen_coefficients.py — do not edit manually\n"
header += f"// Butterworth {ORDER}-order bandpass {BANDPASS[0]}-{BANDPASS[1]} Hz @ {int(FS)} Hz\\n"
header += f"static const SOSCoeffs COMPUTED_SOS = {{\\n"
for field, col in [("b0",0),("b1",1),("b2",2),("a1",4),("a2",5)]:
    vals = ", ".join(f"{v:.10f}f" for v in sos[:, col])
    header += f"    /* {field} */ {{ {vals} }},\\n"
header += "};\\n"

out_h = pathlib.Path("coefficients/sos_8_30hz.h")
out_h.write_text(header)
print(f"Saved: {out_h}")
print("\\nInclude in signal_processor.cu:")
print(\'  #include "../../coefficients/sos_8_30hz.h"\')
print(\'  proc.init(&COMPUTED_SOS);\')
'

echo ""

# =============================================================================
# 6. Move existing files to canonical locations
# =============================================================================
info "Organising existing source files..."

move_if_exists() {
    local src="$1"; local dst="$2"
    if [[ -f "$src" && ! -f "$dst" ]]; then
        mv "$src" "$dst"
        success "  Moved $src → $dst"
    elif [[ -f "$src" && -f "$dst" ]]; then
        warn "  $dst already exists, leaving $src in place"
    elif [[ -f "$dst" ]]; then
        success "  $dst (already in place)"
    fi
}

# Move files generated in the previous session (if they're in the root)
move_if_exists "ETHICS.md"                        "docs/ETHICS.md"
move_if_exists "src/sovereignty_monitor.cpp"      "src/security/sovereignty_monitor.cpp"
move_if_exists "src/neuro/signal_processor.cu"    "src/neuro/signal_processor.cu"  # already correct

echo ""

# =============================================================================
# 7. Build instructions reminder
# =============================================================================
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} khaos-core scaffold complete${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Generate SOS coefficients:"
echo "    python3 scripts/gen_coefficients.py"
echo ""
echo "  Configure & build:"
echo "    cmake -B build -DCMAKE_BUILD_TYPE=Release -DETHICS_COMPLIANT=ON"
echo "    cmake --build build --parallel"
echo ""
echo "  Run benchmarks:"
echo "    ./build/tests/bench/bench_latency"
echo ""
echo -e "${YELLOW}Note:${RESET} ETHICS_COMPLIANT=ON is required. The build will"
echo "  fail with a fatal error if it is set to OFF."
echo ""
