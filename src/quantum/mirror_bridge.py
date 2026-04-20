#!/usr/bin/env python3
"""
mirror_bridge.py — Quantum Mirror Python bridge
stdin/stdout JSON-line protocol between khaos_mirror (C++) and the
Python quantum + graphene stack.

Protocol:
  ← {"theta":[...12 floats...], "bp_index": float, "timestamp_ns": int}
  → {"fidelity": float, "landmark": str, "qid": bool,
     "ent_alpha": [...12 floats...], "spec": {...}}

Runs at ~10 Hz (driven by C++ sending every 100 EEG frames).
If cudaq is not installed, falls back to a numpy statevector simulator.
"""

import sys
import json
import math
import os
import time
import traceback

import numpy as np

# ── Lightweight optional imports (no JIT at module level) ────────────────────
# cudaq and circuits are imported lazily inside main() AFTER printing "started"
# so the C++ side knows the bridge is alive before CUDA-Q JIT compilation runs.

CUDAQ_AVAILABLE    = False
CIRCUITS_AVAILABLE = False
DIRAC_AVAILABLE    = False

_KhaosBackend  = None
_CircuitParams = None
_DiracEmulator = None
_GrapheneParams = None

# ── Constants ─────────────────────────────────────────────────────────────────

N_QUBITS  = 12
N_LAYERS  = 4          # reduced for CPU simulation
EDGES = [0,1, 2,3, 4,5, 6,7, 8,9, 10,11,   # intra-hemisphere
         2,9, 3,8, 0,11, 5,6]               # inter-hemisphere

# ── Numpy statevector fallback ────────────────────────────────────────────────

def _ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)

def _apply_single(sv: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    """Apply 2x2 unitary U to qubit q of an n-qubit statevector."""
    sv = sv.reshape([2] * n)
    sv = np.tensordot(U, sv, axes=[[1], [q]])
    # tensordot puts the new axis first; move it back to position q
    order = list(range(1, q + 1)) + [0] + list(range(q + 1, n))
    sv = np.transpose(sv, order)
    return sv.reshape(-1)

def numpy_circuit(theta: list, ent_alpha: list) -> np.ndarray:
    """Minimal statevector simulation of khaos_circuit."""
    n = N_QUBITS
    sv = np.zeros(2**n, dtype=np.complex64)
    sv[0] = 1.0

    for layer in range(N_LAYERS):
        for q in range(n):
            angle = theta[layer * n + q] if layer * n + q < len(theta) else theta[q % len(theta)]
            sv = _apply_single(sv, n, q, _ry_matrix(angle))
        # CRY approximation: apply small Ry rotation on target conditioned on edge
        ea = ent_alpha[layer] if layer < len(ent_alpha) else 0.0
        for i in range(0, len(EDGES), 2):
            ctrl, tgt = EDGES[i], EDGES[i + 1]
            sv = _apply_single(sv, n, tgt, _ry_matrix(ea * math.pi * 0.5))

    return sv


def statevector_fidelity(sv_a: np.ndarray, sv_b: np.ndarray) -> float:
    """|⟨ψ_a|ψ_b⟩|²"""
    return float(abs(np.dot(np.conj(sv_a), sv_b)) ** 2)


# ── Entanglement entropy (von Neumann, first qubit partition) ─────────────────

def entanglement_entropy(sv: np.ndarray) -> float:
    n = N_QUBITS
    rho = sv.reshape(2, 2**(n-1))
    rho = rho @ rho.conj().T
    eigs = np.linalg.eigvalsh(rho).clip(1e-12, None)
    return float(-np.sum(eigs * np.log2(eigs)))


# ── Landmark library ──────────────────────────────────────────────────────────

# Default vault path — relative to this file's directory
_VAULT_PATH = os.path.join(os.path.dirname(__file__),
                           '../../vault/landmarks.json')

# Maps calibration_wizard phase IDs → LandmarkLibrary keys
_PHASE_TO_LANDMARK = {
    "zen_rest":      "rest",
    "motor_alpha":   "calm",
    "focus_beta":    "focus",
    "motor_imagery": "motor",
    "flow":          "flow",
}


def _load_vault(vault_path: str, n_qubits: int, n_layers: int) -> dict:
    """
    Load calibrated landmarks from a vault JSON written by calibration_wizard.py.

    The vault stores theta_mean as N_QUBITS (12) values per phase.
    LandmarkLibrary needs N_QUBITS × N_LAYERS (48) values per landmark.

    Expansion strategy: tile theta_mean across layers.  Layer l gets a
    small amplitude decay (0.95^l) to give the parameterised circuit some
    variation across layers while preserving the dominant frequency signature.
    """
    try:
        with open(vault_path) as f:
            vault = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[bridge] vault load failed ({exc}) — using default priors",
              file=sys.stderr)
        return {}

    landmarks = {}
    for entry in vault.get("chain", []):
        phase_id = entry.get("phase_id", "")
        lm_name  = _PHASE_TO_LANDMARK.get(phase_id)
        if lm_name is None:
            continue
        theta_12 = entry["landmark"]["theta_mean"]   # 12 floats
        # Tile across N_LAYERS with gentle per-layer decay
        theta_full = []
        for layer in range(n_layers):
            scale = 0.95 ** layer
            theta_full.extend(float(t * scale) for t in theta_12)
        landmarks[lm_name] = theta_full[:n_qubits * n_layers]

    n_loaded = len(landmarks)
    if n_loaded:
        print(f"[bridge] vault loaded: {n_loaded} landmarks from {vault_path}",
              file=sys.stderr)
    return landmarks


class LandmarkLibrary:
    def __init__(self, vault_path: str | None = None):
        n = N_QUBITS * N_LAYERS
        # Seed defaults with interpretable priors (used as fallback)
        rng = np.random.default_rng(42)
        defaults = {
            "rest":  rng.uniform(0.1, 0.5, n).tolist(),
            "focus": rng.uniform(1.0, 2.0, n).tolist(),
            "flow":  rng.uniform(0.8, 1.6, n).tolist(),
            "calm":  rng.uniform(0.2, 0.8, n).tolist(),
        }
        # Merge vault over defaults (vault wins for matching keys)
        if vault_path is None:
            vault_path = _VAULT_PATH
        calibrated = _load_vault(vault_path, N_QUBITS, N_LAYERS)
        self.landmarks = {**defaults, **calibrated}
        self._svs: dict = {}

    def get_sv(self, name: str, ent_alpha: list) -> np.ndarray:
        if name not in self._svs:
            self._svs[name] = numpy_circuit(self.landmarks[name], ent_alpha)
        return self._svs[name]

    def nearest(self, sv: np.ndarray, ent_alpha: list) -> tuple[str, float]:
        best_name, best_f = "unknown", -1.0
        for name in self.landmarks:
            lm_sv = self.get_sv(name, ent_alpha)
            f = statevector_fidelity(sv, lm_sv)
            if f > best_f:
                best_f, best_name = f, name
        return best_name, best_f

    def update(self, name: str, theta: list, alpha: float = 0.05):
        """Running average update."""
        if name in self.landmarks:
            old = np.array(self.landmarks[name])
            new = np.array(theta[:len(old)])
            self.landmarks[name] = ((1 - alpha) * old + alpha * new).tolist()
            self._svs.pop(name, None)  # invalidate cached sv


# ── Graphene ent_alpha ────────────────────────────────────────────────────────

def compute_ent_alpha_graphene(theta: list) -> list:
    """Use DiracEmulator if available, else heuristic."""
    if DIRAC_AVAILABLE and _DiracEmulator is not None and _GrapheneParams is not None:
        try:
            emu = _DiracEmulator(_GrapheneParams())
            emu.update_from_theta(theta)
            out = emu.get_output()
            if out is not None:
                return list(out.ent_alpha[:N_LAYERS])
        except Exception:
            pass
    # Heuristic: map mean theta power to ent_alpha via tanh
    mean_power = float(np.mean(np.array(theta[:N_QUBITS]) ** 2))
    base = math.tanh(mean_power / (math.pi ** 2))
    return [base * (1.0 - 0.05 * i) for i in range(N_LAYERS)]


# ── Main bridge loop ──────────────────────────────────────────────────────────

def main():
    global CUDAQ_AVAILABLE, CIRCUITS_AVAILABLE, DIRAC_AVAILABLE
    global _KhaosBackend, _CircuitParams, _DiracEmulator, _GrapheneParams

    held_sv = None
    frame = 0

    # Ensure unbuffered I/O (belt-and-suspenders alongside python3 -u)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # ── Signal to C++ that the bridge is alive BEFORE any slow imports ────────
    print("[bridge] started (initialising...)", file=sys.stderr, flush=True)

    # ── Deferred heavy imports (CUDA-Q JIT may take 10-30s on first run) ──────
    try:
        import cudaq as _cudaq
        CUDAQ_AVAILABLE = True
    except ImportError:
        pass

    try:
        from circuits import KhaosBackend as _KB, CircuitParams as _CP
        _KhaosBackend  = _KB
        _CircuitParams = _CP
        CIRCUITS_AVAILABLE = True
    except Exception as e:
        print(f"[bridge] circuits import failed: {e}", file=sys.stderr, flush=True)

    try:
        sys.path.insert(0, "src/graphene")
        from dirac_emulator import DiracEmulator as _DE, GrapheneParams as _GP
        _DiracEmulator  = _DE
        _GrapheneParams = _GP
        DIRAC_AVAILABLE = True
    except Exception as e:
        print(f"[bridge] dirac import failed: {e}", file=sys.stderr, flush=True)

    print(f"[bridge] ready (cudaq={CUDAQ_AVAILABLE}, "
          f"circuits={CIRCUITS_AVAILABLE}, dirac={DIRAC_AVAILABLE})",
          file=sys.stderr, flush=True)

    # Auto-load calibrated vault if present (written by calibration_wizard.py).
    # Falls back to hardcoded priors when vault is missing.
    lib = LandmarkLibrary()

    # Signal C++ that the bridge is fully initialised and ready to receive frames.
    # C++ calls wait_ready() which blocks on this line before starting the EEG loop.
    print(json.dumps({"ready": True}), flush=True)
    print("[bridge] accepting frames", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[bridge] bad JSON: {e}", file=sys.stderr)
            continue

        theta      = msg.get("theta", [0.0] * N_QUBITS)
        bp_index   = float(msg.get("bp_index", 0.0))
        ts_ns      = int(msg.get("timestamp_ns", 0))

        # Pad theta to N_QUBITS * N_LAYERS
        full_theta = (theta * N_LAYERS)[:N_QUBITS * N_LAYERS]

        t0 = time.perf_counter()

        # Entanglement alpha from graphene model
        ent_alpha = compute_ent_alpha_graphene(theta)

        # Statevector
        sv = numpy_circuit(full_theta, ent_alpha)

        # Fidelity against held state (circuit-breaker RECOVERING criterion)
        fidelity = statevector_fidelity(sv, held_sv) if held_sv is not None else 1.0
        held_sv  = sv

        # Nearest landmark
        landmark, lm_fidelity = lib.nearest(sv, ent_alpha)
        lib.update(landmark, full_theta)

        # Quantum Intent Divergence: high fidelity drop + strong bp signal
        qid = (fidelity < 0.6 and bp_index > 0.6 and frame > 5)

        # Entropy
        entropy = entanglement_entropy(sv)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Convert all numpy scalars to native Python types for JSON
        response = {
            "fidelity":    round(float(fidelity), 4),
            "landmark":    landmark,
            "lm_fidelity": round(float(lm_fidelity), 4),
            "qid":         bool(qid),
            "ent_alpha":   [round(float(x), 4) for x in ent_alpha],
            "entropy":     round(float(entropy), 4),
            "elapsed_ms":  round(float(elapsed_ms), 2),
            "ts_ns":       int(ts_ns),
        }

        print(json.dumps(response), flush=True)
        frame += 1


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass   # C++ side closed the pipe — clean exit
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[bridge] fatal: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
