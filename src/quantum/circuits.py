"""
circuits.py — KHAOS Quantum Backend
==========================================

Implements the parameterized quantum circuit that maps neural intention
(theta angles from the DWT pipeline) to quantum states, and the Latent
Space Navigator that measures proximity to calibrated landmark states.

Pipeline position:
    dwt.cu → NeuralPhaseVector.theta → THIS FILE → fidelity scores
                                                  → haptic feedback params

Architecture:
    khaos_circuit       — the main parameterized BCI circuit (Ry + CRY layers)
    khaos_swap_test     — SWAP test circuit for hardware-compatible fidelity
    KhaosBackend        — Python interface: wraps CUDA-Q, manages landmarks,
                          exposes the LatentSpaceNavigator API
    LandmarkLibrary     — calibrated neural states (rest, focus, motor-L/R, ...)

Qubit layout (N_QUBITS_TOTAL = 16):
    [0  .. 11]  Main register   — 12 neural hub channels → Ry(theta[i])
    [12 .. 15]  Auxiliary register — 4 qubits reserved for state control
                                    (ancilla for SWAP test, future extensions)

Extensibility:
    The main circuit takes n_main and n_aux as runtime parameters.
    Adding auxiliary qubits requires only updating those two values —
    the circuit kernel adapts its register sizes dynamically.

CUDA-Q backend selection:
    Simulation (Phase 0.5):  cudaq.set_target("nvidia")   — GPU statevector
    Hardware (future):       cudaq.set_target("quantinuum") or similar QPU

Fidelity computation:
    Simulation mode:  statevector inner product (exact, O(2^n))
    Hardware mode:    SWAP test circuit (sampling-based, O(shots))
    The backend selects automatically based on cudaq.get_target().

Ethics note:
    Landmark states are calibrated from the user's own neural data.
    This module never defines or imposes target states — it only
    measures distance to user-calibrated attractors.
    See docs/ETHICS.md §I.3 (Psychological Continuity).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cudaq
import numpy as np

# =============================================================================
# Configuration constants — must match khaos_bridge.h
# =============================================================================

N_QUBITS_MAIN  : int = 12      # neural hub channels
N_QUBITS_AUX   : int = 4       # auxiliary / ancilla qubits
N_QUBITS_TOTAL : int = N_QUBITS_MAIN + N_QUBITS_AUX
N_LAYERS       : int = 20      # circuit depth (20–25 as designed)
SHOTS_DEFAULT  : int = 1024    # for sampling-based fidelity (hardware mode)

# Minimum fidelity for RECOVERING → NOMINAL transition
FIDELITY_NOMINAL_THRESHOLD: float = 0.92

# =============================================================================
# Circuit parameters container
# =============================================================================

@dataclass
class CircuitParams:
    """
    All parameters needed to instantiate one khaos circuit evaluation.
    Passed by value to avoid shared-state bugs in the hot loop.
    """
    theta       : np.ndarray          # shape (N_LAYERS * N_QUBITS_MAIN,)  [0, 2π]
    ent_alpha   : np.ndarray          # shape (N_LAYERS,)                   [0, 1]
    edges_flat  : np.ndarray          # shape (2 * n_edges,) int32 — hub connectivity
    n_main      : int = N_QUBITS_MAIN
    n_aux       : int = N_QUBITS_AUX
    n_layers    : int = N_LAYERS

    @property
    def n_edges(self) -> int:
        return len(self.edges_flat) // 2

    @staticmethod
    def default_connectivity() -> np.ndarray:
        """
        Default sparse hub graph: nearest-neighbour ring + cross-hemisphere pairs.
        For 12 qubits (0..5 left hemisphere, 6..11 right hemisphere):
          - Intra-hemisphere: 0-1, 1-2, 2-3, 3-4, 4-5 (left)
                              6-7, 7-8, 8-9, 9-10, 10-11 (right)
          - Inter-hemisphere: 2-9 (motor cortex L-R), 3-8 (premotor L-R)
        Overwrite with the ICA covariance-derived graph after calibration.
        """
        edges = [
            # Left intra
            0, 1,  1, 2,  2, 3,  3, 4,  4, 5,
            # Right intra
            6, 7,  7, 8,  8, 9,  9, 10,  10, 11,
            # Inter-hemisphere (functional connectivity priors)
            2, 9,   3, 8,   0, 11,  5, 6,
        ]
        return np.array(edges, dtype=np.int32)

    @staticmethod
    def uniform(n_main: int = N_QUBITS_MAIN,
                n_layers: int = N_LAYERS) -> "CircuitParams":
        """Creates a circuit in the maximally neutral state (all theta = π/2)."""
        return CircuitParams(
            theta      = np.full(n_layers * n_main, math.pi / 2, dtype=np.float32),
            ent_alpha  = np.ones(n_layers, dtype=np.float32),
            edges_flat = CircuitParams.default_connectivity(),
            n_main     = n_main,
            n_layers   = n_layers,
        )


# =============================================================================
# CUDA-Q kernels
# =============================================================================

@cudaq.kernel
def khaos_circuit(
    theta     : list[float],   # [n_layers * n_main] Ry rotation angles
    ent_alpha : list[float],   # [n_layers]          CRY entanglement strength
    edges     : list[int],     # [2 * n_edges]       hub connectivity (src, dst pairs)
    n_main    : int,
    n_aux     : int,
    n_layers  : int,
    n_edges   : int,
) -> None:
    """
    Main khaos BCI circuit.

    Structure per layer:
      1. Ry(theta[layer * n_main + q])  for q in [0, n_main)
         — encodes neural intention into single-qubit rotations

      2. CRY(ent_alpha[layer] * π, edge[k].src, edge[k].dst)  for k in edges
         — encodes inter-hub functional connectivity
         — ent_alpha = 0 → identity (no entanglement)
         — ent_alpha = 1 → full CRY(π) ≈ CNOT (maximal entanglement)

    Auxiliary qubits [n_main .. n_main+n_aux) are initialised to |0⟩ and
    left idle in the main circuit. They are used by the SWAP test and
    future state control extensions without modifying this kernel.
    """
    qubits = cudaq.qvector(n_main + n_aux)

    for layer in range(n_layers):
        base = layer * n_main

        # ── Single-qubit rotations (neural intention encoding) ────────────────
        for q in range(n_main):
            ry(theta[base + q], qubits[q])

        # ── Sparse entanglement (functional connectivity) ──────────────────────
        alpha = ent_alpha[layer]
        if alpha > 0.0:
            for e in range(n_edges):
                src = edges[2 * e]
                dst = edges[2 * e + 1]
                # CRY(angle, control, target)
                # Interpolates between identity (alpha=0) and CNOT-equivalent (alpha=1)
                cry(alpha * math.pi, qubits[src], qubits[dst])


@cudaq.kernel
def khaos_swap_test(
    theta_a   : list[float],   # current state parameters
    theta_b   : list[float],   # landmark state parameters
    ent_a     : list[float],   # ent_alpha for state A
    ent_b     : list[float],   # ent_alpha for state B
    edges     : list[int],
    n_main    : int,
    n_layers  : int,
    n_edges   : int,
) -> None:
    """
    SWAP test circuit for hardware-compatible fidelity estimation.

    Allocates 2 * n_main + 1 qubits:
      register_a  [0       .. n_main-1]  — current state
      register_b  [n_main  .. 2*n_main-1]— landmark state
      ancilla     [2*n_main]             — SWAP test ancilla

    Measurement of ancilla gives:
      P(ancilla = 0) = (1 + F) / 2
      P(ancilla = 1) = (1 - F) / 2
      where F = |⟨ψ_a|ψ_b⟩|²

    Equivalently:  ⟨Z_ancilla⟩ = 2F - 1  →  F = (1 + ⟨Z⟩) / 2

    Use this kernel for QPU execution where statevector access is unavailable.
    For simulation, KhaosBackend.fidelity_statevector() is faster and exact.
    """
    n_total = 2 * n_main + 1
    qubits  = cudaq.qvector(n_total)

    # Indices
    # register_a: qubits[0 .. n_main-1]
    # register_b: qubits[n_main .. 2*n_main-1]
    # ancilla:    qubits[2*n_main]

    # ── Prepare register_a (current state) ───────────────────────────────────
    for layer in range(n_layers):
        base_a = layer * n_main
        for q in range(n_main):
            ry(theta_a[base_a + q], qubits[q])
        alpha_a = ent_a[layer]
        if alpha_a > 0.0:
            for e in range(n_edges):
                src = edges[2 * e]
                dst = edges[2 * e + 1]
                cry(alpha_a * math.pi, qubits[src], qubits[dst])

    # ── Prepare register_b (landmark state) ──────────────────────────────────
    for layer in range(n_layers):
        base_b = layer * n_main
        for q in range(n_main):
            ry(theta_b[base_b + q], qubits[n_main + q])
        alpha_b = ent_b[layer]
        if alpha_b > 0.0:
            for e in range(n_edges):
                src = edges[2 * e]
                dst = edges[2 * e + 1]
                cry(alpha_b * math.pi,
                    qubits[n_main + src],
                    qubits[n_main + dst])

    # ── SWAP test ─────────────────────────────────────────────────────────────
    anc = qubits[2 * n_main]
    h(anc)
    for q in range(n_main):
        swap.ctrl(anc, qubits[q], qubits[n_main + q])
    h(anc)
    # ancilla is now ready for measurement / Z-expectation


# =============================================================================
# Landmark library
# =============================================================================

@dataclass
class Landmark:
    """One calibrated neural attractor state."""
    name        : str
    params      : CircuitParams
    calibrated_at: float = field(default_factory=time.time)
    session_count: int   = 0         # number of calibration sessions averaged

    def proximity_label(self, score: float) -> str:
        """Human-readable proximity description for haptic UI."""
        if score > 0.85:   return f"{self.name}: very close"
        elif score > 0.65: return f"{self.name}: approaching"
        elif score > 0.40: return f"{self.name}: distant"
        else:              return f"{self.name}: far"


class LandmarkLibrary:
    """
    Stores and retrieves user-calibrated landmark states.

    Landmarks are never externally defined — they are always derived
    from the user's own neural data during the calibration protocol.
    See docs/ETHICS.md §I.3 (Psychological Continuity).
    """

    def __init__(self) -> None:
        self._landmarks: dict[str, Landmark] = {}

    def add(self, landmark: Landmark) -> None:
        self._landmarks[landmark.name] = landmark

    def get(self, name: str) -> Optional[Landmark]:
        return self._landmarks.get(name)

    def all(self) -> list[Landmark]:
        return list(self._landmarks.values())

    def names(self) -> list[str]:
        return list(self._landmarks.keys())

    def __len__(self) -> int:
        return len(self._landmarks)

    def update_from_session(self, name: str, new_params: CircuitParams) -> None:
        """
        Online landmark update: running average of theta across calibration sessions.
        Implements the multi-session bootstrap described in the calibration protocol.
        """
        if name not in self._landmarks:
            self._landmarks[name] = Landmark(name=name, params=new_params)
            return

        lm = self._landmarks[name]
        n  = lm.session_count + 1
        # Running mean: θ_avg = ((n-1)*θ_old + θ_new) / n
        lm.params.theta = ((n - 1) * lm.params.theta + new_params.theta) / n
        # Entanglement: same averaging
        lm.params.ent_alpha = ((n - 1) * lm.params.ent_alpha
                               + new_params.ent_alpha) / n
        lm.session_count  = n
        lm.calibrated_at  = time.time()


# =============================================================================
# KhaosBackend — main interface
# =============================================================================

class KhaosBackend:
    """
    High-level interface to the CUDA-Q quantum simulation layer.

    Responsibilities:
      - Select the appropriate CUDA-Q target (simulation vs QPU)
      - Run the main circuit and compute observables
      - Compute fidelity against landmark states
      - Drive the LatentSpaceNavigator
      - Expose quantum homeostasis metrics (S(ρ), ⟨Z_i⟩)

    Thread safety:
      KhaosBackend is NOT thread-safe. Call from the CUDA-Q dispatch thread only.
      The CUDA pipeline (signal_processor.cu, dwt.cu) runs on CUDA streams;
      this class runs on the Python host thread that consumes the θ-Frame.
    """

    def __init__(self,
                 landmarks   : Optional[LandmarkLibrary] = None,
                 target      : str = "nvidia",
                 shots       : int = SHOTS_DEFAULT) -> None:
        """
        @param landmarks  Pre-loaded landmark library (or None to start empty)
        @param target     CUDA-Q target: "nvidia" (GPU simulation) or QPU name
        @param shots      Sampling count for hardware-mode fidelity
        """
        cudaq.set_target(target)
        self.target     = target
        self.shots      = shots
        self.landmarks  = landmarks or LandmarkLibrary()
        self._sim_mode  = (target in ("nvidia", "qpp-cpu", "density-matrix-cpu"))

        print(f"[khaos-q] CUDA-Q target: {target} "
              f"({'simulation' if self._sim_mode else 'hardware'} mode)")

    # -------------------------------------------------------------------------
    # Core circuit evaluation
    # -------------------------------------------------------------------------

    def get_state(self, params: CircuitParams) -> np.ndarray:
        """
        Return the full statevector |ψ⟩ for the given circuit parameters.
        Shape: (2^n_main,) complex128.
        Only available in simulation mode.
        """
        if not self._sim_mode:
            raise RuntimeError("get_state() is only available in simulation mode. "
                               "Use sample_state() on hardware targets.")
        state = cudaq.get_state(
            khaos_circuit,
            params.theta.tolist(),
            params.ent_alpha.tolist(),
            params.edges_flat.tolist(),
            params.n_main,
            params.n_aux,
            params.n_layers,
            params.n_edges,
        )
        return np.array(state, dtype=np.complex128)

    def expectation_z(self, params: CircuitParams) -> np.ndarray:
        """
        Compute ⟨Z_i⟩ for each main qubit i.
        Returns shape (n_main,), values ∈ [-1, 1].

        Interpretation:
          +1  → qubit in |0⟩ (idle / rest)
          -1  → qubit in |1⟩ (active / motor intent)
           0  → equal superposition
        """
        z_ops = [cudaq.spin.z(i) for i in range(params.n_main)]
        hamiltonian = sum(z_ops)
        result = cudaq.observe(
            khaos_circuit,
            hamiltonian,
            params.theta.tolist(),
            params.ent_alpha.tolist(),
            params.edges_flat.tolist(),
            params.n_main,
            params.n_aux,
            params.n_layers,
            params.n_edges,
        )
        # Extract per-qubit expectations from the individual terms
        return np.array([
            cudaq.observe(
                khaos_circuit,
                cudaq.spin.z(i),
                params.theta.tolist(),
                params.ent_alpha.tolist(),
                params.edges_flat.tolist(),
                params.n_main,
                params.n_aux,
                params.n_layers,
                params.n_edges,
            ).expectation()
            for i in range(params.n_main)
        ], dtype=np.float32)

    def entanglement_entropy(self, params: CircuitParams,
                             partition: Optional[list[int]] = None) -> float:
        """
        Compute the Von Neumann entanglement entropy S(ρ_A) for a subsystem.

        @param partition  Qubit indices forming subsystem A.
                          Default: left hemisphere [0..5] (first 6 main qubits)
        @returns          S(ρ_A) ∈ [0, log2(min(|A|, |Ā|))]

        High entropy → hemispheres strongly correlated → "expansion" sensation
        Low entropy  → independent processing          → "focus" sensation
        """
        if partition is None:
            partition = list(range(params.n_main // 2))  # left hemisphere

        sv = self.get_state(params)            # full statevector
        n  = params.n_main
        dim_total = 1 << n
        dim_a     = 1 << len(partition)
        dim_b     = dim_total // dim_a

        # Build the reduced density matrix ρ_A by tracing out subsystem B
        # Reshape statevector into (dim_a, dim_b) matrix
        # (This works cleanly when partition = [0..k-1]; general case below)
        complement = [q for q in range(n) if q not in partition]

        # Rearrange qubit ordering: partition first, complement second
        reorder = partition + complement
        sv_matrix = sv.reshape([2] * n)
        sv_reordered = np.transpose(sv_matrix, reorder).reshape(dim_a, dim_b)

        # ρ_A = Tr_B(|ψ⟩⟨ψ|) = sv_reordered @ sv_reordered†
        rho_a  = sv_reordered @ sv_reordered.conj().T
        eigvals = np.linalg.eigvalsh(rho_a).real
        eigvals = eigvals[eigvals > 1e-12]          # drop numerical noise

        entropy = -np.sum(eigvals * np.log2(eigvals))
        return float(np.clip(entropy, 0.0, float(len(partition))))

    # -------------------------------------------------------------------------
    # Fidelity computation
    # -------------------------------------------------------------------------

    def fidelity_statevector(self, params_a: CircuitParams,
                              params_b: CircuitParams) -> float:
        """
        Exact fidelity via statevector inner product.
        F = |⟨ψ_a|ψ_b⟩|²  ∈ [0, 1]

        Simulation mode only. O(2^n_main) time and memory.
        For n_main=12: 4096 complex128 values — 64 KB, negligible.
        """
        sv_a = self.get_state(params_a)
        sv_b = self.get_state(params_b)
        overlap = np.dot(np.conj(sv_a), sv_b)
        return float(np.abs(overlap) ** 2)

    def fidelity_swap_test(self, params_a: CircuitParams,
                            params_b: CircuitParams) -> float:
        """
        Sampling-based fidelity via SWAP test circuit.
        Hardware-compatible. Precision ∝ 1/√shots.

        F = (1 + ⟨Z_ancilla⟩) / 2
        where ⟨Z_ancilla⟩ is measured from the SWAP test circuit.
        """
        ancilla_idx = 2 * params_a.n_main   # last qubit in the SWAP test circuit
        z_ancilla   = cudaq.spin.z(ancilla_idx)

        result = cudaq.observe(
            khaos_swap_test,
            z_ancilla,
            params_a.theta.tolist(),
            params_b.theta.tolist(),
            params_a.ent_alpha.tolist(),
            params_b.ent_alpha.tolist(),
            params_a.edges_flat.tolist(),
            params_a.n_main,
            params_a.n_layers,
            params_a.n_edges,
        )
        z_exp = result.expectation()
        return float(np.clip((1.0 + z_exp) / 2.0, 0.0, 1.0))

    def fidelity(self, params_a: CircuitParams,
                 params_b: CircuitParams) -> float:
        """
        Auto-selects the fidelity method based on the active target.
        Simulation → statevector (exact, ~40µs for n=12)
        Hardware   → SWAP test (sampling, ~shots * gate_time)
        """
        if self._sim_mode:
            return self.fidelity_statevector(params_a, params_b)
        else:
            return self.fidelity_swap_test(params_a, params_b)

    # -------------------------------------------------------------------------
    # Latent Space Navigator
    # -------------------------------------------------------------------------

    def navigate(self, current: CircuitParams) -> dict[str, float]:
        """
        Compute proximity to all registered landmarks.

        Returns a dict {landmark_name: proximity_score ∈ [0, 1]}
        where 1.0 = identical state, 0.0 = orthogonal state.

        This is the core of the haptic feedback loop:
          - The proximity scores are mapped to haptic parameters in inject_sensory_feedback
          - The user feels which landmark they are closest to / moving toward
          - Over weeks, the brain learns to navigate this space deliberately

        Complexity: O(n_landmarks * 2^n_main) for simulation mode.
        For n_main=12, n_landmarks=4: ~4 * 40µs = 160µs — fits in the 9.8ms budget.
        """
        return {
            lm.name: self.fidelity(current, lm.params)
            for lm in self.landmarks.all()
        }

    def nearest_landmark(self, current: CircuitParams
                         ) -> tuple[Optional[str], float]:
        """
        Returns the name and fidelity of the closest landmark.
        Returns (None, 0.0) if no landmarks are registered.
        """
        scores = self.navigate(current)
        if not scores:
            return None, 0.0
        name   = max(scores, key=scores.__getitem__)
        return name, scores[name]

    def is_recovering_converged(self, current: CircuitParams,
                                held_landmark_name: str) -> bool:
        """
        Used by the circuit breaker RECOVERING → NOMINAL transition.
        Returns True when fidelity against the held state exceeds the threshold.
        """
        held = self.landmarks.get(held_landmark_name)
        if held is None:
            return False
        return self.fidelity(current, held.params) >= FIDELITY_NOMINAL_THRESHOLD

    # -------------------------------------------------------------------------
    # Quantum Intent Divergence (QID) detection
    # -------------------------------------------------------------------------

    def detect_qid(self,
                   current     : CircuitParams,
                   held_name   : str,
                   snr         : float,
                   theta_stability: float,
                   stable_streak  : int,
                   fidelity_velocity: float,
                   # Thresholds — tune per user during calibration
                   fidelity_max   : float = 0.40,
                   snr_min        : float = 0.75,
                   stability_max  : float = 0.08,
                   streak_min     : int   = 3,
                   ) -> bool:
        """
        Returns True if the user has genuinely changed intention during PANIC
        (rather than simply having a noisy signal).

        Condition: low fidelity to the held state + high SNR + stable new attractor.
        See circuits.py design notes for full derivation.

        @param fidelity_velocity  d(fidelity)/dt — negative means diverging from hold
        """
        held = self.landmarks.get(held_name)
        if held is None:
            return False

        fidelity = self.fidelity(current, held.params)

        return (
            fidelity         <  fidelity_max   and
            snr              >  snr_min         and
            theta_stability  <  stability_max   and
            stable_streak    >= streak_min      and
            fidelity_velocity < 0.0
        )

    # -------------------------------------------------------------------------
    # Calibration helpers
    # -------------------------------------------------------------------------

    def calibrate_landmark(self, name: str,
                            theta_frames: list[np.ndarray],
                            ent_alpha_frames: Optional[list[np.ndarray]] = None,
                            edges: Optional[np.ndarray] = None) -> Landmark:
        """
        Compute a landmark from a batch of theta frames collected during a
        calibration session. Averages theta across frames and wraps into
        a CircuitParams / Landmark.

        @param theta_frames       List of (N_LAYERS * N_QUBITS_MAIN,) arrays,
                                  one per calibration frame (typ. 300–1000 frames)
        @param ent_alpha_frames   Optional per-frame ent_alpha; defaults to ones
        @param edges              Hub connectivity graph; defaults to default_connectivity
        """
        theta_mean = np.mean(np.stack(theta_frames), axis=0).astype(np.float32)

        if ent_alpha_frames is not None:
            ent_mean = np.mean(np.stack(ent_alpha_frames), axis=0).astype(np.float32)
        else:
            ent_mean = np.ones(N_LAYERS, dtype=np.float32)

        params = CircuitParams(
            theta      = theta_mean,
            ent_alpha  = ent_mean,
            edges_flat = edges if edges is not None
                         else CircuitParams.default_connectivity(),
        )
        lm = Landmark(name=name, params=params)
        self.landmarks.update_from_session(name, params)
        return lm

    def calibration_quality(self, theta_frames: list[np.ndarray],
                             rest_params: Optional[CircuitParams] = None) -> float:
        """
        Compute a quality score [0, 1] for a calibration session.
        Combines entropy (uniqueness), intra-session stability, and
        separation from the rest landmark.

        score > 0.70: acceptable calibration
        score > 0.85: excellent calibration
        """
        stack = np.stack(theta_frames)   # (n_frames, n_params)

        # Intra-session stability: 1 - mean normalized std deviation
        mean_std = np.mean(np.std(stack, axis=0)) / math.pi
        stability = float(np.clip(1.0 - mean_std, 0.0, 1.0))

        # Entropy of the mean state
        mean_params = CircuitParams(
            theta      = np.mean(stack, axis=0).astype(np.float32),
            ent_alpha  = np.ones(N_LAYERS, dtype=np.float32),
            edges_flat = CircuitParams.default_connectivity(),
        )
        entropy = self.entanglement_entropy(mean_params)
        max_ent = math.log2(N_QUBITS_MAIN // 2)
        entropy_score = float(np.clip(entropy / max_ent, 0.0, 1.0))

        # Separation from rest (0 if no rest landmark registered)
        separation = 0.5  # neutral default
        rest_lm    = self.landmarks.get("rest")
        if rest_lm is not None:
            f = self.fidelity(mean_params, rest_lm.params)
            separation = float(np.clip(1.0 - f, 0.0, 1.0))

        return 0.40 * stability + 0.35 * entropy_score + 0.25 * separation


# =============================================================================
# Convenience: build a default backend for Phase 0.5 (simulation)
# =============================================================================

def build_simulation_backend(gpu_index: int = 0) -> KhaosBackend:
    """
    Initialises a GPU statevector simulation backend on the specified device.
    Appropriate for Phase 0.5: no real hardware required.
    """
    cudaq.set_target("nvidia", option=f"device={gpu_index}")
    print(f"[khaos-q] GPU statevector simulation on device {gpu_index}")
    return KhaosBackend(target="nvidia")


# =============================================================================
# Quick smoke test — run with: python3 src/quantum/circuits.py
# =============================================================================

if __name__ == "__main__":
    print("KHAOS quantum circuit smoke test")
    print("=" * 50)

    backend = build_simulation_backend()

    # Register a "rest" landmark at the neutral state
    rest_params = CircuitParams.uniform()
    backend.landmarks.add(Landmark(name="rest", params=rest_params))

    # Create a "motor_left" state: qubits 0-5 rotated toward |1⟩
    motor_l = CircuitParams.uniform()
    motor_l.theta[:N_QUBITS_MAIN // 2] = math.pi * 0.85   # left hemisphere active
    motor_l.theta[N_QUBITS_MAIN // 2:] = math.pi * 0.15   # right hemisphere quiet
    backend.landmarks.add(Landmark(name="motor_left", params=motor_l))

    # Test 1: fidelity of rest vs itself (should be ≈ 1.0)
    f_self = backend.fidelity(rest_params, rest_params)
    print(f"Fidelity(rest, rest)        = {f_self:.6f}  (expected ≈ 1.0)")
    assert f_self > 0.999, f"Self-fidelity failed: {f_self}"

    # Test 2: fidelity of rest vs motor_left (should be < 0.5)
    f_diff = backend.fidelity(rest_params, motor_l)
    print(f"Fidelity(rest, motor_left)  = {f_diff:.6f}  (expected < 0.5)")
    assert f_diff < 0.5, f"Cross-fidelity too high: {f_diff}"

    # Test 3: entanglement entropy
    S = backend.entanglement_entropy(rest_params)
    print(f"Entanglement entropy (rest) = {S:.4f} bits")

    # Test 4: Z expectations
    z_exp = backend.expectation_z(rest_params)
    print(f"⟨Z⟩ per qubit (rest, first 4): {z_exp[:4]}")

    # Test 5: navigator
    test_state = CircuitParams.uniform()
    test_state.theta[:4] = math.pi * 0.9  # slightly toward motor_left
    scores = backend.navigate(test_state)
    print(f"Navigator scores: { {k: f'{v:.3f}' for k,v in scores.items()} }")

    print("\nAll smoke tests passed.")
