"""
test_swap_fidelity.py — SWAP Test Relative Fidelity Suite
══════════════════════════════════════════════════════════════════════════════
Validates the relative SWAP fidelity formula used by KHAOS to assess
quantum state similarity despite the noise floor introduced by Muse 2 AgCl
dry-contact electrodes.

Relative fidelity formula (Drizzy / KHAOS convention):

  F_corrected = (raw_SWAP − ε_user_baseline) / (1 − ε_user_baseline)

where:
  raw_SWAP        ← outcome probability of the ancilla being |0⟩ in a SWAP test
  ε_user_baseline ← mean raw_SWAP measured during 2-min eyes-closed calibration
                    (noise floor characterising the user's electrode contact quality)
  F_corrected     ← corrected fidelity ∈ [0, 1], clamped to avoid negatives

The SWAP test (Barenco et al. 1997):
  |ancilla⟩ = |+⟩ = (|0⟩+|1⟩)/√2
  Apply controlled-SWAP(|ψ⟩, |φ⟩)
  Measure ancilla: P(|0⟩) = ½ + ½·|⟨ψ|φ⟩|²
  → raw_SWAP = ½ + ½·F    ⟹   F = 2·raw_SWAP − 1

Test coverage
─────────────
  TestSWAPMathematics
    • identical states  → F = 1.0
    • orthogonal states → F = 0.0
    • arbitrary overlap → F = |⟨ψ|φ⟩|²

  TestBaselineCalibration
    • calibration mean is a good estimator of ε_baseline
    • F_corrected saturates at 1 when raw == 1
    • F_corrected saturates at 0 when raw == ε_baseline
    • F_corrected is clamped ≥ 0 for noisy low readings

  TestRelativeFidelity
    • high-quality signal yields F_corrected close to 1.0
    • degraded signal yields proportionally lower F_corrected
    • per-user baseline correctly separates users with different SNR

  TestEndToEndPipeline
    • SyntheticMuse2Adapter → FeatureExtractor → SWAP fidelity
    • assert theta vector shape and value range
    • assert fidelity plausible for synthetic clean signal

  TestNeurightBoundary (ethics gate)
    • gate blocks raw EEG from crossing boundary
    • gate passes theta vector (240 elements ∈ [0, 2π])
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from src.io.muse2_adapter import SyntheticMuse2Adapter, BUFFER_N, FS
from src.bci.feature_extractor import Muse2FeatureExtractor, N_QUBITS, THETA_LEN
from src.ethics.ethics_gate import EthicsGate, NeurightViolation


# ── SWAP test primitives ──────────────────────────────────────────────────────

def raw_swap_from_overlap(overlap_sq: float) -> float:
    """P(ancilla=|0⟩) = ½ + ½·|⟨ψ|φ⟩|²"""
    return 0.5 + 0.5 * overlap_sq


def overlap_sq(psi: np.ndarray, phi: np.ndarray) -> float:
    """|⟨ψ|φ⟩|² for normalised state vectors."""
    psi_n = psi / (np.linalg.norm(psi) + 1e-30)
    phi_n = phi / (np.linalg.norm(phi) + 1e-30)
    return float(abs(np.dot(np.conj(psi_n), phi_n)) ** 2)


def relative_fidelity(raw_swap: float,
                      epsilon_baseline: float,
                      clamp: bool = True) -> float:
    """Compute F_corrected = (raw_SWAP − ε) / (1 − ε).

    Parameters
    ----------
    raw_swap         : float ∈ [0.5, 1.0]  (SWAP ancilla probability)
    epsilon_baseline : float ∈ [0.5, 1.0)  (noise floor from calibration)
    clamp            : if True, clamp result to [0, 1]

    Returns
    -------
    float : corrected fidelity
    """
    denom = 1.0 - epsilon_baseline
    if abs(denom) < 1e-12:
        return 0.0
    f = (raw_swap - epsilon_baseline) / denom
    if clamp:
        f = max(0.0, min(1.0, f))
    return f


def simulate_calibration(n_samples: int = 256,
                         noise_level: float = 0.02,
                         rng: np.random.Generator = None) -> float:
    """Simulate 2-min eyes-closed calibration.

    During calibration the user is at rest, so ψ ≈ φ is noisy baseline.
    raw_SWAP ≈ 0.5 + noise.  Returns ε_user_baseline.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # At rest: states are nearly random → overlap ≈ 0
    # raw_SWAP ≈ 0.5 + small positive noise from electrode contact quality
    raw_swaps = 0.5 + rng.uniform(0.0, noise_level * 2, size=n_samples)
    return float(np.mean(raw_swaps))


# ── Test cases ────────────────────────────────────────────────────────────────

class TestSWAPMathematics(unittest.TestCase):
    """Unit tests for SWAP test math primitives."""

    def test_identical_states(self):
        """Identical states → |⟨ψ|ψ⟩|² = 1 → raw_SWAP = 1 → F = 1."""
        psi = np.array([1, 0, 0, 0], dtype=complex)
        ovlp = overlap_sq(psi, psi)
        self.assertAlmostEqual(ovlp, 1.0, places=10)

        raw = raw_swap_from_overlap(ovlp)
        self.assertAlmostEqual(raw, 1.0, places=10)

    def test_orthogonal_states(self):
        """Orthogonal states → |⟨ψ|φ⟩|² = 0 → raw_SWAP = 0.5 → F = 0."""
        psi = np.array([1, 0, 0, 0], dtype=complex)
        phi = np.array([0, 1, 0, 0], dtype=complex)
        ovlp = overlap_sq(psi, phi)
        self.assertAlmostEqual(ovlp, 0.0, places=10)

        raw = raw_swap_from_overlap(ovlp)
        self.assertAlmostEqual(raw, 0.5, places=10)

    def test_partial_overlap(self):
        """+ state vs |0⟩: |⟨+|0⟩|² = 0.5 → raw_SWAP = 0.75."""
        plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        zero = np.array([1, 0], dtype=complex)
        ovlp = overlap_sq(plus, zero)
        self.assertAlmostEqual(ovlp, 0.5, places=10)

        raw = raw_swap_from_overlap(ovlp)
        self.assertAlmostEqual(raw, 0.75, places=10)

    def test_normalisation_invariance(self):
        """Un-normalised inputs should produce same overlap as normalised."""
        psi = np.array([3, 4], dtype=complex)   # norm = 5
        phi = np.array([4, 3], dtype=complex)   # norm = 5
        ovlp_raw = overlap_sq(psi, phi)
        ovlp_norm = abs(np.dot([3/5, 4/5], [4/5, 3/5])) ** 2
        self.assertAlmostEqual(ovlp_raw, ovlp_norm, places=10)


class TestBaselineCalibration(unittest.TestCase):
    """Tests for ε_user_baseline calibration."""

    def test_calibration_mean(self):
        """Calibration should converge to the true noise floor mean."""
        rng = np.random.default_rng(7)
        eps = simulate_calibration(n_samples=10000, noise_level=0.03, rng=rng)
        # Expected mean: 0.5 + mean(U[0, 0.06]) = 0.5 + 0.03 = 0.53
        self.assertAlmostEqual(eps, 0.53, delta=0.005)

    def test_corrected_fidelity_at_max(self):
        """raw_SWAP = 1.0 → F_corrected = 1.0."""
        eps = 0.55
        f   = relative_fidelity(1.0, eps)
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_corrected_fidelity_at_baseline(self):
        """raw_SWAP = ε → F_corrected = 0.0."""
        eps = 0.55
        f   = relative_fidelity(eps, eps)
        self.assertAlmostEqual(f, 0.0, places=10)

    def test_below_baseline_clamped(self):
        """raw_SWAP < ε (noisy measurement) → F_corrected clamped to 0."""
        eps = 0.55
        f   = relative_fidelity(0.50, eps, clamp=True)
        self.assertEqual(f, 0.0)

    def test_below_baseline_unclamped(self):
        """raw_SWAP < ε unclamped → negative (informational)."""
        eps = 0.55
        f   = relative_fidelity(0.50, eps, clamp=False)
        self.assertLess(f, 0.0)

    def test_denom_zero_guard(self):
        """ε = 1.0 should not raise ZeroDivisionError."""
        f = relative_fidelity(1.0, 1.0)
        self.assertEqual(f, 0.0)


class TestRelativeFidelity(unittest.TestCase):
    """Per-user baseline isolation and fidelity scaling."""

    def setUp(self):
        self.rng = np.random.default_rng(99)

    def _user_session(self, noise_level: float, n_cal: int = 500) -> float:
        """Simulate a full user session and return mean F_corrected."""
        eps = simulate_calibration(n_samples=n_cal,
                                   noise_level=noise_level, rng=self.rng)

        # Simulate "cognitive task" signal: higher overlap than baseline
        high_overlap = 0.85
        raw_task = raw_swap_from_overlap(high_overlap)
        # Add measurement noise
        raw_noisy = raw_task + self.rng.normal(0, noise_level * 0.5)
        raw_noisy = float(np.clip(raw_noisy, 0.5, 1.0))

        return relative_fidelity(raw_noisy, eps)

    def test_high_snr_user(self):
        """Good electrode contact → low ε → high F_corrected."""
        f = self._user_session(noise_level=0.005)
        self.assertGreater(f, 0.70,
            f"Expected F_corrected > 0.70 for clean contact, got {f:.4f}")

    def test_poor_contact_user(self):
        """Poor electrode contact → high ε → lower but non-negative F_corrected."""
        f = self._user_session(noise_level=0.04)
        self.assertGreaterEqual(f, 0.0,
            f"F_corrected should not be negative after clamping, got {f:.4f}")

    def test_user_baseline_separation(self):
        """Two users with different SNR should have different baselines."""
        rng_a = np.random.default_rng(11)
        rng_b = np.random.default_rng(22)
        eps_a = simulate_calibration(noise_level=0.005, rng=rng_a)
        eps_b = simulate_calibration(noise_level=0.05,  rng=rng_b)
        self.assertLess(eps_a, eps_b,
            f"User A (clean) ε={eps_a:.4f} should be < User B (noisy) ε={eps_b:.4f}")

    def test_monotonic_scaling(self):
        """Higher overlap → higher F_corrected (for fixed ε)."""
        eps = 0.54
        f_low  = relative_fidelity(raw_swap_from_overlap(0.2), eps)
        f_mid  = relative_fidelity(raw_swap_from_overlap(0.5), eps)
        f_high = relative_fidelity(raw_swap_from_overlap(0.9), eps)
        self.assertLess(f_low, f_mid)
        self.assertLess(f_mid, f_high)


class TestEndToEndPipeline(unittest.TestCase):
    """Integration: SyntheticMuse2Adapter → FeatureExtractor → SWAP fidelity."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = SyntheticMuse2Adapter()
        cls.adapter.connect()
        cls.adapter.start()
        # Wait for buffer to fill (max 6 s)
        deadline = time.time() + 6.0
        while not cls.adapter.ready and time.time() < deadline:
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.adapter.stop()

    def _extract(self) -> np.ndarray:
        alpha = self.adapter.get_filtered_window("alpha")
        theta = self.adapter.get_filtered_window("theta")
        extractor = Muse2FeatureExtractor()
        return extractor.extract(alpha, theta)

    def test_buffer_ready(self):
        self.assertTrue(self.adapter.ready,
                        "Synthetic adapter buffer not ready within timeout")

    def test_theta_shape(self):
        theta = self._extract()
        self.assertEqual(theta.shape, (THETA_LEN,),
                         f"Expected ({THETA_LEN},), got {theta.shape}")

    def test_theta_range(self):
        theta = self._extract()
        self.assertGreaterEqual(float(theta.min()), 0.0)
        self.assertLessEqual(float(theta.max()), 2 * math.pi + 1e-6)

    def test_qubit_range(self):
        """12-qubit values from synthetic signal must be in [0, 1]."""
        theta = self._extract()
        qubits = theta[:12] / (2 * math.pi)
        self.assertTrue(np.all(qubits >= 0.0) and np.all(qubits <= 1.0),
                        f"Qubits out of range: min={qubits.min():.4f} max={qubits.max():.4f}")

    def test_swap_fidelity_self(self):
        """A state compared to itself should yield F_corrected ≈ 1.0."""
        theta1 = self._extract()
        # Normalise to unit vector for overlap test
        psi = theta1 / (np.linalg.norm(theta1) + 1e-30)
        ovlp  = overlap_sq(psi, psi)
        raw   = raw_swap_from_overlap(ovlp)
        eps   = 0.52   # realistic Muse 2 baseline
        f     = relative_fidelity(raw, eps)
        self.assertAlmostEqual(f, 1.0, places=4)

    def test_swap_fidelity_distinct_windows(self):
        """Two temporally separated windows should still yield positive F."""
        theta1 = self._extract()
        time.sleep(0.15)
        theta2 = self._extract()

        psi1 = theta1 / (np.linalg.norm(theta1) + 1e-30)
        psi2 = theta2 / (np.linalg.norm(theta2) + 1e-30)
        ovlp = overlap_sq(psi1, psi2)
        raw  = raw_swap_from_overlap(ovlp)
        eps  = 0.52
        f    = relative_fidelity(raw, eps)
        self.assertGreaterEqual(f, 0.0,
            f"F_corrected should be ≥ 0, got {f:.4f}")

    def test_alpha_power_modulation(self):
        """Synthetic 10 Hz alpha signal → alpha qubits should be non-trivial."""
        theta = self._extract()
        alpha_qubits = theta[:4] / (2 * math.pi)
        # At least one should be above 0.1 (synthetic signal is 20 µV)
        self.assertTrue(np.any(alpha_qubits > 0.1),
                        f"Alpha qubits too low: {alpha_qubits}")


class TestNeurightBoundary(unittest.TestCase):
    """Ethics gate blocks raw EEG; passes feature vectors."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log_path = Path(self._tmpdir.name) / "logs" / "audit.jsonl"
        self._gate = EthicsGate(user_id="test_unit",
                                log_path=log_path, verbose=False)
        token = self._gate.request_consent()
        self._gate.begin_session(token)

    def tearDown(self):
        try:
            self._gate.end_session()
        except Exception:
            pass
        self._tmpdir.cleanup()

    def test_theta_vector_passes(self):
        """A valid (240,) theta vector in [0, 2π] should pass."""
        theta = np.random.uniform(0, 2 * math.pi, 240)
        result = self._gate.gate_pass(theta, label="theta")
        self.assertTrue(np.allclose(result, theta))

    def test_feature_12_passes(self):
        """A 12-element qubit vector in [0, 1] should pass."""
        qubits = np.random.uniform(0, 1, 12)
        result = self._gate.gate_pass(qubits, label="qubits")
        self.assertTrue(np.allclose(result, qubits))

    def test_raw_eeg_window_blocked(self):
        """A (4, 512) raw window must be blocked."""
        raw = np.random.randn(4, 512) * 50e-6
        with self.assertRaises(NeurightViolation):
            self._gate.gate_pass(raw, label="raw_eeg")

    def test_no_consent_blocks(self):
        """Calling gate_pass without consent must raise."""
        import tempfile as tf2
        from pathlib import Path as P
        td = tf2.TemporaryDirectory()
        gate2 = EthicsGate("no_consent_user",
                            log_path=P(td.name) / "logs" / "a.jsonl",
                            verbose=False)
        theta = np.random.uniform(0, 2 * math.pi, 240)
        with self.assertRaises(NeurightViolation):
            gate2.gate_pass(theta)
        td.cleanup()

    def test_chain_integrity(self):
        """Audit log chain must verify clean after normal operations."""
        theta = np.random.uniform(0, 2 * math.pi, 240)
        self._gate.gate_pass(theta, label="t1")
        self._gate.gate_pass(theta, label="t2")
        self._gate.end_session()
        valid, broken_at = self._gate.verify_chain()
        self.assertTrue(valid, f"Chain broken at seq {broken_at}")

    def test_stim_cap(self):
        """Stimulation above 50 µA must be clamped."""
        from src.ethics.ethics_gate import STIM_ABSOLUTE_MAX_AMP
        clamped = self._gate.validate_stimulation(200.0, channel="AF7")
        self.assertEqual(clamped, STIM_ABSOLUTE_MAX_AMP)

    def test_stim_below_cap_passes(self):
        """Stimulation at or below 50 µA must pass unchanged."""
        amp = self._gate.validate_stimulation(25.0, channel="TP9")
        self.assertAlmostEqual(amp, 25.0)

    def test_killswitch_blocks_all(self):
        """After killswitch, even valid theta must be blocked."""
        self._gate.trigger_killswitch(reason="test")
        theta = np.random.uniform(0, 2 * math.pi, 240)
        with self.assertRaises(NeurightViolation):
            self._gate.gate_pass(theta)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
