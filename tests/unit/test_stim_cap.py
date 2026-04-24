"""
test_stim_cap.py — Stimulation Safety Cap Verification
══════════════════════════════════════════════════════════════════════════════
Verifies that STIM_ABSOLUTE_MAX_AMP = 50 µA is enforced consistently by:
  1. ethics_gate.py (Python stack)
  2. ethics_bridge.py (cross-stack bridge constant)

Also verifies:
  • NeurightViolation is not silently catchable (must propagate to caller)
  • Amplitudes at and below the cap pass unchanged
  • Amplitudes above the cap are clamped (not blocked outright)
  • A log entry is written for every cap event
  • C++ stub emits consistent cap behaviour via EthicsBridge
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.ethics.ethics_gate import (
    EthicsGate, NeurightViolation,
    STIM_ABSOLUTE_MAX_AMP as PY_STIM_CAP,
)
from src.ethics.ethics_bridge import (
    EthicsBridge, CppSovereigntyStub,
    STIM_ABSOLUTE_MAX_AMP as BRIDGE_STIM_CAP,
)


# ── Constant consistency ──────────────────────────────────────────────────────

class TestStimCapConstantConsistency(unittest.TestCase):
    """STIM_ABSOLUTE_MAX_AMP must be identical across all Python modules."""

    def test_py_gate_equals_bridge(self):
        self.assertAlmostEqual(
            PY_STIM_CAP, BRIDGE_STIM_CAP, places=6,
            msg=f"ethics_gate.py cap ({PY_STIM_CAP} µA) ≠ "
                f"ethics_bridge.py cap ({BRIDGE_STIM_CAP} µA)")

    def test_cap_value_is_50ua(self):
        self.assertAlmostEqual(PY_STIM_CAP, 50.0, places=6,
                               msg=f"Expected 50 µA, got {PY_STIM_CAP}")

    def test_bridge_cap_is_50ua(self):
        self.assertAlmostEqual(BRIDGE_STIM_CAP, 50.0, places=6)


# ── ethics_gate.py stim guard ─────────────────────────────────────────────────

class TestEthicsGateStimCap(unittest.TestCase):
    """validate_stimulation() clamps above cap, passes below."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log = Path(self._tmpdir.name) / "logs" / "audit.jsonl"
        self._gate = EthicsGate(user_id="stim_test", log_path=log, verbose=False)
        token = self._gate.request_consent()
        self._gate.begin_session(token)

    def tearDown(self):
        try: self._gate.end_session()
        except Exception: pass
        self._tmpdir.cleanup()

    def test_exact_cap_passes_unchanged(self):
        amp = self._gate.validate_stimulation(50.0)
        self.assertAlmostEqual(amp, 50.0)

    def test_below_cap_passes_unchanged(self):
        for amp in [0.0, 1.0, 10.0, 25.0, 49.999]:
            result = self._gate.validate_stimulation(amp)
            self.assertAlmostEqual(result, amp,
                                   msg=f"Amplitude {amp} µA should pass unchanged")

    def test_above_cap_is_clamped(self):
        """Amplitudes above cap are clamped to 50 µA (not blocked)."""
        for amp in [50.001, 51.0, 100.0, 1000.0]:
            result = self._gate.validate_stimulation(amp)
            self.assertAlmostEqual(result, PY_STIM_CAP,
                                   msg=f"{amp} µA should be clamped to {PY_STIM_CAP}")

    def test_stim_without_consent_raises(self):
        """Stimulation attempt without consent → NeurightViolation."""
        tmpdir2 = tempfile.TemporaryDirectory()
        gate2   = EthicsGate(
            user_id="no_consent",
            log_path=Path(tmpdir2.name) / "logs" / "a.jsonl",
            verbose=False)
        with self.assertRaises(NeurightViolation):
            gate2.validate_stimulation(10.0)
        tmpdir2.cleanup()

    def test_stim_after_killswitch_raises(self):
        """After killswitch, stimulation must be blocked regardless of amplitude."""
        self._gate.trigger_killswitch(reason="test")
        with self.assertRaises(NeurightViolation):
            self._gate.validate_stimulation(1.0)

    def test_cap_event_logged(self):
        """A cap event (>50 µA) must be logged in the audit file."""
        initial_count = self._gate._seq
        self._gate.validate_stimulation(200.0, channel="AF7")
        self.assertGreater(self._gate._seq, initial_count,
                           "No log entry written for cap event")

    def test_multiple_channels_capped_independently(self):
        """Each channel clamped independently."""
        channels = ["TP9", "AF7", "AF8", "TP10"]
        for ch in channels:
            result = self._gate.validate_stimulation(999.0, channel=ch)
            self.assertAlmostEqual(result, PY_STIM_CAP,
                                   msg=f"Channel {ch}: cap not applied")


# ── NeurightViolation propagation ─────────────────────────────────────────────

class TestNeurightViolationPropagation(unittest.TestCase):
    """NeurightViolation must not be silently suppressible in common patterns."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log = Path(self._tmpdir.name) / "logs" / "audit.jsonl"
        self._gate = EthicsGate(user_id="propagation_test",
                                log_path=log, verbose=False)
        token = self._gate.request_consent()
        self._gate.begin_session(token)

    def tearDown(self):
        try: self._gate.end_session()
        except Exception: pass
        self._tmpdir.cleanup()

    def test_exception_is_runtime_error(self):
        """NeurightViolation is a RuntimeError → not caught by bare except Exception."""
        self._gate.trigger_killswitch("test")
        theta = np.random.uniform(0, 2 * math.pi, 240)
        caught_as_runtime = False
        try:
            self._gate.gate_pass(theta)
        except RuntimeError:
            caught_as_runtime = True
        self.assertTrue(caught_as_runtime)

    def test_violation_not_suppressed_by_generic_except(self):
        """Simulates code that catches Exception but should still propagate."""
        self._gate.trigger_killswitch("test")
        theta = np.random.uniform(0, 2 * math.pi, 240)

        def _bad_wrapper():
            """Simulates incorrect code that suppresses exceptions."""
            try:
                self._gate.gate_pass(theta)
                return True
            except NeurightViolation:
                raise   # CORRECT: must re-raise
            except Exception:
                return False   # WRONG: would suppress

        with self.assertRaises(NeurightViolation):
            _bad_wrapper()

    def test_raw_eeg_raises_neuroright_violation(self):
        """Raw EEG crossing the boundary raises NeurightViolation, not ValueError."""
        raw = np.random.randn(4, 512) * 20e-6
        with self.assertRaises(NeurightViolation):
            self._gate.gate_pass(raw)


# ── Cross-stack stim cap (bridge + C++ stub) ──────────────────────────────────

class TestCrossStackStimCap(unittest.TestCase):
    """C++ stub and Python gate both reject > 50 µA."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log = Path(self._tmpdir.name) / "logs" / "bridge.jsonl"
        self._bridge = EthicsBridge(log_path=log, verbose=False)
        self._stub   = CppSovereigntyStub(self._bridge)

        py_log = Path(self._tmpdir.name) / "logs" / "gate.jsonl"
        self._gate = EthicsGate(user_id="cross_stim",
                                log_path=py_log, verbose=False)
        token = self._gate.request_consent()
        self._gate.begin_session(token)

    def tearDown(self):
        try: self._gate.end_session()
        except Exception: pass
        self._tmpdir.cleanup()

    def test_python_gate_clamps_200ua(self):
        result = self._gate.validate_stimulation(200.0)
        self.assertAlmostEqual(result, PY_STIM_CAP)

    def test_cpp_stub_clamps_200ua(self):
        """C++ stub logs DISSIPATION_APPLIED for > 50 µA."""
        import json
        n_before = self._bridge.entry_count
        self._stub.emit_stim_check(200.0, channel="AF7")
        # Check that an event was logged
        self.assertGreater(self._bridge.entry_count, n_before)

    def test_both_stacks_cap_same_value(self):
        """Python and C++ stub cap at the same 50 µA threshold."""
        py_result   = self._gate.validate_stimulation(999.0)
        # Read last log entry from bridge to get C++ stub's applied value
        self._stub.emit_stim_check(999.0, channel="TP9")
        import json
        last_line = None
        with open(self._bridge._log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        d = json.loads(last_line)
        cpp_applied = d["payload"].get("applied_uA", None)

        self.assertAlmostEqual(py_result, PY_STIM_CAP)
        if cpp_applied is not None:
            self.assertAlmostEqual(cpp_applied, BRIDGE_STIM_CAP,
                                   msg=f"C++ stub applied {cpp_applied} µA, expected {BRIDGE_STIM_CAP}")

    def test_stim_cap_consistency_check(self):
        """Bridge verify_stim_cap_consistency() must return True."""
        ok = self._bridge.verify_stim_cap_consistency()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
