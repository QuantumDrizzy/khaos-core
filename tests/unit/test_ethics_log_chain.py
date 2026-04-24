"""
test_ethics_log_chain.py — Cross-Stack Audit Chain Integrity
══════════════════════════════════════════════════════════════════════════════
Verifies that 1000 events generated alternating between the Python ethics_gate
and the C++ stub (via CppSovereigntyStub) maintain a valid SHA-256 chain.

Also verifies:
  • No seq gaps in a mixed Python/C++ chain
  • Tampered entry detected by verify_chain()
  • BridgeEntry canonical form is deterministic
  • Raw EEG payload blocked at bridge level
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.ethics.ethics_bridge import (
    BridgeEntry, BridgeEvent, BridgeStack,
    CppSovereigntyStub, EthicsBridge, STIM_ABSOLUTE_MAX_AMP,
)
from src.ethics.ethics_gate import EthicsGate, NeurightViolation


class TestBridgeEntryCanonical(unittest.TestCase):
    """BridgeEntry canonical form is deterministic and matches SHA-256."""

    def _make_entry(self, seq: int = 0) -> BridgeEntry:
        e = BridgeEntry(
            seq=seq, timestamp_ns=1_000_000_000,
            stack=BridgeStack.PYTHON,
            event_type=BridgeEvent.GATE_PASS,
            payload={"label": "theta", "norm": 3.14},
            hash_prev="0" * 64,
        )
        e.hash = e.compute_hash()
        return e

    def test_canonical_is_deterministic(self):
        e1 = self._make_entry(0)
        e2 = self._make_entry(0)
        self.assertEqual(e1.canonical(), e2.canonical())

    def test_hash_matches_canonical(self):
        e = self._make_entry(0)
        expected = hashlib.sha256(e.canonical().encode()).hexdigest()
        self.assertEqual(e.hash, expected)

    def test_different_seq_different_hash(self):
        e0 = self._make_entry(0)
        e1 = self._make_entry(1)
        self.assertNotEqual(e0.hash, e1.hash)

    def test_roundtrip_json(self):
        e = self._make_entry(0)
        d = e.to_dict()
        e2 = BridgeEntry.from_dict(d)
        self.assertEqual(e.canonical(), e2.canonical())
        self.assertEqual(e.hash, e2.hash)


class TestMixedChainIntegrity(unittest.TestCase):
    """1000 alternating Python/C++ events produce valid chain."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log_path = Path(self._tmpdir.name) / "logs" / "bridge.jsonl"
        self._bridge = EthicsBridge(log_path=log_path, verbose=False)
        self._stub   = CppSovereigntyStub(self._bridge)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_1000_alternating_events(self):
        """1000 events, alternating stacks → chain valid."""
        for i in range(500):
            self._bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS,
                              {"i": i, "label": "theta"})
            self._stub.emit_gate_pass(f"cpp_{i}")

        self.assertEqual(self._bridge.entry_count, 1000)
        valid, broken_at = self._bridge.verify_chain()
        self.assertTrue(valid, f"Chain broken at seq {broken_at}")

    def test_no_seq_gaps(self):
        """Sequences must be contiguous 0, 1, 2, …"""
        for i in range(10):
            self._bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS, {"i": i})
            self._stub.emit_gate_pass()

        entries = []
        with open(self._bridge._log_path, encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        seqs = [e["seq"] for e in entries]
        self.assertEqual(seqs, list(range(len(seqs))),
                          f"Seq gaps detected: {seqs}")

    def test_tampered_entry_detected(self):
        """Changing a payload field after writing breaks the chain."""
        for i in range(5):
            self._bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS, {"i": i})

        # Read and tamper entry 2
        lines = self._bridge._log_path.read_text().splitlines()
        d = json.loads(lines[2])
        d["payload"]["i"] = 9999
        lines[2] = json.dumps(d)
        self._bridge._log_path.write_text("\n".join(lines) + "\n")

        valid, broken_at = self._bridge.verify_chain()
        self.assertFalse(valid, "Tampered entry not detected")
        self.assertEqual(broken_at, 2)

    def test_handshake_completes(self):
        """3-way handshake signs and verifies successfully."""
        challenge = self._bridge.initiate_handshake()
        response  = self._stub.sign_challenge(challenge)
        ok        = self._bridge.verify_handshake(challenge, response)
        self.assertTrue(ok)

    def test_wrong_response_fails_handshake(self):
        """Wrong HMAC response must fail verification and log violation."""
        challenge = self._bridge.initiate_handshake()
        ok        = self._bridge.verify_handshake(challenge, "0" * 64)
        self.assertFalse(ok)

    def test_raw_eeg_payload_blocked(self):
        """Raw EEG array in payload must raise ValueError."""
        raw = np.random.randn(4, 512)
        with self.assertRaises(ValueError):
            self._bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_BLOCK,
                              {"raw_eeg": raw})

    def test_stim_cap_consistency(self):
        """Stim cap must be identical in bridge and ethics_gate."""
        ok = self._bridge.verify_stim_cap_consistency()
        self.assertTrue(ok, "STIM_ABSOLUTE_MAX_AMP mismatch between stacks")


class TestEthicsGateChain(unittest.TestCase):
    """ethics_gate.py chain stays valid across many gate operations."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        log = Path(self._tmpdir.name) / "logs" / "audit.jsonl"
        self._gate = EthicsGate(user_id="chain_test", log_path=log, verbose=False)
        token = self._gate.request_consent()
        self._gate.begin_session(token)

    def tearDown(self):
        try: self._gate.end_session()
        except Exception: pass
        self._tmpdir.cleanup()

    def test_100_gate_passes_valid_chain(self):
        """100 gate passes → chain intact."""
        import math
        for i in range(100):
            theta = np.random.uniform(0, 2 * math.pi, 240)
            self._gate.gate_pass(theta, label=f"theta_{i}")
        self._gate.end_session()
        valid, broken_at = self._gate.verify_chain()
        self.assertTrue(valid, f"Chain broken at {broken_at}")

    def test_mixed_pass_block_chain(self):
        """Interleaved PASS and BLOCK events still form a valid chain."""
        import math
        for i in range(50):
            # pass
            theta = np.random.uniform(0, 2 * math.pi, 240)
            self._gate.gate_pass(theta)
            # try to pass raw EEG → BLOCK
            raw = np.random.randn(4, 512) * 20e-6
            try:
                self._gate.gate_pass(raw)
            except NeurightViolation:
                pass

        self._gate.end_session()
        valid, _ = self._gate.verify_chain()
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
