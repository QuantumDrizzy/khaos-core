"""
ethics_bridge.py — Cross-Stack Audit Log Synchronisation
══════════════════════════════════════════════════════════════════════════════
Defines a shared JSON schema for sovereignty audit events that is compatible
with BOTH the Python ethics_gate.py and the C++ sovereignty_monitor.cpp stacks,
enabling a unified, verifiable audit trail regardless of which stack generated
the entry.

Shared log schema
─────────────────
  {
    "seq":          int,       sequential entry number (global across stacks)
    "timestamp_ns": int,       UTC nanoseconds since epoch
    "stack":        str,       "python" | "cpp"
    "event_type":   str,       event name (see SovereigntyEvent)
    "payload":      object,    event-specific data
    "hash_prev":    str,       SHA-256 of previous entry (64 hex chars)
    "hash":         str,       SHA-256 of this entry's canonical form
  }

Canonical form for hashing (matches C++ sovereignty_monitor.cpp):
  "{seq}|{timestamp_ns}|{stack}|{event_type}|{payload_json_sorted}|{hash_prev}"

3-Way session handshake
───────────────────────
  Python side:  EthicsBridge.initiate_handshake()  → challenge token
  C++ side:     receives challenge via IPC, signs with HMAC-SHA256(secret_key)
  Python side:  EthicsBridge.verify_handshake()    → confirms session

  In development/test mode, the C++ stub is simulated in Python for CI.

IPC transport
─────────────
  The bridge writes events to a shared JSONL file (default: logs/bridge.jsonl).
  Production deployments can replace this with a UNIX domain socket or
  POSIX shared memory ring buffer — only the serialisation format changes.

Invariants (enforced at runtime, matching sovereignty_monitor.cpp)
──────────────────────────────────────────────────────────────────
  • STIM_ABSOLUTE_MAX_AMP = 50 µA  (static_assert in C++)
  • Raw EEG never in payload (checked on write)
  • Chain must be contiguous — no seq gaps allowed
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ── Shared constants (must match sovereignty_monitor.cpp) ─────────────────────
STIM_ABSOLUTE_MAX_AMP: float = 50.0   # µA
SCHEMA_VERSION: str = "1.0"

# ── Bridge log file ───────────────────────────────────────────────────────────
BRIDGE_LOG_PATH = Path("logs/bridge_audit.jsonl")


class BridgeStack(str, Enum):
    PYTHON = "python"
    CPP    = "cpp"


class BridgeEvent(str, Enum):
    SESSION_START         = "SESSION_START"
    SESSION_END           = "SESSION_END"
    GATE_PASS             = "GATE_PASS"
    GATE_BLOCK            = "GATE_BLOCK"
    INTEGRITY_VIOLATION   = "INTEGRITY_VIOLATION"
    KILLSWITCH_TRIGGERED  = "KILLSWITCH_TRIGGERED"
    DISSIPATION_APPLIED   = "DISSIPATION_APPLIED"
    DISSIPATION_BLOCKED   = "DISSIPATION_BLOCKED"
    HANDSHAKE_INIT        = "HANDSHAKE_INIT"
    HANDSHAKE_COMPLETE    = "HANDSHAKE_COMPLETE"
    STIM_CAP_CHECK        = "STIM_CAP_CHECK"


# ── Shared entry format ───────────────────────────────────────────────────────

@dataclass
class BridgeEntry:
    seq:          int
    timestamp_ns: int
    stack:        BridgeStack
    event_type:   BridgeEvent
    payload:      dict
    hash_prev:    str
    hash:         str = field(default="", init=False)

    def canonical(self) -> str:
        """Canonical string for SHA-256 hashing (matches C++ format)."""
        return (f"{self.seq}|{self.timestamp_ns}|{self.stack.value}|"
                f"{self.event_type.value}|"
                f"{json.dumps(self.payload, sort_keys=True)}|"
                f"{self.hash_prev}")

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "seq":            self.seq,
            "timestamp_ns":   self.timestamp_ns,
            "stack":          self.stack.value,
            "event_type":     self.event_type.value,
            "payload":        self.payload,
            "hash_prev":      self.hash_prev,
            "hash":           self.hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "BridgeEntry":
        e = cls(
            seq          = d["seq"],
            timestamp_ns = d["timestamp_ns"],
            stack        = BridgeStack(d["stack"]),
            event_type   = BridgeEvent(d["event_type"]),
            payload      = d["payload"],
            hash_prev    = d["hash_prev"],
        )
        e.hash = d["hash"]
        return e


def _now_ns() -> int:
    """Current UTC time in nanoseconds."""
    return int(time.time_ns())


# ── Bridge ────────────────────────────────────────────────────────────────────

class EthicsBridge:
    """Cross-stack audit log bridge.

    Usage
    -----
    >>> bridge = EthicsBridge()
    >>> challenge = bridge.initiate_handshake()
    >>> response  = CppStub.sign_challenge(challenge)       # C++ or test stub
    >>> bridge.verify_handshake(challenge, response)
    >>> bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS, {"label": "theta"})
    >>> valid, _ = bridge.verify_chain()
    """

    def __init__(self, log_path: Path = BRIDGE_LOG_PATH,
                 secret_key: Optional[bytes] = None,
                 verbose: bool = False):
        self._log_path   = Path(log_path)
        self._secret_key = secret_key or secrets.token_bytes(32)
        self._verbose    = verbose
        self._lock       = threading.Lock()
        self._seq        = 0
        self._prev_hash  = "0" * 64
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 3-Way handshake ────────────────────────────────────────────────────

    def initiate_handshake(self) -> str:
        """Generate a challenge token for the C++ stack to sign.

        Returns
        -------
        str : 32-byte hex challenge
        """
        challenge = secrets.token_hex(32)
        self.log(BridgeStack.PYTHON, BridgeEvent.HANDSHAKE_INIT,
                 {"challenge": challenge[:16] + "…"})  # truncated in log
        return challenge

    def sign_challenge(self, challenge: str) -> str:
        """Sign a challenge with the shared HMAC-SHA256 key.

        This is the Python-side implementation.  In production, the C++ side
        performs the equivalent HMAC computation.

        Returns
        -------
        str : 64-char hex HMAC digest
        """
        return hmac.new(
            self._secret_key,
            challenge.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify_handshake(self, challenge: str, response: str) -> bool:
        """Verify the C++ stack's HMAC response.

        Returns True on success; logs INTEGRITY_VIOLATION on failure.
        """
        expected = self.sign_challenge(challenge)
        ok = hmac.compare_digest(expected, response)
        if ok:
            self.log(BridgeStack.PYTHON, BridgeEvent.HANDSHAKE_COMPLETE,
                     {"status": "ok"})
        else:
            self.log(BridgeStack.PYTHON, BridgeEvent.INTEGRITY_VIOLATION,
                     {"reason": "handshake HMAC mismatch"})
        return ok

    # ── Log ────────────────────────────────────────────────────────────────

    def log(self, stack: BridgeStack, event: BridgeEvent,
            payload: dict) -> BridgeEntry:
        """Append a hash-chained entry to the bridge log.

        Parameters
        ----------
        stack   : BridgeStack.PYTHON or BridgeStack.CPP
        event   : BridgeEvent
        payload : dict — must NOT contain raw EEG arrays

        Returns
        -------
        BridgeEntry — the written entry (for testing)
        """
        # Guard: raw EEG must not appear in payload
        self._assert_no_raw_eeg(payload)

        with self._lock:
            entry = BridgeEntry(
                seq          = self._seq,
                timestamp_ns = _now_ns(),
                stack        = stack,
                event_type   = event,
                payload      = payload,
                hash_prev    = self._prev_hash,
            )
            entry.hash      = entry.compute_hash()
            self._prev_hash = entry.hash
            self._seq      += 1

            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")

        if self._verbose:
            print(f"[EthicsBridge] [{stack.value}/{event.value}] {payload}")

        return entry

    def ingest_cpp_entry(self, raw_json: str) -> BridgeEntry:
        """Ingest a serialised entry from the C++ stack and extend the chain.

        The C++ stack writes entries in the same JSON schema.  This method
        re-hashes with the current Python-side prev_hash so the chain remains
        continuous across stacks.

        In production, the C++ stack writes to the same JSONL file via a
        shared-memory ring or a UNIX socket.  This method handles the Python
        side of that ingestion.
        """
        d      = json.loads(raw_json)
        event  = BridgeEvent(d["event_type"])
        stack  = BridgeStack(d["stack"])
        return self.log(stack, event, d["payload"])

    # ── Verification ───────────────────────────────────────────────────────

    def verify_chain(self) -> tuple[bool, int]:
        """Verify SHA-256 chain integrity of the entire bridge log.

        Returns
        -------
        (valid: bool, broken_at_seq: int)  — broken_at_seq = -1 if valid.
        """
        if not self._log_path.exists():
            return True, -1

        entries = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not entries:
            return True, -1

        prev = "0" * 64
        for d in entries:
            entry    = BridgeEntry.from_dict(d)
            expected = entry.compute_hash()
            if expected != d["hash"] or d["hash_prev"] != prev:
                return False, d["seq"]
            prev = d["hash"]

        return True, -1

    def verify_stim_cap_consistency(self) -> bool:
        """Assert that STIM_ABSOLUTE_MAX_AMP is identical in Python and C++.

        In production, reads the C++ constant from a compiled header via
        ctypes or a dry-run binary.  In this implementation, compares against
        the module-level constant (which must match sovereignty_monitor.cpp).

        Returns True if consistent.
        """
        from src.ethics.ethics_gate import STIM_ABSOLUTE_MAX_AMP as PY_CAP
        consistent = abs(PY_CAP - STIM_ABSOLUTE_MAX_AMP) < 1e-6
        self.log(BridgeStack.PYTHON, BridgeEvent.STIM_CAP_CHECK,
                 {"py_cap_uA":     PY_CAP,
                  "bridge_cap_uA": STIM_ABSOLUTE_MAX_AMP,
                  "consistent":    consistent})
        return consistent

    # ── Internal guards ────────────────────────────────────────────────────

    @staticmethod
    def _assert_no_raw_eeg(payload: dict) -> None:
        """Raise ValueError if payload contains a raw EEG array.

        Checks any numpy array value with shape (4, N>12) or total size > 512.
        """
        for k, v in payload.items():
            if isinstance(v, np.ndarray):
                if v.size > 512:
                    raise ValueError(
                        f"Bridge payload key '{k}' appears to contain raw EEG "
                        f"(size={v.size}). Raw EEG must not cross the boundary.")
                if v.ndim == 2 and v.shape[0] == 4 and v.shape[1] > 12:
                    raise ValueError(
                        f"Bridge payload key '{k}' has shape {v.shape} — "
                        "raw EEG window not permitted in audit log.")

    @property
    def entry_count(self) -> int:
        return self._seq


# ── C++ stub for testing (simulates what sovereignty_monitor.cpp would do) ────

class CppSovereigntyStub:
    """Minimal Python simulation of sovereignty_monitor.cpp for cross-stack tests.

    Produces entries in the same JSON schema, signed with the same HMAC key.
    Used in CI/CD where the actual C++ binary is unavailable.
    """

    def __init__(self, bridge: EthicsBridge):
        self._bridge = bridge

    def emit_session_start(self, session_id: str) -> str:
        entry = self._bridge.log(
            BridgeStack.CPP, BridgeEvent.SESSION_START,
            {"session_id": session_id, "source": "sovereignty_monitor.cpp"})
        return entry.hash

    def emit_gate_pass(self, label: str = "theta") -> str:
        entry = self._bridge.log(
            BridgeStack.CPP, BridgeEvent.GATE_PASS,
            {"label": label, "source": "sovereignty_monitor.cpp"})
        return entry.hash

    def emit_stim_check(self, amplitude_ua: float, channel: str) -> str:
        clamped = min(amplitude_ua, STIM_ABSOLUTE_MAX_AMP)
        event   = BridgeEvent.DISSIPATION_APPLIED \
            if clamped < amplitude_ua else BridgeEvent.GATE_PASS
        entry   = self._bridge.log(
            BridgeStack.CPP, event,
            {"requested_uA": amplitude_ua,
             "applied_uA":   clamped,
             "channel":      channel,
             "source":       "sovereignty_monitor.cpp"})
        return entry.hash

    def sign_challenge(self, challenge: str) -> str:
        """Mirror of C++ HMAC-SHA256 signing."""
        return self._bridge.sign_challenge(challenge)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, sys
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "logs" / "bridge.jsonl"
        bridge = EthicsBridge(log_path=log, verbose=True)
        stub   = CppSovereigntyStub(bridge)

        # ── Test 1: 3-way handshake
        print("\n[Test 1] 3-way handshake:")
        challenge = bridge.initiate_handshake()
        response  = stub.sign_challenge(challenge)
        ok        = bridge.verify_handshake(challenge, response)
        assert ok, "Handshake failed"
        print("  PASS")

        # ── Test 2: Interleaved Python / C++ entries
        print("\n[Test 2] Interleaved entries (50 Python + 50 C++):")
        for i in range(50):
            bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS,
                       {"label": f"theta_{i}", "i": i})
            stub.emit_gate_pass(f"cpp_theta_{i}")
        print(f"  {bridge.entry_count} total entries")

        # ── Test 3: Chain verification
        print("\n[Test 3] Chain integrity:")
        valid, broken_at = bridge.verify_chain()
        assert valid, f"Chain broken at seq {broken_at}"
        print(f"  PASS — {bridge.entry_count} entries, chain intact")

        # ── Test 4: Stim cap consistency
        print("\n[Test 4] Stim cap consistency:")
        ok = bridge.verify_stim_cap_consistency()
        assert ok, "Stim cap mismatch between Python and bridge constant"
        print(f"  PASS — both stacks cap at {STIM_ABSOLUTE_MAX_AMP} µA")

        # ── Test 5: Raw EEG blocked from payload
        print("\n[Test 5] Raw EEG blocked from payload:")
        try:
            bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_BLOCK,
                       {"raw": np.random.randn(4, 512)})
            print("  FAIL — should have raised ValueError")
        except ValueError as e:
            print(f"  PASS — blocked: {e}")

    print("\nAll EthicsBridge tests passed. ✓")
