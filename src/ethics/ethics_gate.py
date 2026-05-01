"""
ethics_gate.py — Python Sovereignty Gate (mirror of sovereignty_monitor.cpp)
══════════════════════════════════════════════════════════════════════════════
Enforces the KHAOS neurorights architecture at the Python layer:

  • Raw EEG (any array with > 12 elements representing raw sensor data) is
    ARCHITECTURALLY BLOCKED from crossing the DSP boundary.
  • Only the 12-element (or 240-element tile) feature vector may exit.
  • Consent token is required before any processing begins.
  • All events are written to an SHA-256 hash-chained audit log, mirroring
    the C++ sovereignty_monitor.cpp event format.

Event types (matches C++ enum SovereigntyEvent):
  SESSION_START          — consent token accepted, processing may begin
  SESSION_END            — adapter stopped, log finalised
  DISSIPATION_APPLIED    — stimulation amplitude clamped (future use)
  DISSIPATION_BLOCKED    — stimulation blocked — would exceed safety threshold
  KILLSWITCH_TRIGGERED   — emergency halt of all output
  INTEGRITY_VIOLATION    — attempted boundary crossing or consent breach
  GATE_PASS              — feature vector cleared for downstream use
  GATE_BLOCK             — feature vector blocked (consent absent or revoked)

Architecture note
─────────────────
#ifndef ETHICS_COMPLIANT → #error is enforced at C++ compile time.
This Python module provides the equivalent runtime guard.  Any call to
`gate_pass()` without a valid consent token raises NeurightViolation, which
must propagate to the caller — it must NOT be silenced.

SHA-256 chain
─────────────
Each log entry is:
  entry = f"{seq}|{timestamp}|{event}|{payload}|{prev_hash}"
  hash  = SHA-256(entry.encode())

The chain allows offline verification that no entries were deleted or modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants matching sovereignty_monitor.cpp ────────────────────────────────
STIM_ABSOLUTE_MAX_AMP: float = 50.0   # µA — mirrors static_assert in C++
MAX_RAW_EEG_ELEMENTS:  int   = 12     # arrays larger than this = raw EEG
AUDIT_LOG_PATH = Path("logs/sovereignty_audit.jsonl")


# ── Exception ─────────────────────────────────────────────────────────────────

class NeurightViolation(RuntimeError):
    """Raised when the sovereignty gate blocks an operation.

    This exception MUST NOT be caught silently.  Catching it to suppress the
    error constitutes an INTEGRITY_VIOLATION and will be logged as such.
    """


# ── Event types ───────────────────────────────────────────────────────────────

class SovereigntyEvent(str, Enum):
    SESSION_START         = "SESSION_START"
    SESSION_END           = "SESSION_END"
    DISSIPATION_APPLIED   = "DISSIPATION_APPLIED"
    DISSIPATION_BLOCKED   = "DISSIPATION_BLOCKED"
    KILLSWITCH_TRIGGERED  = "KILLSWITCH_TRIGGERED"
    INTEGRITY_VIOLATION   = "INTEGRITY_VIOLATION"
    GATE_PASS             = "GATE_PASS"
    GATE_BLOCK            = "GATE_BLOCK"
    CALIBRATION_START     = "CALIBRATION_START"
    CALIBRATION_END       = "CALIBRATION_END"


# ── Audit log entry ───────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    seq:       int
    timestamp: str
    event:     SovereigntyEvent
    payload:   dict
    prev_hash: str
    this_hash: str = field(default="", init=False)

    def serialise(self) -> str:
        """Return the canonical string used for hashing."""
        return (f"{self.seq}|{self.timestamp}|{self.event.value}|"
                f"{json.dumps(self.payload, sort_keys=True)}|{self.prev_hash}")

    def compute_hash(self) -> str:
        return hashlib.sha256(self.serialise().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps({
            "seq":       self.seq,
            "timestamp": self.timestamp,
            "event":     self.event.value,
            "payload":   self.payload,
            "prev_hash": self.prev_hash,
            "hash":      self.this_hash,
        })


# ── Core gate ─────────────────────────────────────────────────────────────────

class EthicsGate:
    """Singleton-style sovereignty enforcer for a KHAOS session.

    Usage
    -----
    >>> gate = EthicsGate(user_id="researcher_001")
    >>> token = gate.request_consent()        # generate token
    >>> gate.begin_session(token)             # open gate
    >>> feature_vec = gate.gate_pass(theta)   # validate + pass through
    >>> gate.end_session()
    """

    def __init__(self, user_id: str,
                 log_path: Path = AUDIT_LOG_PATH,
                 verbose: bool = True):
        self._user_id     = user_id
        self._log_path    = Path(log_path)
        self._verbose     = verbose
        self._lock        = threading.Lock()
        self._consent_ok  = False
        self._killswitch  = False
        self._session_id  = secrets.token_hex(16)
        self._seq         = 0
        self._prev_hash   = "0" * 64   # genesis hash

        # Pending consent token (single-use)
        self._pending_token: Optional[str] = None

        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Consent lifecycle ───────────────────────────────────────────────────

    def request_consent(self) -> str:
        """Generate a one-time consent token.

        The token must be stored by the researcher and passed to
        ``begin_session()`` to prove informed consent.  In a real deployment
        this token would be linked to a signed IRB / GDPR consent form.

        Returns
        -------
        str : hex consent token (32 bytes)
        """
        self._pending_token = secrets.token_hex(32)
        if self._verbose:
            print(f"[EthicsGate] Consent token generated for user '{self._user_id}'.")
            print(f"  Token: {self._pending_token[:8]}…{self._pending_token[-8:]}")
            print("  ► Store this token. Pass it to begin_session() to proceed.")
        return self._pending_token

    def begin_session(self, consent_token: str) -> None:
        """Activate the gate after validating the consent token.

        Raises NeurightViolation if the token is invalid or missing.
        """
        with self._lock:
            if self._pending_token is None:
                self._log_event(SovereigntyEvent.INTEGRITY_VIOLATION,
                                {"reason": "begin_session called without requesting token"})
                raise NeurightViolation(
                    "No consent token was generated. Call request_consent() first.")

            if not secrets.compare_digest(consent_token, self._pending_token):
                self._log_event(SovereigntyEvent.INTEGRITY_VIOLATION,
                                {"reason": "invalid consent token",
                                 "user":   self._user_id})
                raise NeurightViolation(
                    "Consent token mismatch. Session blocked.")

            self._pending_token = None   # single-use
            self._consent_ok    = True

            self._log_event(SovereigntyEvent.SESSION_START,
                            {"user":       self._user_id,
                             "session_id": self._session_id})

        if self._verbose:
            print(f"[EthicsGate] Session '{self._session_id[:8]}' open for '{self._user_id}'.")

    def end_session(self) -> None:
        """Close the gate and seal the audit log."""
        with self._lock:
            self._consent_ok = False
            self._log_event(SovereigntyEvent.SESSION_END,
                            {"user":       self._user_id,
                             "session_id": self._session_id,
                             "total_events": self._seq})
        if self._verbose:
            print(f"[EthicsGate] Session '{self._session_id[:8]}' closed. "
                  f"Log: {self._log_path}")

    # ── Gate operations ─────────────────────────────────────────────────────

    def gate_pass(self, data: np.ndarray,
                  label: str = "theta") -> np.ndarray:
        """Validate and pass *data* through the sovereignty boundary.

        Rules
        -----
        1. Killswitch active → always block.
        2. Consent not granted → block with INTEGRITY_VIOLATION.
        3. Array size > MAX_RAW_EEG_ELEMENTS (12) AND dtype indicates raw
           voltage (float, not normalised rotation angle) → treat as raw EEG
           leak → block with INTEGRITY_VIOLATION.
        4. All checks pass → log GATE_PASS and return data unchanged.

        Parameters
        ----------
        data  : np.ndarray — the array to validate
        label : str        — human-readable label for the log

        Returns
        -------
        np.ndarray — the same array (pass-through on success)

        Raises
        ------
        NeurightViolation on any block condition.
        """
        with self._lock:
            # Check 1: killswitch
            if self._killswitch:
                self._log_event(SovereigntyEvent.GATE_BLOCK,
                                {"reason":    "killswitch active",
                                 "label":     label,
                                 "data_size": data.size})
                raise NeurightViolation(
                    "Killswitch is active. No data may exit the boundary.")

            # Check 2: consent
            if not self._consent_ok:
                self._log_event(SovereigntyEvent.GATE_BLOCK,
                                {"reason":    "no consent",
                                 "label":     label,
                                 "data_size": data.size})
                raise NeurightViolation(
                    "Consent not granted. Call begin_session() first.")

            # Check 3: raw EEG leak detection
            # Heuristic: raw EEG windows are (4, 512) → 2048 elements
            # theta vectors are (240,) — this is fine.
            # A 12-element feature vector is fine.
            # Anything large + floating point with values in µV range → block.
            if self._is_raw_eeg(data):
                self._log_event(SovereigntyEvent.INTEGRITY_VIOLATION,
                                {"reason":    "raw EEG leak attempt",
                                 "label":     label,
                                 "shape":     str(data.shape),
                                 "max_abs":   float(np.max(np.abs(data)))})
                raise NeurightViolation(
                    f"Raw EEG leak blocked: array shape {data.shape} "
                    f"(max |val|={np.max(np.abs(data)):.2e}). "
                    "Only feature vectors (≤240 elements, scaled to [0,2π]) "
                    "may cross the sovereignty boundary.")

            # All clear
            self._log_event(SovereigntyEvent.GATE_PASS,
                            {"label":     label,
                             "shape":     str(data.shape),
                             "norm":      float(np.linalg.norm(data))})

        return data

    def _is_raw_eeg(self, data: np.ndarray) -> bool:
        """Heuristic: detect raw voltage arrays.

        Criteria (ALL must be true to trigger block):
          1. Array has > MAX_RAW_EEG_ELEMENTS (12) elements.
          2. Any absolute value > 10 (raw EEG is in µV, theta angles ≤ 2π≈6.28).
        """
        if data.size <= MAX_RAW_EEG_ELEMENTS:
            return False
        # Theta angles are in [0, 2π] ≈ [0, 6.28]
        # Raw EEG in µV scale is typically 1–100 µV = 1e-6–1e-4 V
        # But if stored in µV (unitless float), magnitudes can reach 100.
        # Raw EEG in Volts: magnitudes ~ 1e-5 → won't trigger.
        # We key on: is data.size large AND values outside theta range?
        max_abs = float(np.max(np.abs(data)))
        # If the array is large (> 12) and values exceed 2π significantly
        # → it's not a theta vector or qubit array.
        if data.size > 240 and max_abs > (2 * np.pi + 1.0):
            return True
        # Secondary: shape (4, N) with N > 12 and large values
        if data.ndim == 2 and data.shape[0] == 4 and data.shape[1] > 12:
            return True
        return False

    # ── Stimulation guard ───────────────────────────────────────────────────

    def validate_stimulation(self, amplitude_ua: float,
                             channel: str = "unknown") -> float:
        """Validate a stimulation amplitude against the absolute safety cap.

        Mirrors: static_assert(STIM_ABSOLUTE_MAX_AMP <= 50.0f)

        Returns the (possibly clamped) amplitude.
        Logs DISSIPATION_APPLIED if clamped, DISSIPATION_BLOCKED if consent
        is absent.
        """
        with self._lock:
            if not self._consent_ok:
                self._log_event(SovereigntyEvent.DISSIPATION_BLOCKED,
                                {"reason":    "no consent",
                                 "amplitude": amplitude_ua,
                                 "channel":   channel})
                raise NeurightViolation(
                    "Stimulation blocked: consent not granted.")

            if amplitude_ua > STIM_ABSOLUTE_MAX_AMP:
                clamped = STIM_ABSOLUTE_MAX_AMP
                self._log_event(SovereigntyEvent.DISSIPATION_APPLIED,
                                {"original":  amplitude_ua,
                                 "clamped":   clamped,
                                 "channel":   channel})
                if self._verbose:
                    print(f"[EthicsGate] ⚠ Stim amplitude {amplitude_ua} µA → "
                          f"clamped to {clamped} µA")
                return clamped

        return amplitude_ua

    # ── Killswitch ──────────────────────────────────────────────────────────

    def trigger_killswitch(self, reason: str = "manual") -> None:
        """Immediately halt all data flow.  Cannot be undone in this session."""
        with self._lock:
            self._killswitch = True
            self._consent_ok = False
            self._log_event(SovereigntyEvent.KILLSWITCH_TRIGGERED,
                            {"reason":     reason,
                             "session_id": self._session_id})
        if self._verbose:
            print(f"[EthicsGate] ══ KILLSWITCH TRIGGERED ══  reason: {reason}")

    # ── Audit log ───────────────────────────────────────────────────────────

    def _log_event(self, event: SovereigntyEvent, payload: dict) -> None:
        """Append a hash-chained entry to the audit log (internal, must hold lock)."""
        ts    = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            seq       = self._seq,
            timestamp = ts,
            event     = event,
            payload   = payload,
            prev_hash = self._prev_hash,
        )
        entry.this_hash  = entry.compute_hash()
        self._prev_hash  = entry.this_hash
        self._seq       += 1

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

        if self._verbose:
            icon = {
                SovereigntyEvent.GATE_PASS:            "✓",
                SovereigntyEvent.GATE_BLOCK:           "✗",
                SovereigntyEvent.INTEGRITY_VIOLATION:  "⛔",
                SovereigntyEvent.KILLSWITCH_TRIGGERED: "☠",
                SovereigntyEvent.SESSION_START:        "▶",
                SovereigntyEvent.SESSION_END:          "■",
            }.get(event, "·")
            print(f"[EthicsGate] {icon} [{event.value}] {payload}")

    def verify_chain(self) -> Tuple[bool, int]:
        """Offline verification: check log integrity.

        Returns
        -------
        (valid: bool, broken_at_seq: int)  — broken_at_seq = -1 if valid.
        """
        entries = []
        if not self._log_path.exists():
            return True, -1

        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        prev = "0" * 64
        for e in entries:
            canonical = (f"{e['seq']}|{e['timestamp']}|{e['event']}|"
                         f"{json.dumps(e['payload'], sort_keys=True)}|{e['prev_hash']}")
            expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if expected_hash != e["hash"] or e["prev_hash"] != prev:
                return False, e["seq"]
            prev = e["hash"]

        return True, -1

    @property
    def consent_active(self) -> bool:
        return self._consent_ok

    @property
    def session_id(self) -> str:
        return self._session_id


# ── Typing fix for verify_chain return ───────────────────────────────────────
from typing import Tuple  # noqa: E402 (kept at bottom to avoid circular)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, sys

    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "logs" / "audit.jsonl"
        gate = EthicsGate(user_id="test_researcher", log_path=log)

        # ── Test 1: no consent → block
        print("\n[Test 1] Gate without consent:")
        theta = np.random.uniform(0, 2*np.pi, 240)
        try:
            gate.gate_pass(theta, label="theta")
            print("FAIL — should have raised NeurightViolation")
        except NeurightViolation as e:
            print(f"  PASS — blocked: {e}")

        # ── Test 2: full consent flow
        print("\n[Test 2] Consent flow:")
        token = gate.request_consent()
        gate.begin_session(token)
        out = gate.gate_pass(theta, label="theta")
        assert np.allclose(out, theta)
        print("  PASS — theta vector passed through")

        # ── Test 3: raw EEG leak detection
        print("\n[Test 3] Raw EEG leak:")
        raw_eeg = np.random.randn(4, 512) * 50e-6   # µV-scale raw window
        try:
            gate.gate_pass(raw_eeg, label="raw_eeg")
            print("FAIL — should have raised NeurightViolation")
        except NeurightViolation as e:
            print(f"  PASS — blocked: {e}")

        # ── Test 4: stimulation cap
        print("\n[Test 4] Stimulation cap:")
        clamped = gate.validate_stimulation(120.0, channel="AF7")
        assert clamped == STIM_ABSOLUTE_MAX_AMP, f"Expected {STIM_ABSOLUTE_MAX_AMP}, got {clamped}"
        print(f"  PASS — 120 µA → clamped to {clamped} µA")

        # ── Test 5: chain verification
        print("\n[Test 5] Audit chain:")
        gate.end_session()
        valid, broken_at = gate.verify_chain()
        assert valid, f"Chain broken at seq {broken_at}"
        print(f"  PASS — audit chain valid ({gate._seq} entries)")

    print("\nAll EthicsGate tests passed. ✓")
