#!/usr/bin/env python3
"""
calibration_wizard.py — khaos-core Baseline Calibration Wizard
==============================================================

Guides the user through a 5-phase "boring handshake" to establish
personal EEG landmarks.  The landmarks are SHA-256 signed and saved
to vault/landmarks.json for use by the quantum bridge (mirror_bridge.py).

5 Calibration Phases
--------------------
  1. Zen / Rest          — eyes closed, no task (μ-band dominant)
  2. Motor Alpha         — relaxed eyes open, idle hands (α/μ mixed)
  3. Focus / Beta        — mental arithmetic or active reading (β dominant)
  4. Motor Imagery       — imagined right-hand movement (contralateral β)
  5. Flow                — engaged with a low-pressure task (β + high conf)

Each phase:
  • Countdown (configurable, default 5 s) — user preparation
  • Collection window (default 10 s × 1000 Hz / 100 = 100 vectors)
  • Computes: mean & std of theta[0..11], confidence, entropy, bp_index
  • Signs the record with SHA-256 via Sovereignty-Monitor-compatible chain

Data sources (in order of priority):
  --khaos  PATH    Read JSON-lines from a running khaos_mirror process
  --lsl             Use pylsl inlet (requires LSL EEG source)
  (default)         Synthetic simulation (unit-test / offline calibration)

Usage:
  python3 calibration_wizard.py [options]

  -o, --output PATH   Vault JSON path  (default: ../vault/landmarks.json)
  --countdown N       Preparation countdown in seconds (default 5)
  --collect N         Collection window in seconds (default 10)
  --mock              Force synthetic mock mode (default if no hardware)
  --lsl               Use Lab Streaming Layer as data source
  --khaos PATH        JSON-line pipe from khaos_mirror executable

Sovereign principle:
  No raw EEG leaves this script.  Only statistical summaries (mean/std of
  derived theta angles) are written to the vault.  Raw waveforms are never
  stored.  Consistent with khaos-core ETHICS.md §I (Mental Privacy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

N_QUBITS     = 12          # NeuralPhaseVector.theta dimensionality
VAULT_VERSION = 2          # Incremented on schema changes

# Phase descriptors
PHASES: List[Dict] = [
    {
        "id":    "zen_rest",
        "label": "Zen / Rest",
        "icon":  "🧘",
        "color": "\033[36m",       # cyan
        "instructions": (
            "Close your eyes.\n"
            "  Let your mind wander without focus.\n"
            "  Rest your hands flat on your thighs.\n"
            "  Breathe slowly — in 4 counts, out 6 counts."
        ),
        # Expected theta profile: μ dominant → near 0 (|0⟩ state)
        "mock_theta_center": 0.35,
        "mock_theta_spread": 0.18,
        "mock_conf_center":  0.82,
        "mock_entropy":      0.28,
    },
    {
        "id":    "motor_alpha",
        "label": "Motor Alpha",
        "icon":  "🤲",
        "color": "\033[33m",       # yellow
        "instructions": (
            "Open your eyes, gaze softly at a fixed point.\n"
            "  Keep your hands relaxed and completely still.\n"
            "  Do not perform any mental task — just be present."
        ),
        "mock_theta_center": 0.90,
        "mock_theta_spread": 0.22,
        "mock_conf_center":  0.75,
        "mock_entropy":      0.38,
    },
    {
        "id":    "focus_beta",
        "label": "Focus / Beta",
        "icon":  "🧮",
        "color": "\033[35m",       # magenta
        "instructions": (
            "Perform serial subtraction: start at 300, subtract 7 repeatedly.\n"
            "  Speak the numbers silently (inner voice only).\n"
            "  Continue until the phase ends."
        ),
        "mock_theta_center": 2.20,
        "mock_theta_spread": 0.30,
        "mock_conf_center":  0.88,
        "mock_entropy":      0.55,
    },
    {
        "id":    "motor_imagery",
        "label": "Motor Imagery",
        "icon":  "🤜",
        "color": "\033[34m",       # blue
        "instructions": (
            "Imagine squeezing a ball tightly in your RIGHT hand.\n"
            "  Visualise every detail — texture, resistance, your fingers.\n"
            "  Repeat the imagined squeeze ~every 2 seconds."
        ),
        "mock_theta_center": 1.85,
        "mock_theta_spread": 0.35,
        "mock_conf_center":  0.80,
        "mock_entropy":      0.48,
    },
    {
        "id":    "flow",
        "label": "Flow State",
        "icon":  "🌊",
        "color": "\033[32m",       # green
        "instructions": (
            "Think of a task you genuinely enjoy — music, code, sketching.\n"
            "  Mentally step through it in detail.\n"
            "  Stay engaged but relaxed — no time pressure."
        ),
        "mock_theta_center": 1.55,
        "mock_theta_spread": 0.25,
        "mock_conf_center":  0.91,
        "mock_entropy":      0.44,
    },
]

# ANSI helpers
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"

# ── Terminal helpers ──────────────────────────────────────────────────────────

def _tty() -> bool:
    return sys.stdout.isatty()

def _print(msg: str = "", color: str = "") -> None:
    if color and _tty():
        print(f"{color}{msg}{RESET}")
    else:
        print(msg)

def _header() -> None:
    print()
    _print("╔══════════════════════════════════════════════════════════════╗", CYAN)
    _print("║   khaos-core  ·  Baseline Calibration Wizard                ║", CYAN)
    _print("║   5-phase EEG landmark acquisition + Sovereignty signing    ║", CYAN)
    _print("╚══════════════════════════════════════════════════════════════╝", CYAN)
    print()

def _phase_banner(idx: int, phase: Dict) -> None:
    color = phase["color"] if _tty() else ""
    icon  = phase["icon"] if _tty() else f"[{idx+1}]"
    line  = f"  Phase {idx+1}/5 — {icon}  {phase['label']}"
    print()
    _print("─" * 64, color)
    _print(line, BOLD if _tty() else "")
    _print("─" * 64, color)
    print()

def _countdown(seconds: int, label: str = "Prepare in") -> None:
    for s in range(seconds, 0, -1):
        if _tty():
            sys.stdout.write(f"\r  {label}: {BOLD}{s:2d}s{RESET}  ")
            sys.stdout.flush()
        else:
            print(f"  {label}: {s}s")
        time.sleep(1)
    if _tty():
        sys.stdout.write("\r  " + " " * 30 + "\r")
        sys.stdout.flush()

def _progress_bar(fraction: float, width: int = 40) -> str:
    filled = int(fraction * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction*100:.0f}%"

# ── Data sources ──────────────────────────────────────────────────────────────

def _mock_vector(phase: Dict, rng: np.random.Generator) -> Dict:
    """Generate a single synthetic NeuralPhaseVector for the given phase."""
    center  = phase["mock_theta_center"]
    spread  = phase["mock_theta_spread"]
    # Vary slightly across qubits (hemisphere asymmetry simulation)
    offsets = rng.normal(0, 0.05, N_QUBITS).astype(float)
    theta   = np.clip(
        rng.normal(center + offsets, spread / 3.0, N_QUBITS),
        0.0, math.pi
    ).tolist()
    conf    = float(np.clip(rng.normal(phase["mock_conf_center"], 0.05), 0.0, 1.0))
    entropy = float(np.clip(rng.normal(phase["mock_entropy"],     0.04), 0.0, 1.0))
    bp      = float(np.clip(rng.normal(0.10 + center * 0.04,     0.03), 0.0, 1.0))
    return {"theta": theta, "confidence": conf,
            "entropy_estimate": entropy, "bp_index": bp}


def collect_mock(phase: Dict, n_vectors: int,
                 countdown_s: int, rng: np.random.Generator) -> List[Dict]:
    """Collect synthetic vectors for one phase."""
    _countdown(countdown_s)
    vectors = []
    t0 = time.monotonic()
    for i in range(n_vectors):
        vectors.append(_mock_vector(phase, rng))
        if _tty():
            elapsed = time.monotonic() - t0
            frac    = (i + 1) / n_vectors
            bar     = _progress_bar(frac)
            sys.stdout.write(f"\r  Collecting {bar}  {i+1}/{n_vectors}")
            sys.stdout.flush()
        time.sleep(0.05)   # 20 Hz mock rate
    if _tty():
        print()
    return vectors


def collect_lsl(phase: Dict, n_vectors: int, countdown_s: int) -> List[Dict]:
    """Collect vectors from a Lab Streaming Layer inlet."""
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError:
        print("  [WARN] pylsl not installed — falling back to mock mode", file=sys.stderr)
        return collect_mock(phase, n_vectors, countdown_s, np.random.default_rng())

    print("  Searching for khaos-NeuralPhaseVector LSL stream …", end="", flush=True)
    streams = resolve_byprop("name", "khaos-NeuralPhaseVector", timeout=5.0)
    if not streams:
        print("\n  [WARN] No LSL stream found — falling back to mock mode", file=sys.stderr)
        return collect_mock(phase, n_vectors, countdown_s, np.random.default_rng())
    print(" found.")

    inlet = StreamInlet(streams[0])
    _countdown(countdown_s)
    vectors = []
    t0 = time.monotonic()

    while len(vectors) < n_vectors:
        sample, _ = inlet.pull_sample(timeout=2.0)
        if sample is None:
            break
        # Expected layout: [theta0..11, confidence, entropy, bp_index]
        if len(sample) < N_QUBITS + 3:
            continue
        vec = {
            "theta":             [float(x) for x in sample[:N_QUBITS]],
            "confidence":        float(sample[N_QUBITS]),
            "entropy_estimate":  float(sample[N_QUBITS + 1]),
            "bp_index":          float(sample[N_QUBITS + 2]),
        }
        vectors.append(vec)
        if _tty():
            frac = len(vectors) / n_vectors
            bar  = _progress_bar(frac)
            sys.stdout.write(f"\r  Collecting {bar}  {len(vectors)}/{n_vectors}")
            sys.stdout.flush()

    if _tty():
        print()
    return vectors


def collect_khaos(phase: Dict, n_vectors: int,
                  countdown_s: int, proc: subprocess.Popen) -> List[Dict]:
    """Read JSON-line NeuralPhaseVectors from a running khaos_mirror process."""
    _countdown(countdown_s)
    vectors = []
    assert proc.stdout is not None
    while len(vectors) < n_vectors:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            obj = json.loads(line.strip())
            if "theta" in obj and len(obj["theta"]) == N_QUBITS:
                vectors.append({
                    "theta":            [float(x) for x in obj["theta"]],
                    "confidence":       float(obj.get("confidence",      0.0)),
                    "entropy_estimate": float(obj.get("entropy_estimate", 0.0)),
                    "bp_index":         float(obj.get("bp_index",         0.0)),
                })
                if _tty():
                    frac = len(vectors) / n_vectors
                    bar  = _progress_bar(frac)
                    sys.stdout.write(
                        f"\r  Collecting {bar}  {len(vectors)}/{n_vectors}")
                    sys.stdout.flush()
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    if _tty():
        print()
    return vectors

# ── Statistics ────────────────────────────────────────────────────────────────

def compute_landmark(vectors: List[Dict], phase_id: str) -> Dict:
    """Reduce a list of raw vectors to a signed landmark record."""
    if not vectors:
        raise ValueError(f"No vectors collected for phase '{phase_id}'")

    thetas     = np.array([v["theta"]             for v in vectors], dtype=float)
    confs      = np.array([v["confidence"]         for v in vectors], dtype=float)
    entropies  = np.array([v["entropy_estimate"]   for v in vectors], dtype=float)
    bps        = np.array([v["bp_index"]           for v in vectors], dtype=float)

    theta_mean = thetas.mean(axis=0).tolist()
    theta_std  = thetas.std(axis=0).tolist()

    # Circular mean for angles (handles 0/π wrap correctly)
    circ_mean = [
        float(np.arctan2(
            np.sin(thetas[:, q]).mean(),
            np.cos(thetas[:, q]).mean()
        ) % (2 * math.pi))
        for q in range(N_QUBITS)
    ]

    return {
        "phase_id":         phase_id,
        "n_samples":        len(vectors),
        "theta_mean":       theta_mean,
        "theta_std":        theta_std,
        "theta_circ_mean":  circ_mean,      # circular mean (angle-safe)
        "confidence_mean":  float(confs.mean()),
        "confidence_std":   float(confs.std()),
        "entropy_mean":     float(entropies.mean()),
        "bp_index_mean":    float(bps.mean()),
    }

# ── SHA-256 chain signing (Sovereignty Monitor compatible) ────────────────────

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_signed_vault(landmarks: Dict[str, Dict],
                       calibration_ts: str,
                       git_hash: str) -> Dict:
    """
    Construct the signed landmark vault.

    Chain structure (mirrors sovereignty_monitor.cpp):
      entry_hash = sha256( JSON(landmark_record) )
      chain starts with genesis hash of zeros.
      Each phase entry hashes its own data + prev_hash.
      Final vault_hash covers all phase hashes.

    No private key is used — this is an integrity chain, not authentication.
    For production deployments, add CRYSTALS-Dilithium signing over vault_hash.
    """
    GENESIS = "0" * 64   # 32 zero bytes, hex-encoded

    chain: List[Dict] = []
    prev_hash = GENESIS

    for phase_id, record in landmarks.items():
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        entry_hash = _sha256_hex(payload)
        signed_entry = {
            "phase_id":   phase_id,
            "landmark":   record,
            "entry_hash": entry_hash,
            "prev_hash":  prev_hash,
        }
        chain.append(signed_entry)
        prev_hash = entry_hash

    # Vault-level hash covers all chain entries
    chain_blob = json.dumps(chain, sort_keys=True, separators=(",", ":")).encode()
    vault_hash = _sha256_hex(
        chain_blob + calibration_ts.encode() + git_hash.encode()
    )

    return {
        "version":            VAULT_VERSION,
        "calibration_timestamp": calibration_ts,
        "git_hash":           git_hash,
        "n_phases":           len(chain),
        "chain":              chain,
        "vault_hash":         vault_hash,
        "integrity_note":     (
            "SHA-256 chain — no raw EEG stored (ETHICS.md §I: Mental Privacy). "
            "Vault contains only statistical summaries of derived theta angles."
        ),
    }


def verify_vault(vault: Dict) -> bool:
    """Re-compute hashes and verify chain integrity. Returns True if intact."""
    chain = vault.get("chain", [])
    if not chain:
        return False

    prev = "0" * 64
    for entry in chain:
        payload = json.dumps(
            entry["landmark"], sort_keys=True, separators=(",", ":")).encode()
        computed = _sha256_hex(payload)
        if computed != entry["entry_hash"]:
            return False
        if entry["prev_hash"] != prev:
            return False
        prev = entry["entry_hash"]

    # Recompute vault_hash
    chain_blob = json.dumps(
        [{k: v for k, v in e.items()} for e in chain],
        sort_keys=True, separators=(",", ":")).encode()
    ts       = vault.get("calibration_timestamp", "")
    git_hash = vault.get("git_hash", "")
    computed_vault = _sha256_hex(chain_blob + ts.encode() + git_hash.encode())
    return computed_vault == vault.get("vault_hash", "")

# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(vault: Dict) -> None:
    print()
    _print("═" * 64, CYAN)
    _print("  Calibration Summary", BOLD if _tty() else "")
    _print("═" * 64, CYAN)
    print()
    for entry in vault["chain"]:
        lm    = entry["landmark"]
        pid   = lm["phase_id"]
        phase = next((p for p in PHASES if p["id"] == pid), None)
        label = phase["label"] if phase else pid
        icon  = phase["icon"]  if phase and _tty() else ""
        color = phase["color"] if phase and _tty() else ""

        theta_m = lm["theta_mean"]
        q0_mean = f"{theta_m[0]:.3f}"
        q_range = f"{min(theta_m):.3f} – {max(theta_m):.3f}"

        _print(f"  {icon}  {label}", color)
        print(f"       θ̄[0]    = {q0_mean} rad")
        print(f"       θ range = {q_range} rad")
        print(f"       conf    = {lm['confidence_mean']:.3f}  "
              f"entropy = {lm['entropy_mean']:.3f}")
        print(f"       n_samp  = {lm['n_samples']}")
        print()

    ok = verify_vault(vault)
    _print(f"  Chain integrity: {'✓ VERIFIED' if ok else '✗ BROKEN'}",
           GREEN if ok else RED)
    print(f"  Vault hash: {vault['vault_hash'][:32]}…")
    print()

# ── Git hash ──────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="khaos-core 5-phase EEG baseline calibration wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default=None,
                   help="Vault JSON output path (default: ../vault/landmarks.json)")
    p.add_argument("--countdown", type=int, default=5,
                   help="Preparation countdown per phase, seconds (default: 5)")
    p.add_argument("--collect", type=int, default=10,
                   help="Collection window per phase, seconds (default: 10)")
    p.add_argument("--mock", action="store_true",
                   help="Force mock mode (ignore hardware)")
    p.add_argument("--lsl", action="store_true",
                   help="Read from Lab Streaming Layer inlet")
    p.add_argument("--khaos", metavar="PATH",
                   help="Path to khaos_mirror binary (read JSON-line output)")
    p.add_argument("--verify", metavar="VAULT",
                   help="Verify an existing vault and exit")
    p.add_argument("--phases", metavar="IDS",
                   help="Comma-separated subset of phase IDs to run")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Verify-only mode ────────────────────────────────────────────────────
    if args.verify:
        vault_path = Path(args.verify)
        if not vault_path.exists():
            print(f"[error] Vault not found: {vault_path}", file=sys.stderr)
            return 1
        vault = json.loads(vault_path.read_text())
        ok = verify_vault(vault)
        _print(f"Vault: {vault_path}", BOLD if _tty() else "")
        _print(f"Integrity: {'✓ VERIFIED' if ok else '✗ BROKEN'}",
               GREEN if ok else RED)
        print(f"Hash: {vault.get('vault_hash','?')}")
        return 0 if ok else 1

    # ── Output path ─────────────────────────────────────────────────────────
    if args.output:
        vault_path = Path(args.output)
    else:
        repo_root  = Path(__file__).parent.parent
        vault_path = repo_root / "vault" / "landmarks.json"
    vault_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase filter ────────────────────────────────────────────────────────
    phases_to_run = PHASES
    if args.phases:
        ids = {s.strip() for s in args.phases.split(",")}
        phases_to_run = [p for p in PHASES if p["id"] in ids]
        if not phases_to_run:
            print(f"[error] No matching phase IDs: {args.phases}", file=sys.stderr)
            return 1

    # ── Determine data source ───────────────────────────────────────────────
    use_mock   = args.mock
    use_lsl    = args.lsl and not args.mock
    khaos_proc: Optional[subprocess.Popen] = None

    if args.khaos and not args.mock:
        binary = Path(args.khaos)
        if not binary.exists():
            print(f"[warn] khaos_mirror not found at {binary} — using mock mode",
                  file=sys.stderr)
            use_mock = True
        else:
            khaos_proc = subprocess.Popen(
                [str(binary), "--calibrate-stream"],
                stdout=subprocess.PIPE, text=True)
    elif not use_lsl:
        use_mock = True

    # Vectors per phase
    vectors_per_phase = args.collect * 20   # 20 vectors/s (mock/LSL rate)

    rng = np.random.default_rng(seed=int(time.time()))
    git_hash = _git_hash()

    _header()

    if use_mock:
        _print("  Mode: SYNTHETIC SIMULATION (no hardware required)", YELLOW)
    elif use_lsl:
        _print("  Mode: Lab Streaming Layer", GREEN)
    elif khaos_proc:
        _print(f"  Mode: khaos_mirror — {args.khaos}", GREEN)

    print(f"  Git hash   : {git_hash}")
    print(f"  Phases     : {len(phases_to_run)}")
    print(f"  Countdown  : {args.countdown} s")
    print(f"  Collection : {args.collect} s  (~{vectors_per_phase} vectors)")
    print(f"  Output     : {vault_path}")
    print()
    print("  Press ENTER to begin, or Ctrl-C to abort.")
    try:
        input("  > ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Calibration aborted.")
        return 0

    # ── Phase loop ─────────────────────────────────────────────────────────
    calibration_ts = datetime.now(timezone.utc).isoformat()
    landmarks: Dict[str, Dict] = {}

    for idx, phase in enumerate(phases_to_run):
        _phase_banner(idx, phase)

        # Instructions
        for line in phase["instructions"].split("\n"):
            print(f"  {line.strip()}")
        print()

        # Collect
        if use_mock:
            vectors = collect_mock(phase, vectors_per_phase, args.countdown, rng)
        elif use_lsl:
            vectors = collect_lsl(phase, vectors_per_phase, args.countdown)
        elif khaos_proc:
            vectors = collect_khaos(phase, vectors_per_phase, args.countdown, khaos_proc)
        else:
            vectors = collect_mock(phase, vectors_per_phase, args.countdown, rng)

        if not vectors:
            _print(f"  [WARN] No data collected for phase '{phase['id']}'", YELLOW)
            continue

        lm = compute_landmark(vectors, phase["id"])
        landmarks[phase["id"]] = lm

        _print(f"  ✓ Collected {len(vectors)} vectors  "
               f"θ̄[0]={lm['theta_mean'][0]:.3f}  "
               f"conf={lm['confidence_mean']:.3f}", GREEN)

    # ── Shutdown khaos_mirror ───────────────────────────────────────────────
    if khaos_proc:
        khaos_proc.terminate()
        khaos_proc.wait(timeout=5)

    if not landmarks:
        print("[error] No landmarks collected — aborting.", file=sys.stderr)
        return 1

    # ── Build and sign vault ────────────────────────────────────────────────
    vault = build_signed_vault(landmarks, calibration_ts, git_hash)

    # ── Write vault ─────────────────────────────────────────────────────────
    vault_path.write_text(
        json.dumps(vault, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # ── Print summary ───────────────────────────────────────────────────────
    print_summary(vault)

    _print(f"  Vault written → {vault_path}", GREEN)
    print()
    print("  The landmark vault is ready.  Load it in mirror_bridge.py via:")
    print(f"    LandmarkLibrary(vault_path='{vault_path}')")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
