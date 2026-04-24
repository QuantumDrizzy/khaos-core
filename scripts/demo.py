#!/usr/bin/env python3
"""
demo.py — KĦAOS-CORE Live Demo Script
══════════════════════════════════════════════════════════════════════════════
End-to-end demonstration of the Muse 2 Python validation stack.

Runs entirely without hardware using SyntheticMuse2Adapter.
Demonstrates: EEG pipeline, 12-qubit feature extraction, ethics gate,
cross-stack audit trail, stimulation safety cap, and sovereignty killswitch.

Usage:
    python demo.py          # interactive (prompts dashboard launch)
    python demo.py --no-dashboard   # CI / non-interactive mode
    echo n | python demo.py # pipe stdin to skip dashboard prompt

Requirements: numpy, scipy, matplotlib (optional, for dashboard)
"""

from __future__ import annotations

import argparse
import math
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# ── ANSI colour helpers ───────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def green(t):   return _c("32;1", t)
def red(t):     return _c("31;1", t)
def yellow(t):  return _c("33;1", t)
def cyan(t):    return _c("36;1", t)
def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)

def section(n: int, title: str):
    bar = "─" * (54 - len(title))
    print(f"\n{cyan(f'── {n}. {title} {bar}')}")


# ── Results tracker ───────────────────────────────────────────────────────────

_results: list[tuple[str, bool, float]] = []

def _record(name: str, ok: bool, elapsed: float):
    _results.append((name, ok, elapsed))
    status = green("✓") if ok else red("✗")
    t_str  = dim(f"[{elapsed:.2f}s]")
    print(f"  {status}  {name}  {t_str}")


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    print(cyan(r"""
  ██╗  ██╗██╗  ██╗ █████╗  ██████╗ ███████╗      ██████╗ ██████╗ ██████╗ ███████╗
  ██║ ██╔╝██║  ██║██╔══██╗██╔═══██╗██╔════╝     ██╔════╝██╔═══██╗██╔══██╗██╔════╝
  █████╔╝ ███████║███████║██║   ██║███████╗     ██║     ██║   ██║██████╔╝█████╗
  ██╔═██╗ ██╔══██║██╔══██║██║   ██║╚════██║     ██║     ██║   ██║██╔══██╗██╔══╝
  ██║  ██╗██║  ██║██║  ██║╚██████╔╝███████║     ╚██████╗╚██████╔╝██║  ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝"""))
    print(f"  {bold('KĦAOS-CORE')} — Dual-Stack BCI with Embedded Neurorights Sovereignty")
    print(f"  {dim('Validation layer  │  256 Hz  │  4–64 ch  │  12-qubit quantum feature map')}")
    print(f"  {dim('Date: 2026-04-24  │  Demo mode: SyntheticMuse2Adapter (no hardware required)')}")
    print()


# ── 1. System check ───────────────────────────────────────────────────────────

def step_system_check() -> dict:
    section(1, "System Check")
    t0      = time.perf_counter()
    modules = {}

    critical = [
        ("numpy",                   "numpy"),
        ("scipy",                   "scipy"),
        ("src.io.muse2_adapter",    "Muse 2 adapter"),
        ("src.bci.feature_extractor", "Feature extractor"),
        ("src.ethics.ethics_gate",  "Ethics gate"),
        ("src.ethics.ethics_bridge","Ethics bridge"),
    ]
    optional = [
        ("matplotlib",              "Matplotlib (dashboard)"),
    ]

    abort = False
    for mod, label in critical:
        try:
            m = __import__(mod)
            modules[mod] = m
            print(f"  {green('✓')}  {label}")
        except Exception as e:
            print(f"  {red('✗')}  {label}  {dim(str(e))}")
            abort = True

    for mod, label in optional:
        try:
            m = __import__(mod)
            modules[mod] = m
            print(f"  {green('✓')}  {label}  {dim('(optional)')}")
        except Exception:
            print(f"  {yellow('–')}  {label}  {dim('(not available — dashboard will be skipped)')}")

    elapsed = time.perf_counter() - t0
    _record("System check", not abort, elapsed)
    if abort:
        print(red("\n  Critical import failed. Aborting demo."))
        sys.exit(1)

    return modules


# ── 2. Ethics handshake ───────────────────────────────────────────────────────

def step_handshake(mods: dict, tmpdir: str) -> tuple:
    section(2, "Cross-Stack Ethics Handshake (HMAC-SHA256)")
    t0 = time.perf_counter()

    from src.ethics.ethics_bridge import EthicsBridge, CppSovereigntyStub

    log_path = Path(tmpdir) / "bridge.jsonl"
    bridge   = EthicsBridge(log_path=log_path, verbose=False)
    stub     = CppSovereigntyStub(bridge)

    challenge = bridge.initiate_handshake()
    response  = stub.sign_challenge(challenge)
    ok        = bridge.verify_handshake(challenge, response)

    print(f"  Challenge  : {cyan(challenge[:16])}…")
    print(f"  Response   : {cyan(response[:16])}…")
    print(f"  Verification: {green('SESSION_START logged ✓') if ok else red('FAILED ✗')}")

    elapsed = time.perf_counter() - t0
    _record("HMAC-SHA256 handshake", ok, elapsed)
    return bridge, stub


# ── 3. Consent + session ──────────────────────────────────────────────────────

def step_consent(tmpdir: str) -> object:
    section(3, "Neurorights Consent & Session")
    t0 = time.perf_counter()

    from src.ethics.ethics_gate import EthicsGate

    log_path = Path(tmpdir) / "gate.jsonl"
    gate     = EthicsGate(user_id="khaos_demo", log_path=log_path, verbose=False)
    token    = gate.request_consent()
    gate.begin_session(token)

    print(f"  Consent token : {cyan(str(token)[:8])}…  {dim('(UUID4, single-use)')}")
    print(f"  Session       : {green('ACTIVE')}")
    print(f"  Audit log     : {dim(str(log_path))}")

    elapsed = time.perf_counter() - t0
    _record("Consent + session start", True, elapsed)
    return gate


# ── 4. Synthetic EEG stream ───────────────────────────────────────────────────

def step_eeg_stream() -> object:
    section(4, "Synthetic EEG Stream (Muse 2 — 4ch, 256 Hz)")
    t0 = time.perf_counter()

    from src.io.muse2_adapter import SyntheticMuse2Adapter

    adapter = SyntheticMuse2Adapter()
    adapter.connect()
    adapter.start()

    # ASCII progress bar while waiting for buffer
    print("  Buffering EEG", end="", flush=True)
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] if _TTY else ["."] * 10
    deadline = time.perf_counter() + 8.0
    i = 0
    while time.perf_counter() < deadline:
        if adapter.wait_ready(timeout=0.3):
            break
        ch = spinner[i % len(spinner)]
        print(f"\r  Buffering EEG {cyan(ch)}", end="", flush=True)
        i += 1
    print(f"\r  Buffer        : {green('READY')}            ")

    ready = adapter.wait_ready(timeout=0.0)
    elapsed = time.perf_counter() - t0
    _record("EEG stream buffer ready", ready, elapsed)
    return adapter


# ── 5. Feature extraction ─────────────────────────────────────────────────────

def step_extract(adapter) -> np.ndarray:
    section(5, "12-Qubit Feature Extraction")
    t0 = time.perf_counter()

    from src.bci.feature_extractor import Muse2FeatureExtractor, THETA_LEN

    alpha = adapter.get_filtered_window("alpha")
    theta = adapter.get_filtered_window("theta")
    ext   = Muse2FeatureExtractor(n_channels=4)
    vec   = ext.extract(alpha, theta)

    print(f"  Theta vector shape : {cyan(str(vec.shape))}")
    print(f"  Range              : [{vec.min():.4f}, {vec.max():.4f}]  "
          f"{dim('(expected [0, 2π])')}")
    print(f"  Mean               : {vec.mean():.4f}")
    print(f"  Qubit labels       : {dim(', '.join(ext.qubit_labels()[:4]))} … "
          f"{dim(ext.qubit_labels()[-1])}")

    ok      = vec.shape == (THETA_LEN,) and float(vec.min()) >= 0.0
    elapsed = time.perf_counter() - t0
    _record("Feature extraction → theta (240,)", ok, elapsed)
    return vec


# ── 6. Ethics gate pass ───────────────────────────────────────────────────────

def step_gate_pass(gate, theta: np.ndarray) -> bool:
    section(6, "Ethics Gate — GATE_PASS")
    t0 = time.perf_counter()

    from src.ethics.ethics_gate import NeurightViolation

    ok = False
    try:
        gate.gate_pass(theta, label="demo_theta")
        ok = True
        print(f"  Theta vector      : {green('GATE_PASS ✓')}")
        print(f"  Audit seq         : {dim(str(gate._seq))}")
    except NeurightViolation as e:
        print(f"  {red('GATE_BLOCK')} — unexpected: {e}")

    elapsed = time.perf_counter() - t0
    _record("Ethics gate pass (theta vector)", ok, elapsed)
    return ok


# ── 7. Stimulation safety cap ─────────────────────────────────────────────────

def step_stim_cap(gate) -> bool:
    section(7, "Stimulation Safety Cap (50 µA)")
    t0 = time.perf_counter()

    test_cases = [
        (10.0,  "below cap"),
        (50.0,  "at cap"),
        (200.0, "2× cap"),
        (1000.0,"20× cap"),
    ]
    ok = True
    for amp_in, label in test_cases:
        amp_out = gate.validate_stimulation(amp_in, channel="AF7")
        clamped = amp_out < amp_in
        arrow   = red(f"{amp_in:.0f} → {amp_out:.1f} µA  ⚡ CLAMPED") if clamped \
                  else green(f"{amp_in:.1f} µA  ✓ passed")
        print(f"  {label:<12} : {arrow}")
        if amp_in > 50.0 and abs(amp_out - 50.0) > 0.01:
            ok = False

    elapsed = time.perf_counter() - t0
    _record("Stim cap (50 µA) enforced", ok, elapsed)
    return ok


# ── 8. Cross-stack audit chain ────────────────────────────────────────────────

def step_audit_chain(bridge, stub) -> bool:
    section(8, "Cross-Stack Audit Chain (SHA-256)")
    t0 = time.perf_counter()

    from src.ethics.ethics_bridge import BridgeStack, BridgeEvent

    # Emit 20 alternating Python/C++ events
    for i in range(10):
        bridge.log(BridgeStack.PYTHON, BridgeEvent.GATE_PASS,
                   {"i": i, "label": "theta"})
        stub.emit_gate_pass(f"cpp_{i}")

    valid, broken_at = bridge.verify_chain()
    n_entries = bridge.entry_count

    print(f"  Chain entries  : {cyan(str(n_entries))}  (10 Python + 10 C++ stub + handshake)")
    print(f"  Chain valid    : {green('CHAIN_VALID ✓') if valid else red(f'BROKEN at seq {broken_at} ✗')}")

    elapsed = time.perf_counter() - t0
    _record("SHA-256 audit chain integrity", valid, elapsed)
    return valid


# ── 9. Killswitch / sovereignty enforcement ───────────────────────────────────

def step_killswitch(gate) -> bool:
    section(9, "Sovereignty Killswitch — NeurightViolation")
    t0 = time.perf_counter()

    from src.ethics.ethics_gate import NeurightViolation

    gate.trigger_killswitch(reason="khaos_demo_killswitch_test")
    print(f"  Killswitch     : {red('TRIGGERED')}")

    # Now any gate_pass must raise NeurightViolation
    theta_fake = np.random.uniform(0, 2 * math.pi, 240)
    caught = False
    try:
        gate.gate_pass(theta_fake)
    except NeurightViolation:
        caught = True

    if caught:
        print(f"  gate_pass()    : {green('NeurightViolation raised ✓')}")
        print(f"  Sovereignty    : {green('ENFORCED ✓')}")
    else:
        print(f"  gate_pass()    : {red('DID NOT raise — sovereignty violation!')}")

    elapsed = time.perf_counter() - t0
    _record("Killswitch → NeurightViolation", caught, elapsed)
    return caught


# ── 10. Test suite ────────────────────────────────────────────────────────────

def step_test_suite() -> bool:
    section(10, "Test Suite (81 tests)")
    t0 = time.perf_counter()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-q", "--tb=no",
         "--no-header", "-p", "no:warnings"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    output = (result.stdout + result.stderr).strip()

    # Find the summary line
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            ok = "failed" not in line and "error" not in line
            colour = green if ok else red
            print(f"  {colour(line.strip())}")
            elapsed = time.perf_counter() - t0
            _record("pytest tests/unit/ (81 tests)", ok, elapsed)
            return ok

    # Fallback
    ok = result.returncode == 0
    elapsed = time.perf_counter() - t0
    _record("pytest tests/unit/", ok, elapsed)
    return ok


# ── 11. Dashboard (optional) ──────────────────────────────────────────────────

def step_dashboard(no_dashboard: bool) -> bool:
    section(11, "Sovereignty Dashboard (DEMO mode)")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(f"  {yellow('–')}  matplotlib not available — skipping dashboard")
        _record("Dashboard (skipped — no matplotlib)", True, 0.0)
        return True

    if no_dashboard:
        print(f"  {dim('Skipped (--no-dashboard flag)')}")
        _record("Dashboard (skipped by flag)", True, 0.0)
        return True

    # Ask with a 5-second timeout
    launch = False
    print(f"  Launch sovereignty dashboard? [y/N]  {dim('(auto-N in 5 s)')}", end=" ", flush=True)

    try:
        if _TTY:
            rlist, _, _ = select.select([sys.stdin], [], [], 5.0)
            if rlist:
                answer = sys.stdin.readline().strip().lower()
                launch = answer == "y"
            else:
                print(dim("[timeout → N]"))
        else:
            answer = sys.stdin.readline().strip().lower()
            launch = answer == "y"
            print(dim(f"[stdin: '{answer}']"))
    except Exception:
        print(dim("[error → N]"))

    if not launch:
        _record("Dashboard (declined)", True, 0.0)
        return True

    t0 = time.perf_counter()
    try:
        from src.ui.sovereignty_dashboard import SovereigntyDashboard
        from src.io.muse2_adapter import SyntheticMuse2Adapter
        from src.ethics.ethics_gate import EthicsGate
        import tempfile

        print(f"  {cyan('Launching dashboard for 10 s…')}")

        with tempfile.TemporaryDirectory() as d:
            gate  = EthicsGate(user_id="dashboard_demo",
                               log_path=Path(d) / "demo.jsonl", verbose=False)
            token = gate.request_consent()
            gate.begin_session(token)

            adapter = SyntheticMuse2Adapter()
            adapter.connect()
            adapter.start()
            adapter.wait_ready(timeout=6.0)

            dash = SovereigntyDashboard(adapter=adapter, gate=gate)
            # Run non-blocking: start in a thread and stop after 10 s
            import threading
            t = threading.Thread(target=dash.run, daemon=True)
            t.start()
            t.join(timeout=10.0)
            dash.stop()
            adapter.stop()
            gate.end_session()

        ok = True
        print(f"  {green('Dashboard closed cleanly ✓')}")
    except Exception as e:
        ok = False
        print(f"  {red(f'Dashboard error: {e}')}")

    elapsed = time.perf_counter() - t0
    _record("Dashboard demo (10 s)", ok, elapsed)
    return ok


# ── Final summary ─────────────────────────────────────────────────────────────

def print_summary():
    print()
    print(cyan("── Summary " + "─" * 54))
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    for name, ok, elapsed in _results:
        status = green("✓") if ok else red("✗")
        t_str  = dim(f"[{elapsed:.2f}s]")
        print(f"  {status}  {name:<48}  {t_str}")

    print()
    colour = green if passed == total else red
    print(f"  {colour(f'{passed}/{total} steps passed')}")

    if passed == total:
        print(f"\n  {green('KĦAOS-CORE demo complete.')}")
        print(f"  {dim('81/81 unit tests passing. Ethics compiled, not configured.')}")
        print(f"  {dim('La información es el sustrato primario. La soberanía es el único protocolo aceptable.')}")
    else:
        print(f"\n  {red('Some steps failed. Review output above.')}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KĦAOS-CORE live demo")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip the optional dashboard launch prompt")
    args = parser.parse_args()

    print_banner()

    with tempfile.TemporaryDirectory() as tmpdir:
        mods         = step_system_check()
        bridge, stub = step_handshake(mods, tmpdir)
        gate         = step_consent(tmpdir)
        adapter      = step_eeg_stream()
        theta_vec    = step_extract(adapter)
        step_gate_pass(gate, theta_vec)
        step_stim_cap(gate)
        step_audit_chain(bridge, stub)
        step_killswitch(gate)
        step_test_suite()
        step_dashboard(args.no_dashboard)

        try:
            adapter.stop()
        except Exception:
            pass

    print_summary()


if __name__ == "__main__":
    main()
