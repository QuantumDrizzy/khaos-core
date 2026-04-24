#!/usr/bin/env python3
"""
validate_celegans.py — KĦAOS-CORE Functional Fidelity Validator (Gap 1)
══════════════════════════════════════════════════════════════════════════════
Prototype implementation of the experiment proposed in Section 10 of the
KĦAOS-CORE Technical Paper v1.1:

  "Future Horizons: Functional Fidelity Benchmarking in Whole-Brain Emulation"

Validates whether a computational emulation of a neural circuit (e.g., OpenWorm)
produces a 12-component functional fingerprint that matches the biological ground
truth (e.g., Kato et al. 2015 calcium-imaging data) at Pearson r > 0.95.

Pipeline
────────
  1. Load biotic data  (real C. elegans: Kato 2015 or mock)
  2. Load simulated data (OpenWorm or mock)
  3. Fit SpatialEmbedding (SVD) on biotic data → 12 dominant spatial modes
  4. Project both datasets through the shared SVD basis
  5. Extract 12-component fingerprint per dataset (band powers, asymmetry, engagement)
  6. Compute Pearson r across 12 effective dimensions
  7. Report: PASS (r > 0.95) or FAIL with component-level divergence map

Band configuration (C. elegans — re-parametrised from mammalian EEG defaults)
───────────────────────────────────────────────────────────────────────────────
  FS       = 4.0 Hz  (Kato 2015 calcium-imaging sample rate)
  SLOW_BAND = (0.05, 0.3) Hz  — locomotion dynamics, deep attractor transitions
  FAST_BAND = (0.3,  1.0) Hz  — state-switch transients, omega-turn signatures

References
──────────
  Kato, S. et al. (2015). Global brain dynamics embed the motor command
    sequence of C. elegans. Cell 163(3):656–669.
  White, J.G. et al. (1986). The structure of the nervous system of the
    nematode C. elegans. Phil. Trans. R. Soc. Lond. B 314:1–340.
  OpenWorm Project (2014–present). openworm.org

Usage
─────
  # Mock data (no download required)
  python scripts/validate_celegans.py

  # High-fidelity mock (emulation closer to biology — expect r > 0.95)
  python scripts/validate_celegans.py --high-fidelity

  # Real data (provide paths once downloaded from CRCNS)
  python scripts/validate_celegans.py \\
      --biotic   data/celegans/kato2015_traces.npy \\
      --simulated data/celegans/openworm_traces.npy

  # Verbose: show per-component breakdown
  python scripts/validate_celegans.py --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, welch
from scipy.stats import pearsonr

# ── Repo path ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.bci.feature_extractor import SpatialEmbedding

# ── Experiment configuration ──────────────────────────────────────────────────
FS           = 4.0          # Hz — Kato 2015 calcium-imaging sample rate
N_CHANNELS   = 302          # C. elegans neuron count (White et al. 1986)
N_COMPONENTS = 12           # SVD components = KĦAOS-CORE qubit count
SLOW_BAND    = (0.05, 0.3)  # Hz — locomotion / deep attractor dynamics
FAST_BAND    = (0.3,  1.0)  # Hz — state-switch transients
WINDOW_S     = 120.0        # seconds — calibration / extraction window
N_SAMPLES    = int(FS * WINDOW_S)   # 480 samples at 4 Hz
FIDELITY_THR = 0.95         # Pearson r threshold for emulation acceptance
EPS          = 1e-30

# ANSI colours
_GRN  = "\033[92m"
_RED  = "\033[91m"
_YEL  = "\033[93m"
_CYN  = "\033[96m"
_BLD  = "\033[1m"
_RST  = "\033[0m"

def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RST}" if sys.stdout.isatty() else text


# ── Signal utilities ──────────────────────────────────────────────────────────

def _pink_noise(n_channels: int, n_samples: int,
                fs: float = FS, seed: Optional[int] = None) -> np.ndarray:
    """Generate 1/f (pink) noise — shape (n_channels, n_samples).

    Pink noise is a better model for neural population activity than white
    noise: power spectral density ∝ 1/f, matching the scale-free dynamics
    observed in C. elegans calcium fluorescence traces.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_channels, n_samples))
    # Shape spectrum: multiply FFT by 1/√f
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    freqs[0] = freqs[1]          # avoid division by zero at DC
    shaping = 1.0 / np.sqrt(freqs)
    shaping /= shaping.mean()    # normalise to unit mean gain
    spectrum = np.fft.rfft(white, axis=1) * shaping[np.newaxis, :]
    return np.fft.irfft(spectrum, n=n_samples, axis=1)


def _bandpass(data: np.ndarray, low: float, high: float,
              fs: float = FS, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter applied per channel."""
    nyq = fs / 2.0
    lo  = max(low  / nyq, 1e-4)
    hi  = min(high / nyq, 1.0 - 1e-4)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, data, axis=1)


def _band_power(signal: np.ndarray, low: float, high: float,
                fs: float = FS) -> float:
    """Welch PSD integrated over [low, high] Hz."""
    nperseg = min(len(signal), max(16, int(fs * 30)))   # ≥30 s window
    freqs, pxx = welch(signal, fs=fs, nperseg=nperseg,
                       window="hann", average="mean")
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return EPS
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(pxx[mask], freqs[mask]))


# ── Mock data generator ───────────────────────────────────────────────────────

def load_mock_data(
        high_fidelity: bool = False,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic biotic and simulated datasets.

    Both datasets share a low-rank spatial structure (3 dominant modes that
    drive the top SVD components) plus independent pink-noise contributions
    that model unmodelled dynamics or emulation error.

    Parameters
    ----------
    high_fidelity : bool
        If True, the simulated data matches the biotic data more closely
        (smaller independent noise component → expect r > 0.95).
        If False, the emulation has larger divergence (r ≈ 0.80–0.92),
        representing current OpenWorm fidelity limits.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    biotic     : np.ndarray, shape (302, 480) — biological ground truth
    simulated  : np.ndarray, shape (302, 480) — emulation output
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(N_SAMPLES) / FS   # time vector (seconds)

    # ── Shared spatial structure (low-rank) ───────────────────────────────
    # 3 dominant spatial modes, each with a characteristic oscillation
    # matching C. elegans locomotion frequencies.
    #
    # Mode 0: slow locomotion (~0.1 Hz) — body-wave propagation
    # Mode 1: reversal signal (~0.2 Hz) — AIB/RIM circuit
    # Mode 2: fast state switch (~0.5 Hz) — AVA reversal command neuron
    osc_slow = np.sin(2 * math.pi * 0.1  * t)   # locomotion wave
    osc_rev  = np.sin(2 * math.pi * 0.2  * t + math.pi / 4)
    osc_fast = np.sin(2 * math.pi * 0.5  * t + math.pi / 3)

    # Spatial weight vectors: each mode activates a distinct neuron ensemble
    rng_struct = np.random.default_rng(seed + 1)
    w0 = rng_struct.standard_normal(N_CHANNELS)  # body-wall muscle circuit
    w1 = rng_struct.standard_normal(N_CHANNELS)  # reversal interneurons
    w2 = rng_struct.standard_normal(N_CHANNELS)  # command neurons

    shared = (
        np.outer(w0, osc_slow) +
        np.outer(w1, osc_rev)  +
        np.outer(w2, osc_fast)
    )  # (302, 480)

    # ── Biotic signal: shared structure + pink noise ───────────────────────
    bio_noise = _pink_noise(N_CHANNELS, N_SAMPLES, fs=FS, seed=seed + 10)
    biotic    = shared + 0.4 * bio_noise

    # ── Simulated signal: shared structure + emulation error ──────────────
    # high_fidelity=True  → small error (ideal emulation)
    # high_fidelity=False → larger error (realistic OpenWorm divergence)
    noise_scale = 0.15 if high_fidelity else 0.70
    sim_noise   = _pink_noise(N_CHANNELS, N_SAMPLES, fs=FS, seed=seed + 99)
    simulated   = shared + noise_scale * sim_noise

    return biotic.astype(np.float64), simulated.astype(np.float64)


def load_real_data(biotic_path: str,
                   simulated_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real datasets from .npy files.

    Expected format: (n_channels, n_samples) float arrays.
    For Kato 2015 from CRCNS, each row is a neuron's ΔF/F trace.
    For OpenWorm, each row is a neuron's simulated spike-rate trace.
    """
    biotic    = np.load(biotic_path)
    simulated = np.load(simulated_path)
    for name, arr in [("biotic", biotic), ("simulated", simulated)]:
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected 2-D array, got {arr.ndim}-D")
        if arr.shape[0] != N_CHANNELS:
            raise ValueError(
                f"{name}: expected {N_CHANNELS} channels (C. elegans neurons), "
                f"got {arr.shape[0]}")
    return biotic, simulated


# ── Feature extraction (C. elegans parametrisation) ──────────────────────────

def extract_celegans_fingerprint(
        data: np.ndarray,
        embedding: SpatialEmbedding,
        fs: float = FS,
) -> np.ndarray:
    """Extract 12-component functional fingerprint from a neural activity matrix.

    Parameters
    ----------
    data      : np.ndarray, shape (302, n_samples) — raw activity traces
    embedding : SpatialEmbedding — fitted SVD spatial filter
    fs        : float — sample rate

    Returns
    -------
    np.ndarray, shape (12,) — values ∈ [0, 1]

    Component layout
    ────────────────
      q[ 0..7] : slow-band power of SVD components 0..7   (locomotion)
      q[ 8..9] : fast-band power of SVD components 0..1   (state switches)
      q[10]    : slow-band asymmetry  (component 0 vs. component 1)
      q[11]    : engagement analog    (fast/slow ratio across all components)
    """
    # Project: (302, n_samp) → (12, n_samp)
    projected = embedding.transform(data)   # (12, n_samples)

    # Band-filter each component
    slow_filtered = _bandpass(projected, *SLOW_BAND, fs=fs)
    fast_filtered = _bandpass(projected, *FAST_BAND, fs=fs)

    fingerprint = np.zeros(N_COMPONENTS, dtype=np.float64)

    # Compute log band-powers per component
    slow_log = np.array([
        math.log10(max(_band_power(slow_filtered[c], *SLOW_BAND, fs=fs), EPS))
        for c in range(N_COMPONENTS)
    ])
    fast_log = np.array([
        math.log10(max(_band_power(fast_filtered[c], *FAST_BAND, fs=fs), EPS))
        for c in range(N_COMPONENTS)
    ])

    # q[0..7] — slow-band power, components 0..7 (normalised to [0, 1])
    log_ref  = np.mean(slow_log)
    log_rng  = max(np.ptp(slow_log), 1e-6)
    for c in range(8):
        fingerprint[c] = np.clip((slow_log[c] - log_ref) / log_rng + 0.5, 0.0, 1.0)

    # q[8..9] — fast-band power, components 0..1
    fast_ref = np.mean(fast_log[:2])
    fast_rng = max(np.ptp(fast_log[:2]), 1e-6)
    for c in range(2):
        fingerprint[8 + c] = np.clip(
            (fast_log[c] - fast_ref) / fast_rng + 0.5, 0.0, 1.0)

    # q[10] — slow-band asymmetry: component 0 vs. component 1
    asym = slow_log[0] - slow_log[1]
    fingerprint[10] = np.clip((asym / 10.0 + 1.0) / 2.0, 0.0, 1.0)

    # q[11] — engagement analog: mean fast / mean slow ratio
    log_ratio  = float(np.mean(fast_log)) - float(np.mean(slow_log))
    fingerprint[11] = 1.0 / (1.0 + math.exp(-log_ratio * 0.5))

    return np.clip(fingerprint, 0.0, 1.0)


# ── Validation pipeline ───────────────────────────────────────────────────────

def run_validation(
        biotic: np.ndarray,
        simulated: np.ndarray,
        verbose: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Full Gap 1 validation pipeline.

    Returns
    -------
    r      : Pearson correlation coefficient (12 effective dimensions)
    p      : p-value
    T_bio  : 12-component fingerprint of biological data
    T_sim  : 12-component fingerprint of simulated data
    """
    print(_color("\n[KĦAOS-CORE] Fitting SVD spatial filter on biotic data…", _CYN))
    embedding = SpatialEmbedding(n_channels=N_CHANNELS, n_components=N_COMPONENTS)
    embedding.fit(biotic)
    print(f"  → {embedding}")

    print(_color("[KĦAOS-CORE] Extracting biotic fingerprint…", _CYN))
    T_bio = extract_celegans_fingerprint(biotic, embedding)

    print(_color("[KĦAOS-CORE] Extracting simulated fingerprint…", _CYN))
    T_sim = extract_celegans_fingerprint(simulated, embedding)

    r, p = pearsonr(T_bio, T_sim)
    return float(r), float(p), T_bio, T_sim


# ── Reporting ─────────────────────────────────────────────────────────────────

COMPONENT_LABELS = [
    "slow-PC0", "slow-PC1", "slow-PC2", "slow-PC3",
    "slow-PC4", "slow-PC5", "slow-PC6", "slow-PC7",
    "fast-PC0", "fast-PC1",
    "asymmetry", "engagement",
]


def _print_report(r: float, p: float,
                  T_bio: np.ndarray, T_sim: np.ndarray,
                  verbose: bool) -> None:
    """Print validation report to stdout."""

    print("\n" + "═" * 70)
    print(_color("  KĦAOS-CORE · Gap 1 Validation · C. elegans Pilot", _BLD))
    print("═" * 70)
    print(f"  Sample rate         : {FS} Hz  (Kato 2015)")
    print(f"  Channels            : {N_CHANNELS}  (C. elegans neurons)")
    print(f"  SVD components      : {N_COMPONENTS}")
    print(f"  Slow band           : {SLOW_BAND[0]}–{SLOW_BAND[1]} Hz  (locomotion)")
    print(f"  Fast band           : {FAST_BAND[0]}–{FAST_BAND[1]} Hz  (state switches)")
    print(f"  Fidelity threshold  : r > {FIDELITY_THR}")
    print("─" * 70)

    if verbose:
        print("\n  12-Component Fingerprint (Biotic vs. Simulated)\n")
        print(f"  {'Component':<15} {'T_bio':>8} {'T_sim':>8} {'|Δ|':>8}  {'bar'}")
        print("  " + "─" * 60)
        for i, label in enumerate(COMPONENT_LABELS):
            delta = abs(T_bio[i] - T_sim[i])
            bar_b = "█" * int(T_bio[i] * 15)
            bar_s = "▒" * int(T_sim[i] * 15)
            flag  = " ◄ " if delta > 0.15 else ""
            print(f"  {label:<15} {T_bio[i]:8.4f} {T_sim[i]:8.4f} {delta:8.4f}"
                  f"  {bar_b:<15}|{bar_s:<15}{flag}")
        print()

    print(f"\n  Pearson r   = {r:+.6f}")
    print(f"  p-value     = {p:.2e}")
    print(f"  Threshold   = {FIDELITY_THR}")
    print()

    if r >= FIDELITY_THR:
        print(_color(
            "  ╔══════════════════════════════════════════════════════════╗\n"
            "  ║  [VALIDATION PASS]  Functional Fidelity Confirmed       ║\n"
            "  ║  The emulation reproduces biological dynamics at r>0.95  ║\n"
            "  ╚══════════════════════════════════════════════════════════╝",
            _GRN))
    else:
        divergence = 1.0 - r
        # Find the two most divergent components
        deltas   = np.abs(T_bio - T_sim)
        top2_idx = np.argsort(deltas)[::-1][:2]
        top2_lbl = [COMPONENT_LABELS[i] for i in top2_idx]
        print(_color(
            "  ╔══════════════════════════════════════════════════════════╗\n"
            "  ║  [VALIDATION FAIL]  Emulation Divergence Detected       ║\n"
           f"  ║  r = {r:.4f}  (Δ from threshold: {divergence:.4f})              ║\n"
            "  ╚══════════════════════════════════════════════════════════╝",
            _RED))
        print(f"\n  Top divergent components: "
              f"{_color(top2_lbl[0], _YEL)}, {_color(top2_lbl[1], _YEL)}")
        print("  → These components indicate where the emulation diverges most")
        print("    from biological reality. Use them as targets for OpenWorm")
        print("    parameter tuning or additional constraint data.")

    print("\n" + "═" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="KĦAOS-CORE Gap 1 validator — C. elegans functional fidelity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--biotic",    type=str, default=None,
        metavar="PATH", help=".npy file: real C. elegans traces (302, n_samples)")
    parser.add_argument("--simulated", type=str, default=None,
        metavar="PATH", help=".npy file: OpenWorm simulation traces (302, n_samples)")
    parser.add_argument("--high-fidelity", action="store_true",
        help="Mock only: use high-fidelity emulation (expect r > 0.95)")
    parser.add_argument("--seed",  type=int, default=42,
        help="Random seed for mock data (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Print per-component fingerprint breakdown")
    args = parser.parse_args()

    using_real = args.biotic is not None and args.simulated is not None

    if using_real:
        print(_color(f"\n[KĦAOS-CORE] Loading real data…", _CYN))
        print(f"  Biotic    : {args.biotic}")
        print(f"  Simulated : {args.simulated}")
        try:
            biotic, simulated = load_real_data(args.biotic, args.simulated)
        except Exception as e:
            print(_color(f"\n[ERROR] Could not load data: {e}", _RED))
            return 1
    else:
        mode = "high-fidelity" if args.high_fidelity else "standard"
        print(_color(
            f"\n[KĦAOS-CORE] No real data provided — using mock ({mode} mode).",
            _YEL))
        print("  → Download Kato 2015 from https://crcns.org/data-sets/other/ceh-1")
        print("    and OpenWorm traces from https://github.com/openworm/c302")
        print("    then re-run with --biotic and --simulated flags.\n")
        biotic, simulated = load_mock_data(
            high_fidelity=args.high_fidelity, seed=args.seed)

    r, p, T_bio, T_sim = run_validation(biotic, simulated, verbose=args.verbose)
    _print_report(r, p, T_bio, T_sim, verbose=args.verbose)

    return 0 if r >= FIDELITY_THR else 1


if __name__ == "__main__":
    sys.exit(main())
