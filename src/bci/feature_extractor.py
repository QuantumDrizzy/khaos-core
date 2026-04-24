"""
feature_extractor.py — Multi-Channel EEG → 12-Qubit Neural Feature Vector
══════════════════════════════════════════════════════════════════════════════
Maps N EEG channels (N ∈ {4, 16, 32, 64}) to the 12 neurophysiologically
validated qubits required by khaos-core's quantum circuit layer.

  circuits.py:  N_QUBITS_MAIN=12, N_LAYERS=20  →  theta shape (240,)

Channel mode dispatch
─────────────────────
  n_channels = 4  (Muse 2 / dev mode)
    Direct qubit map — uses channel-specific biomarkers:
      q[ 0..3] = alpha power TP9/AF7/AF8/TP10
      q[ 4..7] = theta power TP9/AF7/AF8/TP10
      q[ 8]    = fronto-temporal coherence α: AF7↔TP9
      q[ 9]    = fronto-temporal coherence α: AF8↔TP10
      q[10]    = FAA Davidson  log(AF8_α) − log(AF7_α)
      q[11]    = Engagement θ/α (Pope et al. 1995)

  n_channels > 4  (clinical 16/32/64-ch array)
    Spatial embedding via truncated SVD (PCA): projects N channels to
    K=12 principal spatial components.  Band power computed per component.
    Qubit map:
      q[ 0..7] = alpha power of components 0..7
      q[ 8..9] = theta power of components 0..1
      q[10]    = FAA proxy: asymmetry of component 0 alpha vs. component 1 alpha
      q[11]    = Engagement: mean theta / mean alpha across all components

    The spatial filter must be fitted via ``fit_spatial_filter()`` before
    use.  An untrained identity filter is used as fallback (warns once).

Output contract (immutable)
───────────────────────────
  theta shape : (240,)  float64  ∈ [0, 2π]
  This contract is device-agnostic.  Calling code must not assume N_CHANNELS.

Ethics guard
────────────
This module operates AT the sovereignty boundary.  Raw EEG windows are
consumed here and only the 12-element (or 240-element tile) exits.
See src/ethics/ethics_gate.py.

References
──────────
Davidson (1988). EEG measures of cerebral asymmetry. Int J Neurosci, 39, 71–89.
Pope et al. (1995). Biocybernetic evaluation of operator engagement. Biol Psych.
Welch (1967). FFT for power spectra estimation. IEEE Trans Audio Electroacoust.
Wold et al. (1987). Principal component analysis. Chemometrics Intel Lab Sys.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.signal import coherence, welch

# ── Constants ─────────────────────────────────────────────────────────────────
FS            = 256          # Hz — Muse 2 sample rate
N_QUBITS      = 12           # main register
N_LAYERS      = 20           # PQC depth → theta shape = (240,)
THETA_LEN     = N_QUBITS * N_LAYERS   # 240

# Band edges (Hz)
ALPHA_BAND = (8.0,  13.0)
THETA_BAND = (4.0,   8.0)

# Channel indices in the (4, 512) window rows
IDX_TP9  = 0
IDX_AF7  = 1
IDX_AF8  = 2
IDX_TP10 = 3

# Sigmoid saturation point: values beyond ±SAT_DB dB are clipped before norm
SAT_DB = 60.0

# Epsilon to avoid log(0)
EPS = 1e-30


# ── Band power helper ──────────────────────────────────────────────────────────

def _band_power(signal: np.ndarray, f_low: float, f_high: float,
                fs: float = FS, nperseg: int = 256) -> float:
    """Welch PSD integrated over [f_low, f_high] Hz → absolute power (V²)."""
    freqs, pxx = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)),
                       window="hann", average="mean")
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return EPS
    # Trapezoidal integration — np.trapezoid added in numpy 2.0, trapz removed
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(pxx[mask], freqs[mask]))


def _log_band_power(signal: np.ndarray, f_low: float, f_high: float,
                    fs: float = FS) -> float:
    """Return log₁₀(band power) — used for ratio features."""
    return math.log10(max(_band_power(signal, f_low, f_high, fs), EPS))


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Sigmoid squashing: σ(x·scale) → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x * scale))


def _log_power_to_unit(log_power: float,
                       log_ref: float = -10.0,
                       dynamic_db: float = SAT_DB) -> float:
    """Map log₁₀ power to [0, 1] via linear clamp.

    log_ref     : expected log₁₀(power) at rest (≈ -10 for µV² signals)
    dynamic_db  : total dynamic range to represent
    """
    # Shift so log_ref maps to 0.5 (midpoint)
    shifted = (log_power - log_ref) / (dynamic_db / 10.0)
    return max(0.0, min(1.0, shifted + 0.5))


def _coherence_mean(sig_a: np.ndarray, sig_b: np.ndarray,
                    f_low: float, f_high: float,
                    fs: float = FS, nperseg: int = 256) -> float:
    """Mean magnitude-squared coherence in [f_low, f_high]."""
    freqs, Cxy = coherence(sig_a, sig_b, fs=fs,
                           nperseg=min(nperseg, len(sig_a)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(Cxy[mask]))


# ── Spatial embedding (N channels → 12 components via truncated SVD) ─────────

class SpatialEmbedding:
    """Projects N EEG channels to K=12 principal spatial components.

    Uses truncated SVD (equivalent to PCA without mean subtraction, which is
    appropriate for EEG where the global mean is artifactual).

    Parameters
    ----------
    n_channels : int   — number of input channels (must be > 4)
    n_components : int — number of output components (default 12 = N_QUBITS)

    Usage
    -----
    >>> emb = SpatialEmbedding(n_channels=64)
    >>> emb.fit(calibration_data)     # shape (64, n_samples)
    >>> projected = emb.transform(window)  # shape (12, n_samples)
    """

    def __init__(self, n_channels: int, n_components: int = N_QUBITS):
        self._n_ch    = n_channels
        self._n_comp  = n_components
        self._fitted  = False
        self._warned  = False
        # Spatial filter matrix: shape (n_components, n_channels)
        # Initialised to first n_components channels (identity-like fallback)
        rows = min(n_components, n_channels)
        self._W = np.eye(rows, n_channels)    # (n_comp, n_ch)
        if rows < n_components:
            # Pad with zeros for extra components
            pad = np.zeros((n_components - rows, n_channels))
            self._W = np.vstack([self._W, pad])

    def fit(self, data: np.ndarray) -> "SpatialEmbedding":
        """Fit the spatial filter from calibration data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Resting-state EEG, typically 2 min @ fs.

        Returns self for chaining.
        """
        if data.shape[0] != self._n_ch:
            raise ValueError(
                f"Expected {self._n_ch} channels, got {data.shape[0]}")
        # Zero-mean per channel (removes DC offset)
        data_c = data - data.mean(axis=1, keepdims=True)
        # Truncated SVD: data_c = U S Vt, keep first n_comp left singular vectors
        # U: (n_ch, n_comp) — these are the spatial filters
        U, S, _ = np.linalg.svd(data_c, full_matrices=False)
        k = min(self._n_comp, U.shape[1])
        # W maps channel space → component space: component = W @ signal
        self._W[:k, :] = U[:, :k].T
        if k < self._n_comp:
            self._W[k:, :] = 0.0   # remaining components set to zero
        self._fitted = True
        return self

    def transform(self, window: np.ndarray) -> np.ndarray:
        """Project window to component space.

        Parameters
        ----------
        window : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_components, n_samples)
        """
        if not self._fitted and not self._warned:
            warnings.warn(
                "SpatialEmbedding.fit() has not been called — using identity "
                "fallback. Call fit() with resting-state calibration data for "
                "optimal spatial filtering.",
                stacklevel=2)
            self._warned = True
        return self._W @ window  # (n_comp, n_samples)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return (f"SpatialEmbedding(n_channels={self._n_ch}, "
                f"n_components={self._n_comp}, fitted={self._fitted})")


# ── Main feature extractor ────────────────────────────────────────────────────

class Muse2FeatureExtractor:
    """Transforms a filtered EEG window into the khaos-core theta vector.

    Parameters
    ----------
    electrode_model : optional AgClDryContactModel
        If provided, its per-band correction_alpha weights are applied to
        scale each band power estimate before normalisation.
    fs : float
        Sample rate (default 256 Hz).

    Usage
    -----
    >>> extractor = Muse2FeatureExtractor()
    >>> window = adapter.get_filtered_window("alpha")  # (4, 512)
    >>> theta = extractor.extract(alpha_win, theta_win)
    >>> assert theta.shape == (240,)
    """

    def __init__(self, electrode_model=None, fs: float = FS,
                 n_channels: int = 4):
        """
        Parameters
        ----------
        electrode_model : optional AgClDryContactModel / ElectrodeModel
            Per-band impedance correction weights.
        fs          : float — sample rate (default 256 Hz)
        n_channels  : int  — number of input EEG channels.
            4  : Muse 2 direct qubit map (default).
            >4 : Clinical array — uses SpatialEmbedding (PCA).
                 Call fit_spatial_filter() before first use.
        """
        self._model      = electrode_model
        self._fs         = fs
        self._n_channels = n_channels

        # Per-band alpha weights from electrode model (defaults to 1.0)
        if electrode_model is not None and hasattr(electrode_model,
                                                    "correction_alpha"):
            alphas = electrode_model.correction_alpha()  # shape (5,)
            self._alpha_theta = float(alphas[1])
            self._alpha_alpha = float(alphas[2])
        else:
            self._alpha_theta = 1.0
            self._alpha_alpha = 1.0

        # Running baseline for relative normalisation (updated by calibrate())
        self._alpha_ref: Optional[np.ndarray] = None
        self._theta_ref: Optional[np.ndarray] = None

        # Spatial embedding for n_channels > 4
        if n_channels > 4:
            self._spatial = SpatialEmbedding(n_channels=n_channels,
                                             n_components=N_QUBITS)
        else:
            self._spatial = None

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, alpha_window: np.ndarray,
                theta_window: np.ndarray) -> np.ndarray:
        """Compute the 240-element theta vector.

        Parameters
        ----------
        alpha_window : np.ndarray, shape (4, 512)
            Band-filtered alpha window (8–13 Hz) per channel.
        theta_window : np.ndarray, shape (4, 512)
            Band-filtered theta window (4–8 Hz) per channel.

        Returns
        -------
        np.ndarray, shape (240,) — dtype float64, values ∈ [0, 2π]

        Raises
        ------
        ValueError if window shapes are incorrect.
        """
        self._validate_windows(alpha_window, theta_window)
        qubits = self._compute_qubits(alpha_window, theta_window)
        return self._tile_to_theta(qubits)

    def extract_from_adapter(self, adapter) -> np.ndarray:
        """Convenience wrapper: pull windows directly from a Muse2Adapter.

        Blocks until the adapter buffer is ready.
        """
        if not adapter.ready:
            raise RuntimeError("Adapter buffer not ready — call wait_ready() first.")
        alpha_win = adapter.get_filtered_window("alpha")
        theta_win = adapter.get_filtered_window("theta")
        return self.extract(alpha_win, theta_win)

    def fit_spatial_filter(self, adapter, duration_s: float = 120.0) -> None:
        """Fit the PCA spatial filter from resting-state data (n_channels > 4).

        Does nothing if n_channels == 4 (spatial embedding not used).

        Parameters
        ----------
        adapter    : any adapter with get_filtered_window()
        duration_s : calibration duration in seconds
        """
        if self._spatial is None:
            return   # 4-channel mode — no spatial filter needed

        import time as _time
        print(f"[FeatureExtractor] Fitting spatial filter: {duration_s:.0f} s…")
        t0 = _time.time()
        windows = []
        while _time.time() - t0 < duration_s:
            if adapter.ready:
                # Use raw (notch-filtered) window for PCA
                windows.append(adapter.get_filtered_window("raw"))
            _time.sleep(0.5)

        if not windows:
            warnings.warn("No data collected for spatial filter fitting.")
            return

        # Concatenate along time axis: (n_channels, total_samples)
        data = np.concatenate(windows, axis=1)
        self._spatial.fit(data)
        print(f"[FeatureExtractor] Spatial filter fitted: {self._spatial}")

    def calibrate(self, adapter, duration_s: float = 120.0) -> None:
        """Record 2-minute eyes-closed resting baseline for relative norms.

        Intended to be called once before live extraction.  Updates internal
        reference log-powers so that FAA and engagement index are expressed
        relative to each user's individual baseline.

        Parameters
        ----------
        adapter  : Muse2Adapter  (or SyntheticMuse2Adapter)
        duration_s : float   — calibration window in seconds
        """
        import time
        print(f"[FeatureExtractor] Calibration: {duration_s:.0f} s eyes-closed…")
        t0 = time.time()

        alpha_accum = []
        theta_accum = []

        while time.time() - t0 < duration_s:
            if adapter.ready:
                alpha_accum.append(adapter.get_filtered_window("alpha"))
                theta_accum.append(adapter.get_filtered_window("theta"))
            time.sleep(0.5)

        if not alpha_accum:
            print("[FeatureExtractor] Calibration failed — no data.")
            return

        # Average log-power per channel over the calibration period
        def _mean_log_power(windows, band):
            return np.mean(
                [[_log_band_power(w[ch], *band) for ch in range(4)]
                 for w in windows], axis=0)

        self._alpha_ref = _mean_log_power(alpha_accum, ALPHA_BAND)
        self._theta_ref = _mean_log_power(theta_accum, THETA_BAND)
        print("[FeatureExtractor] Calibration complete.")
        print(f"  α_ref (log₁₀ V²): {self._alpha_ref}")
        print(f"  θ_ref (log₁₀ V²): {self._theta_ref}")

    # ── Internal computation ────────────────────────────────────────────────

    def _validate_windows(self, alpha: np.ndarray, theta: np.ndarray) -> None:
        for name, arr in [("alpha_window", alpha), ("theta_window", theta)]:
            if arr.ndim != 2:
                raise ValueError(f"{name}: expected 2-D array, got {arr.ndim}-D")
            if arr.shape[0] != self._n_channels:
                raise ValueError(
                    f"{name}: expected {self._n_channels} channels, "
                    f"got {arr.shape[0]}. "
                    f"Instantiate with n_channels={arr.shape[0]}.")
            if arr.shape[1] < 32:
                raise ValueError(
                    f"{name}: need at least 32 samples, got {arr.shape[1]}")

    def _compute_qubits(self, alpha_win: np.ndarray,
                        theta_win: np.ndarray) -> np.ndarray:
        """Return 12 qubit values ∈ [0, 1] (not yet scaled to 2π).

        Dispatches to the 4-channel direct map or the N-channel PCA embedding
        based on self._n_channels.
        """
        if self._n_channels > 4:
            return self._compute_qubits_multichannel(alpha_win, theta_win)
        return self._compute_qubits_4ch(alpha_win, theta_win)

    def _compute_qubits_multichannel(self, alpha_win: np.ndarray,
                                      theta_win: np.ndarray) -> np.ndarray:
        """Multi-channel qubit extraction via SpatialEmbedding (PCA).

        Projects N channels → 12 principal spatial components, then maps:
          q[ 0..7] = alpha power of components 0..7
          q[ 8..9] = theta power of components 0..1
          q[10]    = FAA proxy (component 0 vs. component 1 alpha asymmetry)
          q[11]    = engagement θ/α (global mean across all components)
        """
        # Project: (n_ch, n_samp) → (12, n_samp)
        alpha_proj = self._spatial.transform(alpha_win)
        theta_proj = self._spatial.transform(theta_win)

        qubits = np.zeros(N_QUBITS, dtype=np.float64)

        # Alpha log powers for all 12 components
        alpha_log = np.array([
            _log_band_power(alpha_proj[c], *ALPHA_BAND, fs=self._fs)
            for c in range(N_QUBITS)
        ])

        # Theta log powers for first 2 components
        theta_log_0 = _log_band_power(theta_proj[0], *THETA_BAND, fs=self._fs)
        theta_log_1 = _log_band_power(theta_proj[1], *THETA_BAND, fs=self._fs)

        # q[0..7] — alpha power of spatial components 0..7
        alpha_ref = self._alpha_ref if self._alpha_ref is not None \
            else np.full(N_QUBITS, -10.0)
        for c in range(8):
            qubits[c] = _log_power_to_unit(alpha_log[c],
                                            log_ref=float(alpha_ref[min(c, len(alpha_ref)-1)]))

        # q[8..9] — theta power of components 0 and 1
        theta_ref_scalar = float(self._theta_ref[0]) if self._theta_ref is not None else -10.5
        qubits[8] = _log_power_to_unit(theta_log_0, log_ref=theta_ref_scalar)
        qubits[9] = _log_power_to_unit(theta_log_1, log_ref=theta_ref_scalar)

        # q[10] — FAA proxy: component 0 vs component 1 alpha asymmetry
        faa_proxy = alpha_log[0] - alpha_log[1]
        qubits[10] = float(np.clip((faa_proxy / SAT_DB + 1.0) / 2.0, 0.0, 1.0))

        # q[11] — engagement: mean theta / mean alpha across all components
        all_theta_log = np.array([
            _log_band_power(theta_proj[c], *THETA_BAND, fs=self._fs)
            for c in range(N_QUBITS)
        ])
        engagement = _sigmoid(float(np.mean(all_theta_log)) - float(np.mean(alpha_log)),
                               scale=0.5)
        qubits[11] = float(engagement)

        return np.clip(qubits, 0.0, 1.0)

    def _compute_qubits_4ch(self, alpha_win: np.ndarray,
                             theta_win: np.ndarray) -> np.ndarray:
        """Original 4-channel qubit map (Muse 2 direct)."""
        qubits = np.zeros(N_QUBITS, dtype=np.float64)

        # ── q[0..3] — Alpha power per channel ──────────────────────────────
        alpha_log = np.array([
            _log_band_power(alpha_win[ch], *ALPHA_BAND, fs=self._fs)
            for ch in range(4)
        ])
        alpha_ref = self._alpha_ref if self._alpha_ref is not None \
            else np.full(4, -10.0)

        for ch in range(4):
            raw_norm = _log_power_to_unit(alpha_log[ch], log_ref=alpha_ref[ch])
            qubits[ch] = raw_norm * self._alpha_alpha

        # ── q[4..7] — Theta power per channel ──────────────────────────────
        theta_log = np.array([
            _log_band_power(theta_win[ch], *THETA_BAND, fs=self._fs)
            for ch in range(4)
        ])
        theta_ref = self._theta_ref if self._theta_ref is not None \
            else np.full(4, -10.5)

        for ch in range(4):
            raw_norm = _log_power_to_unit(theta_log[ch], log_ref=theta_ref[ch])
            qubits[4 + ch] = raw_norm * self._alpha_theta

        # ── q[8] — Fronto-temporal coherence α: AF7 ↔ TP9 ─────────────────
        coh_left = _coherence_mean(
            alpha_win[IDX_AF7], alpha_win[IDX_TP9],
            *ALPHA_BAND, fs=self._fs)
        qubits[8] = float(np.clip(coh_left, 0.0, 1.0))

        # ── q[9] — Fronto-temporal coherence α: AF8 ↔ TP10 ────────────────
        coh_right = _coherence_mean(
            alpha_win[IDX_AF8], alpha_win[IDX_TP10],
            *ALPHA_BAND, fs=self._fs)
        qubits[9] = float(np.clip(coh_right, 0.0, 1.0))

        # ── q[10] — Frontal Alpha Asymmetry (FAA) ──────────────────────────
        # Davidson (1988): FAA = log(AF8_alpha) − log(AF7_alpha)
        # Positive = left-frontal dominance (approach/positive affect)
        # Negative = right-frontal dominance (withdrawal/negative affect)
        faa_raw = alpha_log[IDX_AF8] - alpha_log[IDX_AF7]
        # Map [-SAT_DB, +SAT_DB] → [0, 1] linearly
        faa_norm = (faa_raw / SAT_DB + 1.0) / 2.0
        qubits[10] = float(np.clip(faa_norm, 0.0, 1.0))

        # ── q[11] — Engagement Index θ/α ratio (Pope et al. 1995) ──────────
        # Engagement rises with theta and falls with alpha.
        # Use mean alpha and mean theta power across all channels.
        mean_alpha_log = float(np.mean(alpha_log))
        mean_theta_log = float(np.mean(theta_log))
        # Ratio in log domain: θ/α = 10^(θ_log) / 10^(α_log)
        log_ratio = mean_theta_log - mean_alpha_log
        # Sigmoid: log_ratio ≈ 0 → 0.5 (engaged), negative → 0 (relaxed)
        engagement = _sigmoid(log_ratio, scale=0.5)
        qubits[11] = float(engagement)

        return np.clip(qubits, 0.0, 1.0)

    @staticmethod
    def _tile_to_theta(qubits: np.ndarray) -> np.ndarray:
        """Tile 12-qubit values across 20 layers → theta shape (240,).

        Each layer receives the same rotation angles.  The PQC learns
        per-layer weight matrices that transform this shared input.
        Scale from [0, 1] → [0, 2π].
        """
        theta = np.tile(qubits * (2.0 * math.pi), N_LAYERS)
        assert theta.shape == (THETA_LEN,), \
            f"Unexpected theta shape: {theta.shape}"
        return theta

    # ── Diagnostics ────────────────────────────────────────────────────────

    def qubit_labels(self) -> list:
        if self._n_channels == 4:
            return [
                "α-TP9", "α-AF7", "α-AF8", "α-TP10",
                "θ-TP9", "θ-AF7", "θ-AF8", "θ-TP10",
                "Coh-L(AF7↔TP9)", "Coh-R(AF8↔TP10)",
                "FAA", "Engagement",
            ]
        return [
            f"α-PC{i}" for i in range(8)
        ] + ["θ-PC0", "θ-PC1", "FAA-proxy", "Engagement"]

    def explain(self, alpha_window: np.ndarray,
                theta_window: np.ndarray) -> dict:
        """Return a human-readable breakdown of each qubit value."""
        self._validate_windows(alpha_window, theta_window)
        qubits = self._compute_qubits(alpha_window, theta_window)
        theta  = self._tile_to_theta(qubits)
        return {
            "qubits_01":   qubits.tolist(),
            "qubits_2pi":  (qubits * 2 * math.pi).tolist(),
            "theta_shape": theta.shape,
            "labels":      self.qubit_labels(),
        }


# ── Convenience function ───────────────────────────────────────────────────────

def extract_theta(adapter, electrode_model=None) -> np.ndarray:
    """One-shot: extract theta vector from a running adapter.

    Parameters
    ----------
    adapter        : Muse2Adapter or SyntheticMuse2Adapter
    electrode_model: optional ElectrodeModel for impedance correction

    Returns
    -------
    np.ndarray, shape (240,)
    """
    extractor = Muse2FeatureExtractor(electrode_model=electrode_model)
    return extractor.extract_from_adapter(adapter)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

    from src.io.muse2_adapter import SyntheticMuse2Adapter

    print("=== feature_extractor self-test ===\n")

    adapter = SyntheticMuse2Adapter()
    adapter.connect()
    adapter.start()

    print("Filling buffer…")
    if not adapter.wait_ready(timeout=5.0):
        print("Buffer not ready.")
        sys.exit(1)

    extractor = Muse2FeatureExtractor()
    alpha_win = adapter.get_filtered_window("alpha")
    theta_win = adapter.get_filtered_window("theta")

    info = extractor.explain(alpha_win, theta_win)
    print("Qubit values [0, 1]:")
    for label, val in zip(info["labels"], info["qubits_01"]):
        bar = "█" * int(val * 20)
        print(f"  {label:<25} {val:.4f}  {bar}")

    theta = extractor.extract(alpha_win, theta_win)
    print(f"\ntheta vector shape : {theta.shape}")
    print(f"theta min / max    : {theta.min():.4f} / {theta.max():.4f}")
    print(f"Expected range     : [0, {2*math.pi:.4f}]")
    assert theta.shape == (240,), "Shape mismatch!"
    assert 0.0 <= theta.min() and theta.max() <= 2 * math.pi + 1e-9, \
        "Values out of [0, 2π]!"

    adapter.stop()
    print("\nSelf-test passed. ✓")
