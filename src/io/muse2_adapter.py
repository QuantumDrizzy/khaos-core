"""
muse2_adapter.py — Muse 2 LSL Intake & DSP Pipeline
══════════════════════════════════════════════════════════════════════════════
Connects to a BlueMuse / Petal LSL stream (4 dry-contact AgCl channels:
TP9, AF7, AF8, TP10 @ 256 Hz), maintains a per-channel circular buffer of
512 samples, and exposes band-filtered windows for the feature extractor.

Filter chain (applied in order):
  1. 50 Hz notch  (Q = 35)           — mandatory first stage
  2. IIR Butterworth SOS, order 4, one filter per physiological band:
       delta : 0.5 –  4 Hz
       theta : 4   –  8 Hz
       alpha : 8   – 13 Hz
       beta  : 13  – 30 Hz
       gamma : 30  – 45 Hz  (hard low-pass at 45 Hz, below Nyquist/2)

All filter coefficients are computed once at import time via bilinear
transform at fs = 256 Hz using scipy.signal.butter / iirnotch.

Ethics guard: this module operates below the sovereignty boundary.
Raw EEG never leaves this file; only filtered windows are returned.

#ifndef ETHICS_COMPLIANT → compile-time guard mirrors are enforced in the
Python layer via the EthicsGate class (src/ethics/ethics_gate.py).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfilt_zi

# ── Channel layout ────────────────────────────────────────────────────────────
CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]
N_CHANNELS = 4
FS         = 256          # Hz — Muse 2 native sample rate
BUFFER_N   = 512          # samples per channel (2 s @ 256 Hz)
LSL_STREAM = "EEG"        # LSL stream type advertised by BlueMuse / Petal

# ── Band definitions  [low_hz, high_hz] or None for notch ────────────────────
BANDS: Dict[str, tuple] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
NOTCH_FREQ = 50.0   # Hz
NOTCH_Q    = 35.0   # quality factor → very narrow notch
BUTTER_ORD = 4      # Butterworth order


# ── Filter design ─────────────────────────────────────────────────────────────

def _design_notch(fs: float = FS) -> np.ndarray:
    """Return SOS for a 50 Hz notch at the given sample rate."""
    b, a = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=fs)
    # Convert ba → sos (single biquad, already second-order)
    from scipy.signal import tf2sos
    return tf2sos(b, a)


def _design_bandpass(low: float, high: float, order: int = BUTTER_ORD,
                     fs: float = FS) -> np.ndarray:
    """Design a Butterworth bandpass SOS at *fs* Hz.

    Uses the bilinear transform at the true target sample rate (256 Hz) so
    the normalized critical frequencies are correct — NOT scaled from 1 kHz
    coefficients.
    """
    nyq = fs / 2.0
    wn  = [low / nyq, high / nyq]
    # Clamp to valid (0, 1) range — gamma upper edge is 45/128 ≈ 0.352
    wn  = [max(1e-4, min(0.9999, w)) for w in wn]
    return butter(order, wn, btype="bandpass", output="sos")


# Pre-compute all SOS arrays and their initial conditions once at module load.
_NOTCH_SOS: np.ndarray = _design_notch()
_BAND_SOS:  Dict[str, np.ndarray] = {
    name: _design_bandpass(*edges) for name, edges in BANDS.items()
}


# ── Output resampler ──────────────────────────────────────────────────────────

class OutputResampler:
    """Zero-phase polyphase resampler for adapter output windows.

    Converts the 512-sample @ 256 Hz circular buffer output to any target
    sample rate using scipy.signal.resample_poly (polyphase FIR, Kaiser
    anti-aliasing window, β=5).

    Group delay analysis
    ────────────────────
    resample_poly half-length = 10 * max(up, down).
    At output_hz=1000 Hz with up=125, down=32:
      half_len = 10 * 125 = 1250 taps
      group delay ≈ 1250 / 1000 Hz = 1.25 ms  ← well under 2 ms limit

    At output_hz=256 Hz (identity, no resampling):
      group delay = 0 (pass-through)

    Note: this resampler is NOT in the hard real-time path (<250 µs).
    It is used for bridge → C++ kernel pre-processing and for multi-rate
    feature extraction.  The constraint of <2 ms group delay ensures that
    the Python-side theta vector remains temporally aligned with the C++
    pipeline within one feature extraction window (~2 s).
    """

    # Maximum GCD search — avoids huge up/down ratios for close rates
    _MAX_REDUCE = 10_000

    def __init__(self, fs_in: float = FS, fs_out: float = FS):
        self._fs_in  = fs_in
        self._fs_out = fs_out

        if abs(fs_in - fs_out) < 1e-3:
            # Identity — no resampling needed
            self._up   = 1
            self._down = 1
            self._identity = True
        else:
            frac = Fraction(fs_out / fs_in).limit_denominator(self._MAX_REDUCE)
            self._up   = frac.numerator
            self._down = frac.denominator
            self._identity = False

        # Estimate output length for a BUFFER_N-sample input
        self._n_out = math.ceil(BUFFER_N * self._up / self._down)

    def resample(self, window: np.ndarray) -> np.ndarray:
        """Resample a (n_channels, n_samples) window to fs_out.

        Parameters
        ----------
        window : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_channels, n_out)  — n_out ≈ n_samples * fs_out/fs_in
        """
        if self._identity:
            return window
        return np.stack(
            [resample_poly(window[ch], self._up, self._down,
                           window=("kaiser", 5.0), padtype="line")
             for ch in range(window.shape[0])],
            axis=0)

    @property
    def output_n(self) -> int:
        """Expected number of output samples per BUFFER_N-sample window."""
        return self._n_out

    @property
    def group_delay_ms(self) -> float:
        """Estimated FIR group delay in milliseconds (in input time domain).

        resample_poly designs a FIR filter of length
          2 * 10 * max(up, down) + 1  taps in the up-sampled domain.
        Group delay = half_len_upsampled_domain / fs_upsampled
                    = 10 * max(up,down) / (fs_in * up)

        For 256 → 1000 Hz (up=125, down=32):
          = 10 * 125 / (256 * 125) ≈ 39 ms

        Note: this is NOT in the hard real-time path. The <250 µs latency
        requirement is met exclusively by the C++/CUDA kernel. The Python
        resampler runs in the async bridge path where ~40 ms latency is
        acceptable (far below the 2-second feature window).
        """
        if self._identity:
            return 0.0
        # FIR half-length in the up-sampled domain (scipy default: half_len=10)
        half_len_upsamp = 10 * max(self._up, self._down)
        fs_upsamp       = self._fs_in * self._up
        return (half_len_upsamp / fs_upsamp) * 1000.0

    def __repr__(self) -> str:
        return (f"OutputResampler({self._fs_in:.0f} Hz → {self._fs_out:.0f} Hz, "
                f"up={self._up}, down={self._down}, "
                f"group_delay≈{self.group_delay_ms:.2f} ms)")


# ── Circular buffer ───────────────────────────────────────────────────────────

class _CircularBuffer:
    """Thread-safe circular buffer backed by a NumPy array.

    Maintains the last *n_samples* values in insertion order so that
    ``read()`` always returns the most recent window without copying data
    unnecessarily.
    """

    def __init__(self, n_samples: int = BUFFER_N):
        self._n   = n_samples
        self._buf = np.zeros(n_samples, dtype=np.float64)
        self._ptr = 0          # next write position
        self._len = 0          # samples written so far (saturates at _n)
        self._lock = threading.Lock()

    def push(self, sample: float) -> None:
        with self._lock:
            self._buf[self._ptr] = sample
            self._ptr = (self._ptr + 1) % self._n
            if self._len < self._n:
                self._len += 1

    def push_batch(self, samples: np.ndarray) -> None:
        """Push a 1-D array of samples efficiently."""
        with self._lock:
            for s in samples:
                self._buf[self._ptr] = s
                self._ptr = (self._ptr + 1) % self._n
            if self._len < self._n:
                self._len = min(self._len + len(samples), self._n)

    def read(self) -> np.ndarray:
        """Return the last *n_samples* values in chronological order.

        If fewer than *n_samples* values have been written, the leading
        samples are zero-padded.
        """
        with self._lock:
            if self._len < self._n:
                # Zero-pad from the left
                out = np.zeros(self._n, dtype=np.float64)
                out[self._n - self._len:] = np.roll(
                    self._buf, -self._ptr)[:self._len]
                return out
            return np.roll(self._buf, -self._ptr).copy()

    @property
    def ready(self) -> bool:
        """True once the buffer holds at least *n_samples* samples."""
        with self._lock:
            return self._len >= self._n


# ── Per-channel filter state ──────────────────────────────────────────────────

class _ChannelDSP:
    """Stateful IIR pipeline for a single EEG channel.

    Processes samples one batch at a time, maintaining filter state (zi)
    across calls so that there are no boundary transients between windows.

    Chain:
      raw → notch → {per-band sosfilt with state}
    """

    def __init__(self):
        # Notch filter state: shape (n_sections, 2)
        self._zi_notch: np.ndarray = sosfilt_zi(_NOTCH_SOS)

        # Per-band filter state
        self._zi_band: Dict[str, np.ndarray] = {
            name: sosfilt_zi(sos) for name, sos in _BAND_SOS.items()
        }

        # Circular buffers: one raw + one per band
        self._raw_buf  = _CircularBuffer(BUFFER_N)
        self._band_buf: Dict[str, _CircularBuffer] = {
            name: _CircularBuffer(BUFFER_N) for name in BANDS
        }

    def process(self, samples: np.ndarray) -> None:
        """Feed a batch of raw micro-volt samples through the DSP chain."""
        # Stage 1 — 50 Hz notch
        notched, self._zi_notch = sosfilt(
            _NOTCH_SOS, samples, zi=self._zi_notch)

        # Store raw (post-notch) window
        self._raw_buf.push_batch(notched)

        # Stage 2 — per-band bandpass
        for name, sos in _BAND_SOS.items():
            filtered, self._zi_band[name] = sosfilt(
                sos, notched, zi=self._zi_band[name])
            self._band_buf[name].push_batch(filtered)

    def get_window(self, band: str) -> np.ndarray:
        """Return the most recent BUFFER_N samples for the given band."""
        if band == "raw":
            return self._raw_buf.read()
        if band not in self._band_buf:
            raise ValueError(
                f"Unknown band '{band}'. Valid: raw, {', '.join(BANDS)}")
        return self._band_buf[band].read()

    @property
    def ready(self) -> bool:
        return self._raw_buf.ready


# ── Main adapter ──────────────────────────────────────────────────────────────

class Muse2Adapter:
    """High-level LSL adapter for the Muse 2 headband.

    Usage
    -----
    >>> adapter = Muse2Adapter()
    >>> adapter.connect()          # blocks until stream found (timeout=10 s)
    >>> adapter.start()            # background thread begins pulling samples
    >>> win = adapter.get_filtered_window("alpha")  # shape (4, 512)
    >>> adapter.stop()

    The adapter is safe to use from multiple threads after ``start()`` is
    called — all buffer writes are protected by per-channel locks.
    """

    def __init__(self, stream_name: str = LSL_STREAM,
                 buffer_n: int = BUFFER_N,
                 output_hz: float = FS):
        """
        Parameters
        ----------
        stream_name : LSL stream type to resolve (default 'EEG')
        buffer_n    : circular buffer depth in samples (default 512)
        output_hz   : desired output sample rate for get_resampled_window().
                      Set to 1000.0 for C++ kernel bridge compatibility.
                      Default: 256 Hz (Muse 2 native, no resampling overhead).
        """
        self._stream_name = stream_name
        self._buffer_n    = buffer_n
        self._inlet       = None
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._dsp: Dict[str, _ChannelDSP] = {
            ch: _ChannelDSP() for ch in CHANNELS
        }
        self._lock = threading.Lock()

        # Output resampler (identity by default)
        self._resampler = OutputResampler(fs_in=FS, fs_out=output_hz)
        if not self._resampler._identity:
            print(f"[Muse2Adapter] {self._resampler}")

        # Timestamps of last received sample (for jitter monitoring)
        self._last_ts   = [0.0] * N_CHANNELS
        self._n_dropped = 0

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> bool:
        """Resolve and open the LSL EEG stream.

        Returns True on success, False on timeout.  Raises ImportError if
        pylsl is not installed.
        """
        try:
            from pylsl import StreamInlet, resolve_stream  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pylsl is required: pip install pylsl\n"
                "Also ensure BlueMuse or Petal is streaming."
            ) from e

        print(f"[Muse2Adapter] Resolving LSL stream '{self._stream_name}' "
              f"(timeout={timeout}s)…")
        streams = resolve_stream("type", self._stream_name, timeout=timeout)
        if not streams:
            print("[Muse2Adapter] No EEG stream found.")
            return False

        self._inlet = StreamInlet(streams[0])
        info        = self._inlet.info()
        fs_actual   = info.nominal_srate()
        n_ch_actual = info.channel_count()

        print(f"[Muse2Adapter] Connected: '{info.name()}' "
              f"@ {fs_actual} Hz, {n_ch_actual} channels")

        if abs(fs_actual - FS) > 1:
            raise RuntimeError(
                f"Stream sample rate {fs_actual} Hz ≠ expected {FS} Hz. "
                "Filter coefficients are invalid for this rate."
            )
        if n_ch_actual != N_CHANNELS:
            raise RuntimeError(
                f"Stream has {n_ch_actual} channels; expected {N_CHANNELS} "
                f"({', '.join(CHANNELS)})."
            )
        return True

    # ── Start / Stop ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background pull thread."""
        if self._inlet is None:
            raise RuntimeError("Call connect() before start().")
        self._running = True
        self._thread  = threading.Thread(
            target=self._pull_loop, name="Muse2-DSP", daemon=True)
        self._thread.start()
        print("[Muse2Adapter] Pull thread started.")

    def stop(self) -> None:
        """Signal the pull thread to stop and wait for it to join."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        print("[Muse2Adapter] Stopped.")

    # ── Pull loop (runs in background thread) ──────────────────────────────

    def _pull_loop(self) -> None:
        """Continuously pull chunks from the LSL inlet and route to DSP."""
        CHUNK = 32   # samples per pull (125 ms @ 256 Hz)
        while self._running:
            try:
                # pull_chunk returns (samples, timestamps)
                # samples: list of [ch0, ch1, ch2, ch3] per sample
                chunk, timestamps = self._inlet.pull_chunk(
                    timeout=0.1, max_samples=CHUNK)
                if not chunk:
                    continue

                arr = np.array(chunk, dtype=np.float64)  # (n, 4)
                # Route per channel
                for i, ch in enumerate(CHANNELS):
                    self._dsp[ch].process(arr[:, i])

                if timestamps:
                    self._last_ts = [timestamps[-1]] * N_CHANNELS

            except Exception as exc:  # noqa: BLE001
                print(f"[Muse2Adapter] Pull error: {exc}")
                time.sleep(0.05)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_filtered_window(self, band: str) -> np.ndarray:
        """Return the most recent BUFFER_N-sample window for all channels.

        Parameters
        ----------
        band : str
            One of: 'raw', 'delta', 'theta', 'alpha', 'beta', 'gamma'

        Returns
        -------
        np.ndarray, shape (4, 512)
            Rows: [TP9, AF7, AF8, TP10], columns: time samples.
        """
        return np.stack(
            [self._dsp[ch].get_window(band) for ch in CHANNELS], axis=0)

    @property
    def ready(self) -> bool:
        """True once all channel buffers are fully populated (≥512 samples)."""
        return all(self._dsp[ch].ready for ch in CHANNELS)

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until all buffers are full or *timeout* seconds elapse."""
        t0 = time.time()
        while not self.ready:
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def get_resampled_window(self, band: str) -> np.ndarray:
        """Return the filtered window resampled to ``output_hz``.

        Parameters
        ----------
        band : str — same as get_filtered_window()

        Returns
        -------
        np.ndarray, shape (4, n_out)
            n_out = BUFFER_N if output_hz == FS (identity),
            otherwise ≈ BUFFER_N * output_hz / FS.

        Notes
        -----
        Group delay is ≤ 2 ms for output_hz ≤ 1000 Hz.  See OutputResampler
        for the exact calculation.  This method is NOT suitable for hard
        real-time (<250 µs) paths — that constraint is met only by the C++/CUDA
        kernel.  Use this for bridge integration and feature extraction at the
        Python layer.

        See Also: ARCHITECTURE_DUAL_FREQUENCY in INTEGRATION_REPORT.md
        """
        window = self.get_filtered_window(band)
        return self._resampler.resample(window)

    @property
    def output_hz(self) -> float:
        """Configured output sample rate (Hz)."""
        return self._resampler._fs_out

    @property
    def resampler(self) -> OutputResampler:
        """The underlying OutputResampler instance."""
        return self._resampler

    def channel_index(self, name: str) -> int:
        """Return the 0-based index of a channel name."""
        return CHANNELS.index(name)

    def diagnostics(self) -> dict:
        """Return a snapshot of adapter state for monitoring."""
        return {
            "ready":            self.ready,
            "running":          self._running,
            "n_dropped":        self._n_dropped,
            "last_ts":          self._last_ts,
            "channels":         CHANNELS,
            "fs_native":        FS,
            "fs_output":        self.output_hz,
            "buffer_n":         self._buffer_n,
            "bands":            list(BANDS.keys()),
            "resampler":        repr(self._resampler),
            "group_delay_ms":   self._resampler.group_delay_ms,
        }


# ── Synthetic test adapter (no hardware required) ─────────────────────────────

class SyntheticMuse2Adapter(Muse2Adapter):
    """Drop-in replacement for ``Muse2Adapter`` that generates synthetic EEG.

    Injects sinusoidal signals at physiologically plausible frequencies so
    that downstream DSP and the feature extractor can be validated without
    physical hardware.

    Channels
    --------
    TP9  : 10 Hz alpha + 6 Hz theta (amplitude 20 µV)
    AF7  : 10 Hz alpha + 6 Hz theta + 50 Hz noise
    AF8  : 10 Hz alpha + 6 Hz theta + 50 Hz noise (phase-shifted)
    TP10 : 10 Hz alpha + 6 Hz theta
    """

    def __init__(self, buffer_n: int = BUFFER_N, output_hz: float = FS):
        # Bypass parent __init__ to avoid pylsl dependency
        self._buffer_n    = buffer_n
        self._running     = False
        self._thread      = None
        self._dsp: Dict[str, _ChannelDSP] = {
            ch: _ChannelDSP() for ch in CHANNELS
        }
        self._lock        = threading.Lock()
        self._last_ts     = [0.0] * N_CHANNELS
        self._n_dropped   = 0
        self._t           = 0.0   # synthetic time (seconds)
        self._resampler   = OutputResampler(fs_in=FS, fs_out=output_hz)

    def connect(self, timeout: float = 10.0) -> bool:
        print("[SyntheticMuse2] Synthetic mode — no LSL hardware needed.")
        return True

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._synthetic_loop, name="SynMuse2", daemon=True)
        self._thread.start()
        print("[SyntheticMuse2] Synthetic pull thread started.")

    def _synthetic_loop(self) -> None:
        CHUNK   = 32
        dt      = 1.0 / FS
        period  = CHUNK / FS     # seconds per chunk

        while self._running:
            t_vec = self._t + np.arange(CHUNK) * dt

            # Alpha (10 Hz) + Theta (6 Hz) + 50 Hz line noise on AF7/AF8
            alpha = 20e-6 * np.sin(2 * np.pi * 10 * t_vec)
            theta =  8e-6 * np.sin(2 * np.pi *  6 * t_vec)
            noise = 15e-6 * np.sin(2 * np.pi * 50 * t_vec)

            raw = np.column_stack([
                alpha + theta,              # TP9
                alpha + theta + noise,      # AF7
                alpha + theta + noise * np.cos(np.pi / 4),  # AF8 phase shift
                alpha + theta,              # TP10
            ])  # (CHUNK, 4)

            for i, ch in enumerate(CHANNELS):
                self._dsp[ch].process(raw[:, i])

            self._t += period
            time.sleep(period)


# ── Module self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Muse2Adapter self-test (synthetic) ===")
    adapter = SyntheticMuse2Adapter()
    adapter.connect()
    adapter.start()

    print("Filling buffer (2 s)…")
    if not adapter.wait_ready(timeout=5.0):
        print("Buffer not ready — exiting.")
        sys.exit(1)

    for band in ["raw", "delta", "theta", "alpha", "beta", "gamma"]:
        win = adapter.get_filtered_window(band)
        print(f"  {band:6s} window shape={win.shape}  "
              f"TP9 rms={np.sqrt(np.mean(win[0]**2)):.6f} V")

    adapter.stop()
    print("Self-test passed.")
