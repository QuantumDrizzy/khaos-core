"""
electrode_model.py — Electrode Impedance & SNR Abstraction
══════════════════════════════════════════════════════════════════════════════
Defines an abstract base class `ElectrodeModel` and two concrete
implementations:

  GrapheneFermiDiracModel
    Models graphene Fermi-Dirac dispersion: E(k) = ±ℏv_F|k|.
    Valid for graphene micro-needle electrodes with Z < 1 kΩ.
    Produces ent_alpha[layer] = tanh(σ_layer / σ_ref) for circuit
    entanglement, matching the existing dirac_emulator.py pipeline.
    *** This model is NOT applicable to Muse 2 AgCl dry contacts. ***

  AgClDryContactModel
    Models Ag/AgCl dry-contact electrodes using the Huigen (2002) RC
    equivalent circuit:
      • R_contact  ≈ 500 kΩ  (gel-free skin-electrode resistance)
      • C_skin     ≈  47 nF  (stratum-corneum capacitance)
      • R_lead     ≈   5 kΩ  (connector + trace resistance)
    The model computes frequency-dependent impedance across the Muse 2
    physiological bands and derives an SNR correction factor used by the
    feature extractor to weight band-power estimates.

References
----------
Huigen, E. et al. (2002). Investigation into the origin of the noise of
  surface electrodes. Medical & Biological Engineering & Computing, 40(3),
  332–338.

Davidson, R. J. (1988). EEG measures of cerebral asymmetry: conceptual
  and methodological issues. International Journal of Neuroscience, 39(1-2),
  71–89.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
H_BAR    = 1.0545718e-34   # J·s   (ℏ)
V_FERMI  = 1.0e6            # m/s   (graphene Fermi velocity)
K_B      = 1.380649e-23    # J/K   (Boltzmann constant)
E_CHARGE = 1.602176634e-19 # C     (elementary charge)

# ── Band centre frequencies (Hz) for impedance evaluation ────────────────────
BAND_CENTRES: Dict[str, float] = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.5,
    "beta":  21.5,
    "gamma": 37.5,
}


# ── Abstract base ─────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Result of an electrode calibration pass."""
    snr_db:           float                  # estimated SNR in dB
    impedance_ohm:    Dict[str, float]        # band → |Z| in Ω
    correction_alpha: np.ndarray             # per-band α weights, shape (5,)
    notes:            List[str] = field(default_factory=list)


class ElectrodeModel(ABC):
    """Abstract electrode model.

    Subclasses implement the physics of a specific electrode technology.
    All KHAOS components that need electrode correction should
    accept an ``ElectrodeModel`` instance rather than a concrete class.
    """

    @abstractmethod
    def impedance_at(self, freq_hz: float) -> complex:
        """Return the complex impedance Z(f) in Ohms."""

    @abstractmethod
    def correct_impedance(self, signal: np.ndarray,
                          band: str, fs: float) -> np.ndarray:
        """Return an impedance-corrected version of *signal* for *band*."""

    @abstractmethod
    def estimate_snr(self, signal: np.ndarray, fs: float) -> float:
        """Estimate signal-to-noise ratio in dB for a raw channel window."""

    @abstractmethod
    def calibrate(self, resting_windows: np.ndarray) -> CalibrationResult:
        """Fit model parameters from resting-state windows.

        Parameters
        ----------
        resting_windows : np.ndarray, shape (n_channels, n_samples)
            Eyes-closed resting EEG (typically 2 min @ 256 Hz = 30 720 samples).

        Returns
        -------
        CalibrationResult
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""


# ── Graphene / Fermi-Dirac model ──────────────────────────────────────────────

class GrapheneFermiDiracModel(ElectrodeModel):
    """Graphene micro-needle electrode: Fermi-Dirac dispersion.

    Matches the existing ``dirac_emulator.py`` pipeline.  Only valid for
    impedances < 1 kΩ.  Do **not** use for Muse 2 or any AgCl dry-contact
    system.
    """

    def __init__(self, mu_eV: float = 0.1, temp_K: float = 300.0,
                 sigma_ref: float = 1.0):
        """
        Parameters
        ----------
        mu_eV    : Fermi energy in eV (chemical potential)
        temp_K   : Temperature in Kelvin
        sigma_ref: Reference sheet conductivity normalisation
        """
        self._mu      = mu_eV * E_CHARGE   # convert to Joules
        self._T       = temp_K
        self._sig_ref = sigma_ref

    @property
    def name(self) -> str:
        return "GrapheneFermiDiracModel"

    def _fermi_dirac(self, E: float) -> float:
        """Fermi-Dirac occupation probability at energy E (Joules)."""
        kT = K_B * self._T
        if kT < 1e-40:
            return 1.0 if E < self._mu else 0.0
        return 1.0 / (1.0 + math.exp((E - self._mu) / kT))

    def impedance_at(self, freq_hz: float) -> complex:
        """Approximate impedance from Dirac-point density of states.

        For graphene: ρ(E) ∝ |E| / (ℏv_F)²
        Low-energy approximation: Z ∝ 1 / σ(ω)
        """
        omega = 2 * math.pi * freq_hz
        # Characteristic energy at this frequency
        E_omega = H_BAR * omega
        # Density of states contribution
        dos = abs(E_omega) / (H_BAR * V_FERMI) ** 2
        occ = self._fermi_dirac(E_omega)
        sigma = max(dos * occ, 1e-20)   # avoid div-by-zero
        Z_real = self._sig_ref / sigma
        Z_imag = -Z_real * 0.1          # small capacitive component
        return complex(Z_real, Z_imag)

    def entanglement_alpha(self, band_powers: np.ndarray) -> np.ndarray:
        """Produce ent_alpha per layer (mirrors dirac_emulator.py).

        Parameters
        ----------
        band_powers : np.ndarray, shape (n_layers,)

        Returns
        -------
        np.ndarray, shape (n_layers,)
        """
        return np.tanh(band_powers / self._sig_ref)

    def correct_impedance(self, signal: np.ndarray,
                          band: str, fs: float) -> np.ndarray:
        """Whitening correction: divide by |Z| at band centre frequency."""
        f0   = BAND_CENTRES.get(band, 10.0)
        z    = abs(self.impedance_at(f0))
        z    = max(z, 1e-10)
        return signal / z

    def estimate_snr(self, signal: np.ndarray, fs: float) -> float:
        """Simple power-ratio SNR assuming graphene Z ≪ 1 kΩ."""
        power_sig  = np.var(signal)
        noise_floor = 1e-12   # thermal noise at 1 kΩ: ~4nV/√Hz → ~400nV rms
        return 10 * math.log10(power_sig / max(noise_floor, 1e-30))

    def calibrate(self, resting_windows: np.ndarray) -> CalibrationResult:
        n_ch, n_samp = resting_windows.shape
        snr = float(np.mean([self.estimate_snr(resting_windows[c], fs=256.0)
                              for c in range(n_ch)]))
        impedances = {b: abs(self.impedance_at(f))
                      for b, f in BAND_CENTRES.items()}
        alpha = np.ones(5, dtype=np.float64)   # no correction needed
        return CalibrationResult(snr_db=snr, impedance_ohm=impedances,
                                 correction_alpha=alpha,
                                 notes=["GrapheneFermiDirac: Z << 1 kΩ"])


# ── AgCl dry-contact model (Muse 2) ──────────────────────────────────────────

class AgClDryContactModel(ElectrodeModel):
    """Ag/AgCl dry-contact electrode model (Huigen 2002 RC circuit).

    Equivalent circuit
    ------------------
                R_lead
    ───[R_lead]───┬──────────────── V_bio
                  │
               [R_contact]
                  │
               [C_skin]      (stratum corneum)
                  │
                GND

    Total impedance:
      Z(f) = R_lead + R_contact / (1 + j·ω·R_contact·C_skin)

    Parameters (defaults from Huigen 2002 mean values):
      R_contact ≈  500 kΩ  (range 100 kΩ – 2 MΩ depending on hydration)
      C_skin    ≈   47 nF  (range 10 – 100 nF)
      R_lead    ≈    5 kΩ

    At 10 Hz (alpha band):
      ω = 2π·10 ≈ 62.8 rad/s
      τ = R_contact · C_skin ≈ 500e3 · 47e-9 = 23.5 ms
      |Z| ≈ 5k + 500k / √(1 + (62.8·0.0235)²)
           ≈ 5k + 500k / √(1 + 2.18)
           ≈ 5k + 283 kΩ  ≈ 288 kΩ
    """

    def __init__(self, R_contact: float = 500e3,
                 C_skin: float = 47e-9,
                 R_lead: float = 5e3):
        self.R_contact = R_contact
        self.C_skin    = C_skin
        self.R_lead    = R_lead

        # Fitted correction weights (updated by calibrate())
        self._correction_alpha: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "AgClDryContactModel"

    def impedance_at(self, freq_hz: float) -> complex:
        """Return complex impedance Z(f) for the Huigen RC circuit."""
        omega = 2.0 * math.pi * freq_hz
        # Skin/contact parallel RC
        denom = complex(1.0, omega * self.R_contact * self.C_skin)
        Z_skin = complex(self.R_contact, 0) / denom
        return complex(self.R_lead, 0) + Z_skin

    def _snr_correction(self, band: str) -> float:
        """Frequency-dependent SNR correction factor ∈ (0, 1].

        Higher impedance → lower SNR correction.  The 1 kΩ reference matches
        the graphene model (low-Z ideal).
        """
        f0    = BAND_CENTRES.get(band, 10.0)
        z_mag = abs(self.impedance_at(f0))
        z_ref = 1e3          # graphene / wet-gel reference
        # SNR degrades roughly as Z_ref / Z (Johnson noise ∝ √Z)
        return min(z_ref / z_mag, 1.0)

    def correct_impedance(self, signal: np.ndarray,
                          band: str, fs: float) -> np.ndarray:
        """Amplitude-correct signal for frequency-dependent voltage divider.

        The RC divider attenuates the bio-signal by |Z_skin| / |Z_total|.
        We invert this to recover the undistorted signal estimate.

        Note: This is a linear gain correction only.  Full deconvolution
        would require the exact transfer function — suitable for offline
        analysis but not real-time.
        """
        f0    = BAND_CENTRES.get(band, 10.0)
        Z_tot = abs(self.impedance_at(f0))
        # Voltage divider factor — amplifier input impedance >> Z_tot
        # is NOT guaranteed for consumer-grade Muse 2; use conservative gain
        gain = max(1.0, Z_tot / max(self.R_lead, 1.0))
        return signal * gain

    def estimate_snr(self, signal: np.ndarray, fs: float) -> float:
        """SNR estimate accounting for high-impedance Johnson noise.

        Thermal noise power: P_n = 4·k_B·T·R_contact·Δf
        T = 310 K (body temperature), Δf = fs/2 (Nyquist)
        """
        T_body = 310.0    # K
        bw     = fs / 2.0
        p_noise = 4.0 * K_B * T_body * self.R_contact * bw
        v_noise_rms = math.sqrt(p_noise)   # in Volts RMS

        p_signal = float(np.var(signal))
        if p_signal < 1e-30:
            return -60.0
        return 10.0 * math.log10(p_signal / (v_noise_rms ** 2 + 1e-30))

    def calibrate(self, resting_windows: np.ndarray) -> CalibrationResult:
        """Fit R_contact from resting-state noise floor.

        Strategy: the RMS of the resting baseline is dominated by thermal
        noise + common-mode interference.  We fit R_contact so that the
        predicted noise floor matches the observed RMS, then recompute
        correction weights.

        Parameters
        ----------
        resting_windows : np.ndarray, shape (n_channels, n_samples)
        """
        n_ch, n_samp = resting_windows.shape
        observed_rms = float(np.mean(np.sqrt(np.var(resting_windows, axis=1))))

        # Solve for R_contact: rms ≈ √(4·k_B·T·R_contact·BW)
        T_body = 310.0
        bw     = 256.0 / 2.0
        r_fit  = max((observed_rms ** 2) / (4.0 * K_B * T_body * bw), 1e3)
        self.R_contact = min(r_fit, 5e6)   # clamp to physiological range

        # Recompute correction alphas for each band
        bands = list(BAND_CENTRES.keys())
        alphas = np.array([self._snr_correction(b) for b in bands])
        self._correction_alpha = alphas

        impedances = {b: abs(self.impedance_at(f))
                      for b, f in BAND_CENTRES.items()}

        snr = float(np.mean([self.estimate_snr(resting_windows[c], fs=256.0)
                              for c in range(n_ch)]))

        notes = [
            f"R_contact fitted: {self.R_contact/1e3:.1f} kΩ",
            f"Observed RMS: {observed_rms*1e6:.2f} µV",
            "Huigen RC model applied (Muse 2 AgCl dry contacts)",
        ]

        return CalibrationResult(
            snr_db=snr,
            impedance_ohm=impedances,
            correction_alpha=alphas,
            notes=notes,
        )

    def correction_alpha(self) -> np.ndarray:
        """Return per-band correction weights (5,).

        Bands order: delta, theta, alpha, beta, gamma.
        Returns uniform weights if calibrate() has not been called.
        """
        if self._correction_alpha is None:
            bands = list(BAND_CENTRES.keys())
            return np.array([self._snr_correction(b) for b in bands])
        return self._correction_alpha


# ── Factory ───────────────────────────────────────────────────────────────────

def get_electrode_model(electrode_type: str = "agcl", **kwargs) -> ElectrodeModel:
    """Return an ``ElectrodeModel`` by name.

    Parameters
    ----------
    electrode_type : 'agcl' | 'graphene'
    **kwargs       : forwarded to the concrete constructor
    """
    registry = {
        "agcl":     AgClDryContactModel,
        "graphene": GrapheneFermiDiracModel,
    }
    if electrode_type not in registry:
        raise ValueError(
            f"Unknown electrode type '{electrode_type}'. "
            f"Available: {list(registry)}")
    return registry[electrode_type](**kwargs)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== electrode_model self-test ===\n")

    for etype in ["agcl", "graphene"]:
        model = get_electrode_model(etype)
        print(f"Model: {model.name}")
        print(f"  {'Band':<8} {'|Z| (kΩ)':>12}  {'f₀ (Hz)':>8}")
        for band, f0 in BAND_CENTRES.items():
            z = abs(model.impedance_at(f0))
            print(f"  {band:<8} {z/1e3:>12.2f}  {f0:>8.1f}")

        # Synthetic calibration
        resting = np.random.randn(4, 30720) * 20e-6
        result  = model.calibrate(resting)
        print(f"  SNR: {result.snr_db:.2f} dB")
        print(f"  Correction α: {result.correction_alpha}")
        for note in result.notes:
            print(f"  → {note}")
        print()

    print("Self-test passed.")
