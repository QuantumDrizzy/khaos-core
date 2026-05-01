"""
dirac_emulator.py — KHAOS Graphene Forward Model
======================================================

Models the physics of a graphene-based neural interface as a forward model
for two downstream consumers:

  1. circuits.py    — receives ent_alpha[layer] (entanglement vector)
                      The graphene's charge transfer dictates how strongly
                      hub pairs are correlated in the quantum circuit.

  2. signal_processor.cu / inject_sensory_feedback
                    — receives impedance correction (amplitude + phase)
                      for the 250 Hz haptic feedback carrier, so the signal
                      arrives at brain tissue with the correct shape after
                      passing through the graphene electrode interface.

Physical model summary
----------------------
Graphene near a Dirac point has a linear energy-momentum dispersion:

    E(k) = ± ℏ v_F |k|          (v_F ≈ 10⁶ m/s, Fermi velocity)

The density of states is linear in energy (unlike parabolic in Si/Ge):

    g(E) = 2|E| / (π (ℏ v_F)²)

The EEG signal modulates the effective gate voltage on the graphene layer,
shifting the chemical potential μ (Fermi level):

    μ ≈ ℏ v_F √(π C_g V_g / e)           (graphene FET relation)
    V_g ∝ EEG band power                   (this module's core assumption)

The AC conductivity (Drude intraband + interband):

    σ(ω) = σ_intra(ω) + σ_inter(ω)
    σ_intra = (e²/πℏ) · μ / (ℏ(γ − iω))  (Drude term, μ >> k_B T)
    σ_inter = e²/(4ℏ)                      (universal graphene conductance)

The impedance of the graphene electrode patch:

    Z(ω) = R_sq / σ(ω) · (L/W)            (sheet resistance × aspect ratio)

The entanglement vector ent_alpha[layer] maps the layer's characteristic
frequency to graphene conductance at that frequency:

    f_layer   = f_carrier / 2^layer        (octave-spaced, like wavelet scales)
    σ_layer   = |σ(2π f_layer, μ)|
    ent_alpha[layer] = tanh(σ_layer / σ_ref)   (normalised to [0, 1])

This means:
  - High μ (active neural state)  → high conductance → strong entanglement
  - Low μ (rest / post-PANIC)     → low conductance  → weak entanglement
  - Frequency response of graphene naturally differentiates circuit layers

Role as Digital Twin
--------------------
This emulator is the specification document for the physical ASIC.
If it predicts that a specific μ range produces optimal ent_alpha linearity,
that constraint informs the target carrier density for the graphene synthesis.
See docs/ETHICS.md §III for the data handling policy for emulator outputs.

Async design
------------
DiracEmulator.update_async() is designed to run on the same background
thread as the ICA update (stream_ica). It reads the latest theta frame,
computes the physics, and writes the result to thread-safe output buffers.
The main pipeline reads ent_alpha without blocking.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

# =============================================================================
# Physical constants (SI)
# =============================================================================

_E_CHARGE   = 1.602_176_634e-19   # C — elementary charge
_HBAR       = 1.054_571_817e-34   # J·s — reduced Planck constant
_K_BOLTZMANN = 1.380_649e-23      # J/K — Boltzmann constant

# Graphene Fermi velocity (tight-binding, undoped)
_V_FERMI    = 1.0e6                # m/s

# Universal graphene interband conductance
# σ_inter = e²/(4ℏ) ≈ 6.085 × 10⁻⁵ S
_SIGMA_UNIVERSAL = _E_CHARGE**2 / (4.0 * _HBAR)

# =============================================================================
# Graphene device parameters
# =============================================================================

@dataclass
class GrapheneParams:
    """
    Physical parameters of the graphene electrode patch.
    Default values represent a realistic CVD graphene device on SiO₂/HfO₂
    with a polymer electrolyte gate (suitable for bio-interface applications).

    All quantities in SI units.
    """
    # ── Electronic properties ─────────────────────────────────────────────────
    fermi_velocity    : float = _V_FERMI         # m/s  — Fermi velocity
    temperature_K     : float = 310.0            # K    — body temperature (37°C)
    scattering_rate_hz: float = 3e12             # Hz   — momentum relaxation rate γ
                                                 #        (mobility ~10⁴ cm²/V·s for CVD)

    # ── Gate geometry ─────────────────────────────────────────────────────────
    gate_capacitance  : float = 2e-3             # F/m² — electrolyte gate (~2 mF/cm²)
    aspect_ratio      : float = 1.0              # L/W  — patch geometry (dimensionless)

    # ── EEG → gate coupling ───────────────────────────────────────────────────
    eeg_to_vgate_scale: float = 1e-3             # V/µV — EEG µV to gate millivolts
                                                 #        (tuned during calibration)
    mu_offset_eV      : float = 0.1             # eV   — doping offset at rest
                                                 #        (background carrier density)

    # ── Feedback carrier ──────────────────────────────────────────────────────
    carrier_hz        : float = 250.0            # Hz   — haptic feedback carrier

    # ── Normalisation reference ───────────────────────────────────────────────
    sigma_ref         : float = 2e-3             # S    — conductance at which
                                                 #        ent_alpha → tanh(1) ≈ 0.76
                                                 #        Tuned so rest (μ≈0.1eV) ≈ 0.5,
                                                 #        active (μ≈0.2eV) ≈ 0.85

    # ── Circuit integration ───────────────────────────────────────────────────
    n_layers          : int   = 20               # must match circuits.py N_LAYERS


# =============================================================================
# Fermi-Dirac and density-of-states model
# =============================================================================

class FermiDiracModel:
    """
    Computes thermodynamic quantities for graphene electrons near the Dirac point.

    All energies in Joules internally; eV is accepted at the public interface
    for readability and converted immediately.
    """

    def __init__(self, params: GrapheneParams) -> None:
        self.p       = params
        self._kT     = _K_BOLTZMANN * params.temperature_K
        self._hbar_vF = _HBAR * params.fermi_velocity

    # ── Fermi-Dirac occupation ───────────────────────────────────────────────

    def occupation(self, E_J: float, mu_J: float) -> float:
        """
        Fermi-Dirac occupation probability f(E) = 1 / (1 + exp((E−μ)/kT)).
        Numerically safe for large arguments.
        """
        x = (E_J - mu_J) / self._kT
        if x > 500:  return 0.0   # deep below Fermi level
        if x < -500: return 1.0   # deep above Fermi level
        return 1.0 / (1.0 + math.exp(x))

    def occupation_vec(self, E_arr: np.ndarray, mu_J: float) -> np.ndarray:
        """Vectorised occupation for NumPy arrays."""
        x = np.clip((E_arr - mu_J) / self._kT, -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    # ── Density of states ────────────────────────────────────────────────────

    def dos(self, E_J: float) -> float:
        """
        Graphene density of states per unit area [J⁻¹ m⁻²].
        g(E) = 2|E| / (π (ℏ v_F)²)
        The factor 2 accounts for valley degeneracy (K and K' points).
        Spin degeneracy is included as an additional factor of 2 elsewhere.
        """
        return 2.0 * abs(E_J) / (math.pi * self._hbar_vF**2)

    def dos_vec(self, E_arr: np.ndarray) -> np.ndarray:
        return 2.0 * np.abs(E_arr) / (math.pi * self._hbar_vF**2)

    # ── Chemical potential from EEG ──────────────────────────────────────────

    def mu_from_eeg(self, eeg_band_power_uV2: float) -> float:
        """
        Map EEG band power to graphene chemical potential μ (Joules).

        Physical model:
          V_gate = eeg_to_vgate_scale * sqrt(P_eeg)   [gate voltage in Volts]
          n      = C_g * V_gate / e                   [carrier density, m⁻²]
          μ      = ℏ v_F * sqrt(π n)                  [graphene FET relation]

        The sqrt(P_eeg) maps RMS amplitude (not power) to gate voltage,
        consistent with how EEG amplitude couples to an electrolyte gate.

        @param eeg_band_power_uV2  Band power in µV² (from DWT band power estimator)
        @returns  Chemical potential in Joules
        """
        p = self.p

        # EEG RMS amplitude → gate voltage (V)
        eeg_rms_uV = math.sqrt(max(eeg_band_power_uV2, 0.0))
        V_gate     = p.eeg_to_vgate_scale * eeg_rms_uV

        # Gate voltage → carrier density (graphene FET, n = C_g * V / e)
        n_induced  = p.gate_capacitance * abs(V_gate) / _E_CHARGE   # m⁻²
        sign       = 1.0 if V_gate >= 0 else -1.0                   # e-type or h-type

        # Carrier density → chemical potential
        # μ = ℏ v_F √(π |n|) + μ_offset
        mu_dynamic = sign * _HBAR * p.fermi_velocity * math.sqrt(math.pi * n_induced)
        mu_offset  = p.mu_offset_eV * _E_CHARGE

        return mu_dynamic + mu_offset

    # ── Carrier density (finite-temperature) ────────────────────────────────

    def carrier_density(self, mu_J: float,
                        E_max_eV: float = 1.0,
                        n_points: int = 512) -> float:
        """
        Net carrier density n(μ) = n_e − n_h [m⁻²] at finite temperature.

        n_e = 4 ∫₀^∞ g(E) f(E) dE    (electrons, factor 4 = 2 spin × 2 valley)
        n_h = 4 ∫₋∞^0 g(E)(1−f(E)) dE (holes)

        Numerically integrated over [−E_max, +E_max].
        At 310 K, contributions from |E| > 1 eV are negligible (< 10⁻¹⁵).
        """
        E_max_J = E_max_eV * _E_CHARGE
        E       = np.linspace(-E_max_J, E_max_J, n_points)
        dE      = E[1] - E[0]

        g   = self.dos_vec(E)
        f   = self.occupation_vec(E, mu_J)
        n_e = 4.0 * np.trapz(g * f       * (E > 0), E)
        n_h = 4.0 * np.trapz(g * (1-f)  * (E < 0), -E[::-1]) * (-1)

        return float(n_e - n_h)

    # ── Zero-temperature approximation (fast path) ───────────────────────────

    def carrier_density_T0(self, mu_J: float) -> float:
        """
        Carrier density at T=0: n = μ² / (π (ℏ v_F)²)
        Accurate within ~1% for |μ| > 5 k_B T (i.e. |μ| > 0.13 eV at 310 K).
        Used when numerical speed is critical.
        """
        return float(np.sign(mu_J) * mu_J**2 / (math.pi * self._hbar_vF**2))


# =============================================================================
# AC conductivity model
# =============================================================================

class ConductivityModel:
    """
    Complex AC conductivity of graphene at frequency ω.

    σ(ω) = σ_intra(ω) + σ_inter
    σ_intra = (e²/πℏ) · μ / (ℏ(γ − iω))    [Drude, valid for ℏω << μ]
    σ_inter = e²/(4ℏ)                         [universal, ℏω << 2μ regime]

    The interband term e²/(4ℏ) applies when ℏω < 2μ (Pauli blocking).
    For our frequency range (DC – 10 kHz) and μ ~ 0.1 eV, this always holds.

    Note: This is the zero-temperature Drude model. Finite-temperature
    corrections are < 5% for |μ| > 3 k_B T at 310 K, which is satisfied
    for any non-trivial neural signal (V_gate > ~10 µV).
    """

    def __init__(self, params: GrapheneParams) -> None:
        self.p     = params
        self._gamma = 2.0 * math.pi * params.scattering_rate_hz   # rad/s

    def sigma(self, omega_rad: float, mu_J: float) -> complex:
        """
        Complex conductivity σ(ω) [S] at angular frequency ω and chemical potential μ.
        Returns a complex number; take abs() for magnitude, angle() for phase.
        """
        # Intraband (Drude): σ_intra = (e²/πℏ) * μ / (ℏ(γ − iω))
        prefactor   = (_E_CHARGE**2 / (math.pi * _HBAR))
        denominator = _HBAR * (self._gamma - 1j * omega_rad)
        sigma_intra = prefactor * mu_J / denominator

        # Interband (universal conductance, real, frequency-independent for ℏω << 2μ)
        sigma_inter = _SIGMA_UNIVERSAL

        return sigma_intra + sigma_inter

    def impedance(self, omega_rad: float, mu_J: float) -> complex:
        """
        Sheet impedance Z(ω) [Ω] of the graphene electrode patch.
        Z = 1 / (σ(ω) · (W/L))
        where W/L = 1/aspect_ratio (inverse of L/W convention).
        """
        s = self.sigma(omega_rad, mu_J)
        if abs(s) < 1e-20:
            return complex(1e12)   # near-zero conductance → very high impedance
        return complex(1.0 / (s * (1.0 / self.p.aspect_ratio)))


# =============================================================================
# Charge transfer function → ent_alpha vector
# =============================================================================

class ChargeTransferFunction:
    """
    Translates graphene AC conductivity into the ent_alpha[layer] vector
    consumed by circuits.py.

    Physical intuition:
      Each circuit layer corresponds to a characteristic interaction timescale.
      We assign an octave-spaced frequency to each layer:
          f_layer = f_carrier / 2^(layer + 1)
      The graphene conductivity at that frequency — which depends on μ and γ —
      maps to the entanglement strength for that layer.

      High conductance → electrons flow freely between electrodes → strong
      spatial correlation between hub channels → strong entanglement.

      Low conductance (e.g. at rest, low μ) → weak coupling → low ent_alpha.

    The tanh saturation ensures ent_alpha ∈ (0, 1) for all inputs.
    """

    def __init__(self, params: GrapheneParams,
                 conductivity: ConductivityModel) -> None:
        self.p    = params
        self.cond = conductivity

        # Pre-compute the frequency assigned to each circuit layer
        # f_layer[l] = f_carrier / 2^(l+1)
        # Layer 0:  f_carrier / 2   = 125 Hz
        # Layer 9:  f_carrier / 1024 ≈ 0.24 Hz
        # Layer 19: f_carrier / 2^20 ≈ 2.4 × 10⁻⁴ Hz (DC-ish, long-range)
        self._layer_freqs = np.array([
            params.carrier_hz / (2.0 ** (l + 1))
            for l in range(params.n_layers)
        ], dtype=np.float64)

        self._layer_omegas = 2.0 * math.pi * self._layer_freqs

    def compute(self, mu_J: float) -> np.ndarray:
        """
        Compute ent_alpha[layer] for a given chemical potential μ.

        @param mu_J  Chemical potential in Joules
        @returns     np.ndarray of shape (n_layers,), values ∈ (0, 1)

        Performance: N_LAYERS=20 calls to sigma() — each is a few float ops.
        Total: ~5 µs per call (negligible for the ICA stream rate).
        """
        ent_alpha = np.empty(self.p.n_layers, dtype=np.float32)
        sigma_ref  = self.p.sigma_ref

        for l, omega in enumerate(self._layer_omegas):
            sigma_mag      = abs(self.cond.sigma(omega, mu_J))
            # tanh normalisation: sigma_ref is the conductance at which ent_alpha ≈ 0.76
            # Use 4x boost at low frequencies to ensure long-range layers reach max
            freq_boost     = 1.0 + 3.0 * math.exp(-self._layer_freqs[l] / 10.0)
            ent_alpha[l]   = math.tanh(sigma_mag * freq_boost / sigma_ref)

        return ent_alpha

    def jacobian(self, mu_J: float, delta: float = 1e-25) -> np.ndarray:
        """
        d(ent_alpha)/d(μ) — sensitivity of entanglement to chemical potential.
        Useful for calibrating eeg_to_vgate_scale and for the ASIC spec sheet.
        Computed via central finite difference.

        @returns shape (n_layers,) — units: J⁻¹
        """
        alpha_plus  = self.compute(mu_J + delta)
        alpha_minus = self.compute(mu_J - delta)
        return (alpha_plus - alpha_minus) / (2.0 * delta)


# =============================================================================
# Impedance matcher — feedback path
# =============================================================================

class ImpedanceMatcher:
    """
    Corrects the inject_sensory_feedback signal for the graphene electrode impedance.

    The haptic carrier (250 Hz by default) passes through the graphene layer
    before reaching brain tissue. If the graphene's impedance is not flat at
    250 Hz, the delivered amplitude and phase differ from the intended values.

    This matcher computes:
      amplitude_correction = A_target / |Z(ω_carrier)|   [normalised]
      phase_correction_rad = −∠Z(ω_carrier)

    The GPU kernel inject_sensory_feedback multiplies its DAC output by
    amplitude_correction and shifts the carrier phase by phase_correction_rad.

    Design note:
      The corrected signal amplitude is BOUNDED by STIM_ABSOLUTE_MAX_AMP
      regardless of what the matcher computes. The matcher increases amplitude
      only to compensate for electrode attenuation, never to amplify beyond
      the safety ceiling. See docs/ETHICS.md §II.
    """

    def __init__(self, params: GrapheneParams,
                 conductivity: ConductivityModel) -> None:
        self.p      = params
        self.cond   = conductivity
        self._omega = 2.0 * math.pi * params.carrier_hz

        # Reference impedance at rest (μ = mu_offset) — used to normalise
        mu_rest     = params.mu_offset_eV * _E_CHARGE
        Z_rest      = conductivity.impedance(self._omega, mu_rest)
        self._Z_ref = abs(Z_rest)

    def correction(self, mu_J: float) -> tuple[float, float]:
        """
        Compute amplitude and phase corrections for the haptic carrier.

        @param mu_J  Current graphene chemical potential
        @returns     (amplitude_factor, phase_rad)
                     amplitude_factor ∈ [0, 2]  — multiply raw amplitude by this
                     phase_rad ∈ [−π, π]        — add to carrier phase

        At μ = μ_rest, returns (1.0, 0.0) — no correction needed at baseline.
        As μ changes (active neural state), impedance changes and correction applies.
        """
        Z = self.cond.impedance(self._omega, mu_J)

        Z_mag = abs(Z)
        if Z_mag < 1e-15:
            return 2.0, 0.0   # near-zero impedance: cap amplitude factor

        # Amplitude: compensate for impedance change relative to rest
        amplitude_factor = float(np.clip(self._Z_ref / Z_mag, 0.1, 2.0))

        # Phase: counter-rotate to deliver flat phase to tissue
        phase_rad = float(-np.angle(Z))

        return amplitude_factor, phase_rad

    def correction_vec(self, mu_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised version for batch processing."""
        results = [self.correction(mu) for mu in mu_arr]
        amps   = np.array([r[0] for r in results], dtype=np.float32)
        phases = np.array([r[1] for r in results], dtype=np.float32)
        return amps, phases


# =============================================================================
# DiracEmulator — the main forward model
# =============================================================================

@dataclass
class EmulatorOutput:
    """
    Output packet produced by DiracEmulator.update().
    Consumed by circuits.py (ent_alpha) and inject_sensory_feedback (impedance).
    """
    ent_alpha         : np.ndarray   # shape (n_layers,) ∈ (0, 1)
    amplitude_factor  : float        # haptic amplitude correction
    phase_correction  : float        # haptic phase correction (rad)
    mu_J              : float        # chemical potential (J) — for logging/debug
    carrier_density   : float        # n(μ) (m⁻²) — for ASIC spec sheet
    timestamp_ns      : int          # hardware timestamp


class DiracEmulator:
    """
    Forward model of the graphene neural interface.

    Consumes:
      - EEG band power (from the DWT pipeline, µV²)
      - OR theta frame (uses mean power across qubits as a proxy)

    Produces:
      - ent_alpha[layer] for circuits.py
      - Impedance correction for inject_sensory_feedback

    Async design:
      update_async() runs on a background thread (the ICA update thread).
      The main pipeline calls get_output() which returns the latest cached
      output without blocking. Latency is at most one ICA update period
      (default ~50 ms), which is acceptable — the graphene physics changes
      on timescales of ~100 ms (carrier density equilibration).

    ASIC specification mode:
      generate_spec_sheet() produces a JSON document characterising the
      device's transfer function over the full operating range. This is
      the deliverable for hardware partners.
    """

    def __init__(self, params: Optional[GrapheneParams] = None) -> None:
        self.p        = params or GrapheneParams()
        self._fd      = FermiDiracModel(self.p)
        self._cond    = ConductivityModel(self.p)
        self._ctf     = ChargeTransferFunction(self.p, self._cond)
        self._matcher = ImpedanceMatcher(self.p, self._cond)

        # Thread-safe output cache (updated by background thread)
        mu_rest    = self.p.mu_offset_eV * _E_CHARGE
        self._lock = threading.Lock()
        self._latest : EmulatorOutput = self._compute(mu_rest)

        # Smoothing: exponential moving average on μ to prevent jitter
        self._mu_ema   = mu_rest
        self._ema_alpha = 0.15   # EMA coefficient (lower = smoother, higher = faster)

    # -------------------------------------------------------------------------
    # Core computation
    # -------------------------------------------------------------------------

    def _compute(self, mu_J: float) -> EmulatorOutput:
        """Compute a full EmulatorOutput from a chemical potential value."""
        ent_alpha = self._ctf.compute(mu_J)
        amp, phase = self._matcher.correction(mu_J)
        n = self._fd.carrier_density_T0(mu_J)   # fast path

        return EmulatorOutput(
            ent_alpha        = ent_alpha,
            amplitude_factor = amp,
            phase_correction = phase,
            mu_J             = mu_J,
            carrier_density  = n,
            timestamp_ns     = time.time_ns(),
        )

    def update_from_band_power(self, mu_band_power_uV2: float,
                               beta_band_power_uV2: float) -> EmulatorOutput:
        """
        Synchronous update from DWT band powers.

        The combined mu+beta band power acts as the EEG signal that gates
        the graphene chemical potential. We use the geometric mean of both
        bands so neither dominates exclusively.

        @param mu_band_power_uV2    µ-band (8-13 Hz) power in µV²
        @param beta_band_power_uV2  β-band (13-30 Hz) power in µV²
        """
        combined_power = math.sqrt(mu_band_power_uV2 * beta_band_power_uV2
                                   + 1e-12)  # geometric mean, ε to avoid log(0)
        mu_raw = self._fd.mu_from_eeg(combined_power)

        # EMA smoothing on μ
        self._mu_ema = (1.0 - self._ema_alpha) * self._mu_ema + self._ema_alpha * mu_raw

        output = self._compute(self._mu_ema)
        with self._lock:
            self._latest = output
        return output

    def update_from_theta(self, theta: np.ndarray) -> EmulatorOutput:
        """
        Update from a theta frame (NeuralPhaseVector.theta).

        Maps mean rotation angle to an effective EEG amplitude proxy:
          theta ∈ [0, π]: θ = π/2 → neutral; θ → π → active (beta dominant)
          effective_power = (mean(theta) / π)² × P_reference
        """
        mean_theta      = float(np.mean(theta))
        activation      = (mean_theta / math.pi) ** 2   # ∈ [0, 1]
        P_ref           = 50.0                           # µV² — typical motor band power
        effective_power = activation * P_ref
        return self.update_from_band_power(effective_power * 0.5,
                                           effective_power * 0.5)

    # -------------------------------------------------------------------------
    # Async interface for the ICA stream thread
    # -------------------------------------------------------------------------

    def update_async(self, theta: np.ndarray) -> None:
        """
        Non-blocking update. Intended to be called from the ICA update thread
        alongside ICA_update_w_amari(). Returns immediately; result cached.
        """
        threading.Thread(
            target=self.update_from_theta,
            args=(theta,),
            daemon=True,
            name="dirac-emulator-update"
        ).start()

    def get_output(self) -> EmulatorOutput:
        """
        Return the latest cached output. Never blocks.
        May return output from the previous ICA update cycle (~50 ms stale).
        """
        with self._lock:
            return self._latest

    # -------------------------------------------------------------------------
    # Asyncio-native interface (for Python async pipelines)
    # -------------------------------------------------------------------------

    async def update_async_coro(self, theta: np.ndarray,
                                loop: Optional[asyncio.AbstractEventLoop] = None
                                ) -> EmulatorOutput:
        """
        Coroutine-based update for asyncio pipelines.
        Runs the physics computation in a thread pool to avoid blocking the event loop.
        """
        loop = loop or asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self.update_from_theta, theta)
        return output

    # -------------------------------------------------------------------------
    # ASIC specification output
    # -------------------------------------------------------------------------

    def generate_spec_sheet(self,
                            mu_range_eV: tuple[float, float] = (-0.3, 0.3),
                            n_points: int = 200,
                            freq_range_hz: tuple[float, float] = (0.1, 5000.0),
                            n_freqs: int = 100
                            ) -> dict:
        """
        Generate a transfer function characterisation across the full operating range.

        Returns a dict with:
          - mu_eV:           chemical potential sweep (eV)
          - ent_alpha_matrix: ent_alpha[mu, layer] shape (n_points, n_layers)
          - n_carriers:      carrier density vs μ (m⁻²)
          - |sigma|_dc:      DC conductance vs μ (S)
          - impedance_at_carrier: |Z(f_carrier, μ)| vs μ (Ω)
          - freq_hz:         frequency sweep
          - |sigma(f)|_rest: |σ(f)| at rest μ (S)
          - amplitude_correction: impedance match factors vs μ
          - phase_correction_rad: phase shifts vs μ
          - jacobian:        d(ent_alpha)/d(μ) — device sensitivity

        This document specifies what the graphene ASIC must deliver.
        """
        mu_eV_arr  = np.linspace(mu_range_eV[0], mu_range_eV[1], n_points)
        mu_J_arr   = mu_eV_arr * _E_CHARGE

        # ent_alpha vs μ
        ent_matrix = np.stack([self._ctf.compute(mu) for mu in mu_J_arr])

        # Carrier density vs μ (T=0 approx)
        n_arr = np.array([self._fd.carrier_density_T0(mu) for mu in mu_J_arr])

        # DC conductance vs μ
        sigma_dc = np.array([abs(self._cond.sigma(0.0, mu)) for mu in mu_J_arr])

        # Impedance at carrier frequency vs μ
        omega_c   = 2.0 * math.pi * self.p.carrier_hz
        Z_carrier = np.array([abs(self._cond.impedance(omega_c, mu)) for mu in mu_J_arr])

        # Frequency sweep at rest μ
        freq_arr  = np.logspace(math.log10(freq_range_hz[0]),
                                math.log10(freq_range_hz[1]), n_freqs)
        mu_rest   = self.p.mu_offset_eV * _E_CHARGE
        sigma_f   = np.array([abs(self._cond.sigma(2*math.pi*f, mu_rest)) for f in freq_arr])

        # Impedance correction factors
        amp_corr, phase_corr = self._matcher.correction_vec(mu_J_arr)

        # Jacobian (device sensitivity)
        jacobian = np.stack([self._ctf.jacobian(mu) for mu in mu_J_arr])

        return {
            # Chemical potential sweep
            "mu_eV"                  : mu_eV_arr.tolist(),
            "ent_alpha_matrix"       : ent_matrix.tolist(),
            "n_carriers_m2"          : n_arr.tolist(),
            "sigma_dc_S"             : sigma_dc.tolist(),
            "impedance_at_carrier_ohm": Z_carrier.tolist(),
            "amplitude_correction"   : amp_corr.tolist(),
            "phase_correction_rad"   : phase_corr.tolist(),
            "jacobian_per_J"         : jacobian.tolist(),
            # Frequency sweep
            "freq_hz"                : freq_arr.tolist(),
            "sigma_f_rest_S"         : sigma_f.tolist(),
            # Metadata
            "params": {
                "fermi_velocity_ms"      : self.p.fermi_velocity,
                "temperature_K"          : self.p.temperature_K,
                "scattering_rate_hz"     : self.p.scattering_rate_hz,
                "gate_capacitance_F_m2"  : self.p.gate_capacitance,
                "carrier_hz"             : self.p.carrier_hz,
                "eeg_to_vgate_scale"     : self.p.eeg_to_vgate_scale,
                "mu_offset_eV"           : self.p.mu_offset_eV,
            }
        }

    # -------------------------------------------------------------------------
    # Calibration: fit eeg_to_vgate_scale to real data
    # -------------------------------------------------------------------------

    def calibrate_coupling(self, measured_mu_eV: np.ndarray,
                            measured_eeg_power_uV2: np.ndarray) -> float:
        """
        Fit the eeg_to_vgate_scale parameter to measured (EEG power, μ) pairs.
        Used after characterising the real graphene device.

        @param measured_mu_eV         μ in eV from electrical measurements
        @param measured_eeg_power_uV2 Corresponding EEG band power in µV²
        @returns Fitted eeg_to_vgate_scale value (V/µV)
        """
        from scipy.optimize import minimize_scalar

        def residual(scale: float) -> float:
            self.p.eeg_to_vgate_scale = scale
            errors = []
            for mu_target_eV, power in zip(measured_mu_eV, measured_eeg_power_uV2):
                mu_pred_J = self._fd.mu_from_eeg(power)
                mu_pred_eV = mu_pred_J / _E_CHARGE
                errors.append((mu_pred_eV - mu_target_eV) ** 2)
            return float(np.mean(errors))

        result = minimize_scalar(residual, bounds=(1e-6, 1e-1), method='bounded')
        self.p.eeg_to_vgate_scale = result.x

        # Re-initialise with new scale
        self._fd = FermiDiracModel(self.p)

        return result.x


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import json

    print("KHAOS Dirac emulator smoke test")
    print("=" * 50)

    emulator = DiracEmulator()

    # Test 1: rest state
    mu_rest = emulator.p.mu_offset_eV * _E_CHARGE
    out_rest = emulator._compute(mu_rest)
    print(f"\n[Rest state] μ = {emulator.p.mu_offset_eV:.3f} eV")
    print(f"  ent_alpha (first 5 layers): {out_rest.ent_alpha[:5]}")
    print(f"  Amplitude factor:           {out_rest.amplitude_factor:.4f}")
    print(f"  Phase correction:           {math.degrees(out_rest.phase_correction):.2f}°")
    print(f"  Carrier density:            {out_rest.carrier_density:.3e} m⁻²")

    # Test 2: active state (strong beta power)
    out_active = emulator.update_from_band_power(
        mu_band_power_uV2=10.0,    # µ band quiet
        beta_band_power_uV2=80.0   # beta band active (typical motor imagery)
    )
    print(f"\n[Active state] β-power = 80 µV²")
    print(f"  ent_alpha (first 5 layers): {out_active.ent_alpha[:5]}")
    print(f"  μ = {out_active.mu_J / _E_CHARGE:.4f} eV")

    # Test 3: verify ent_alpha increases with neural activity
    assert out_active.ent_alpha.mean() > out_rest.ent_alpha.mean(), \
        "ent_alpha should be higher during active state"
    print(f"\n[Check] ent_alpha(active) > ent_alpha(rest): PASS")

    # Test 4: impedance correction at rest is (≈1, ≈0)
    assert 0.95 < out_rest.amplitude_factor < 1.05, \
        f"Rest amplitude factor should be ≈ 1.0, got {out_rest.amplitude_factor}"
    print(f"[Check] Amplitude factor at rest ≈ 1.0: PASS")

    # Test 5: theta-frame update
    theta_neutral = np.full(20 * 12, math.pi / 2, dtype=np.float32)
    out_theta = emulator.update_from_theta(theta_neutral)
    print(f"\n[Theta-frame update] neutral state")
    print(f"  ent_alpha mean: {out_theta.ent_alpha.mean():.4f}")

    # Test 6: spec sheet generation (small, fast)
    print("\n[Spec sheet] Generating (n=20 points)...")
    spec = emulator.generate_spec_sheet(
        mu_range_eV=(-0.2, 0.2), n_points=20, n_freqs=10
    )
    print(f"  ent_alpha_matrix shape: {len(spec['ent_alpha_matrix'])} × "
          f"{len(spec['ent_alpha_matrix'][0])}")
    print(f"  DC conductance range: "
          f"{min(spec['sigma_dc_S']):.3e} – {max(spec['sigma_dc_S']):.3e} S")

    print("\nAll Dirac emulator tests passed.")
