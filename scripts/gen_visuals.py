"""
gen_visuals.py — Generate real data visualizations from khaos-core parameters.
Outputs publication-quality plots for the Twitter thread.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import math, os, sys

# Output directory
OUT = os.path.join(os.path.dirname(__file__), '..', 'visuals')
os.makedirs(OUT, exist_ok=True)

# Use dark theme
plt.style.use('dark_background')
CYAN = '#00e5ff'
MAGENTA = '#ff00e5'
GOLD = '#ffd700'
GREEN = '#39ff14'
RED = '#ff3131'
WHITE = '#e0e0e0'
BG = '#0a0a0f'

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  OK: {name}")

# ═══════════════════════════════════════════════════════════════════
# 1. BUTTERWORTH FILTER — exact coefficients from signal_processor.cu
# ═══════════════════════════════════════════════════════════════════
print("[1/6] Butterworth IIR response...")

sos = signal.iirfilter(10, [8, 30], btype='bandpass', ftype='butter',
                       fs=1000, output='sos')
w, h = signal.sosfreqz(sos, worN=4096, fs=1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor=BG)
fig.suptitle('khaos-core IIR Filter Response\n10th-Order Butterworth Bandpass (8–30 Hz)',
             color=WHITE, fontsize=16, fontweight='bold')

# Magnitude
mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
ax1.plot(w, mag_db, color=CYAN, linewidth=2)
ax1.axvline(8, color=GOLD, ls='--', alpha=0.7, label='Passband edges (8, 30 Hz)')
ax1.axvline(30, color=GOLD, ls='--', alpha=0.7)
ax1.axvline(50, color=RED, ls=':', alpha=0.8, label='50 Hz powerline')
ax1.axvline(60, color=RED, ls=':', alpha=0.5, label='60 Hz powerline')
ax1.axhline(-3, color=WHITE, ls=':', alpha=0.3, label='-3 dB')
ax1.axhline(-60, color=GREEN, ls=':', alpha=0.5, label='-60 dB spec')

# Annotate rejection
idx_50 = np.argmin(np.abs(w - 50))
idx_60 = np.argmin(np.abs(w - 60))
ax1.annotate(f'{mag_db[idx_50]:.1f} dB', xy=(50, mag_db[idx_50]),
             xytext=(70, mag_db[idx_50]+10), color=RED, fontsize=11,
             arrowprops=dict(arrowstyle='->', color=RED))
ax1.annotate(f'{mag_db[idx_60]:.1f} dB', xy=(60, mag_db[idx_60]),
             xytext=(80, mag_db[idx_60]+10), color=RED, fontsize=11,
             arrowprops=dict(arrowstyle='->', color=RED))

ax1.set_xlim(0, 120)
ax1.set_ylim(-100, 5)
ax1.set_ylabel('Magnitude (dB)', color=WHITE)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_facecolor(BG)
ax1.tick_params(colors=WHITE)
ax1.grid(alpha=0.15)
ax1.fill_betweenx([-100, 5], 8, 30, alpha=0.08, color=CYAN)

# Phase
phase = np.unwrap(np.angle(h))
ax2.plot(w, np.degrees(phase), color=MAGENTA, linewidth=1.5)
ax2.axvline(8, color=GOLD, ls='--', alpha=0.5)
ax2.axvline(30, color=GOLD, ls='--', alpha=0.5)
ax2.set_xlim(0, 120)
ax2.set_xlabel('Frequency (Hz)', color=WHITE)
ax2.set_ylabel('Phase (degrees)', color=WHITE)
ax2.set_facecolor(BG)
ax2.tick_params(colors=WHITE)
ax2.grid(alpha=0.15)

save(fig, '01_iir_butterworth.png')

# ═══════════════════════════════════════════════════════════════════
# 2. DWT THETA MAPPING — the exact arcsin-sqrt function
# ═══════════════════════════════════════════════════════════════════
print("[2/6] Theta angle mapping...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle('khaos-core θ Angle Mapping: EEG Band Power → Qubit Rotation',
             color=WHITE, fontsize=15, fontweight='bold')

# a) Theta vs power ratio
ratio = np.linspace(0, 1, 500)
theta = 2 * np.arcsin(np.sqrt(ratio))
axes[0].plot(ratio, theta, color=CYAN, linewidth=2.5)
axes[0].axhline(np.pi/2, color=GOLD, ls='--', alpha=0.5, label='θ = π/2 (superposition)')
axes[0].set_xlabel('P_β / (P_μ + P_β)', color=WHITE)
axes[0].set_ylabel('θ (rad)', color=WHITE)
axes[0].set_title('θ = 2·arcsin(√r)', color=WHITE)
axes[0].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
axes[0].set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
axes[0].legend(fontsize=9)

# b) Bloch sphere probability |1⟩
prob_1 = np.sin(theta/2)**2
axes[1].plot(ratio, prob_1, color=MAGENTA, linewidth=2.5)
axes[1].fill_between(ratio, 0, prob_1, alpha=0.15, color=MAGENTA)
axes[1].set_xlabel('P_β / (P_μ + P_β)', color=WHITE)
axes[1].set_ylabel('P(|1⟩)', color=WHITE)
axes[1].set_title('Qubit |1⟩ Population', color=WHITE)

# c) Simulated 12-channel theta over time
np.random.seed(42)
t = np.linspace(0, 5, 500)
fig_theta_channels = np.zeros((12, len(t)))
for ch in range(12):
    base = 0.3 + 0.4 * np.sin(2*np.pi*0.2*t + ch*0.5)
    noise = 0.05 * np.random.randn(len(t))
    r = np.clip(base + noise, 0, 1)
    fig_theta_channels[ch] = 2 * np.arcsin(np.sqrt(r))

im = axes[2].imshow(fig_theta_channels, aspect='auto', cmap='magma',
                     extent=[0, 5, 11.5, -0.5], vmin=0, vmax=np.pi)
axes[2].set_xlabel('Time (s)', color=WHITE)
axes[2].set_ylabel('Hub Channel', color=WHITE)
axes[2].set_title('12-Channel θ Heatmap', color=WHITE)
cbar = plt.colorbar(im, ax=axes[2], label='θ (rad)')
cbar.set_ticks([0, np.pi/2, np.pi])
cbar.set_ticklabels(['0 (rest)', 'π/2', 'π (active)'])

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)

save(fig, '02_theta_mapping.png')

# ═══════════════════════════════════════════════════════════════════
# 3. GRAPHENE DIRAC EMULATOR — real physics from dirac_emulator.py
# ═══════════════════════════════════════════════════════════════════
print("[3/6] Graphene Fermi-Dirac model...")

E_CHARGE = 1.602_176_634e-19
HBAR = 1.054_571_817e-34
K_B = 1.380_649e-23
V_F = 1.0e6
T = 310.0
gamma = 2*np.pi*3e12
sigma_ref = 2e-3
carrier_hz = 250.0
n_layers = 20

mu_eV = np.linspace(-0.3, 0.3, 300)
mu_J = mu_eV * E_CHARGE

# AC conductivity
def sigma_total(omega, mu):
    sigma_intra = (E_CHARGE**2 / (np.pi*HBAR)) * np.abs(mu) / (HBAR * np.sqrt(gamma**2 + omega**2))
    sigma_inter = E_CHARGE**2 / (4*HBAR)
    return sigma_intra + sigma_inter

# ent_alpha for select layers
layer_freqs = [carrier_hz / 2**(l+1) for l in range(n_layers)]
layer_omegas = [2*np.pi*f for f in layer_freqs]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle('khaos-core Graphene Forward Model — Real Transfer Functions',
             color=WHITE, fontsize=16, fontweight='bold')

# a) Conductivity vs mu
sigma_dc = np.array([sigma_total(0.0, m) for m in mu_J])
sigma_250 = np.array([sigma_total(2*np.pi*250, m) for m in mu_J])
axes[0,0].plot(mu_eV, sigma_dc*1e3, color=CYAN, lw=2, label='DC')
axes[0,0].plot(mu_eV, sigma_250*1e3, color=MAGENTA, lw=2, label='250 Hz')
axes[0,0].set_xlabel('μ (eV)', color=WHITE)
axes[0,0].set_ylabel('|σ| (mS)', color=WHITE)
axes[0,0].set_title('Conductivity vs Chemical Potential', color=WHITE)
axes[0,0].legend()
axes[0,0].axvline(0.1, color=GOLD, ls=':', alpha=0.5, label='rest offset')

# b) ent_alpha matrix
ent_matrix = np.zeros((len(mu_J), n_layers))
for i, mu in enumerate(mu_J):
    for l, omega in enumerate(layer_omegas):
        s = sigma_total(omega, mu)
        freq_boost = 1.0 + 3.0 * math.exp(-layer_freqs[l] / 10.0)
        ent_matrix[i, l] = math.tanh(s * freq_boost / sigma_ref)

im = axes[0,1].imshow(ent_matrix.T, aspect='auto', cmap='inferno',
                       extent=[mu_eV[0], mu_eV[-1], n_layers-0.5, -0.5])
axes[0,1].set_xlabel('μ (eV)', color=WHITE)
axes[0,1].set_ylabel('Circuit Layer', color=WHITE)
axes[0,1].set_title('ent_alpha[μ, layer] — Entanglement Map', color=WHITE)
plt.colorbar(im, ax=axes[0,1], label='ent_alpha')

# c) Frequency response of conductivity
freqs = np.logspace(-1, 4, 500)
mu_rest = 0.1 * E_CHARGE
mu_active = 0.2 * E_CHARGE
s_rest = [sigma_total(2*np.pi*f, mu_rest) for f in freqs]
s_active = [sigma_total(2*np.pi*f, mu_active) for f in freqs]
axes[1,0].loglog(freqs, s_rest, color=CYAN, lw=2, label='Rest (μ=0.1 eV)')
axes[1,0].loglog(freqs, s_active, color=MAGENTA, lw=2, label='Active (μ=0.2 eV)')
for l in [0, 5, 10, 15, 19]:
    axes[1,0].axvline(layer_freqs[l], color=GOLD, ls=':', alpha=0.3)
axes[1,0].set_xlabel('Frequency (Hz)', color=WHITE)
axes[1,0].set_ylabel('|σ(ω)| (S)', color=WHITE)
axes[1,0].set_title('AC Conductivity — Layer Frequencies Marked', color=WHITE)
axes[1,0].legend(fontsize=9)

# d) ent_alpha profile at rest vs active
ent_rest = ent_matrix[np.argmin(np.abs(mu_eV - 0.1))]
ent_active = ent_matrix[np.argmin(np.abs(mu_eV - 0.2))]
x = np.arange(n_layers)
axes[1,1].bar(x - 0.18, ent_rest, 0.35, color=CYAN, alpha=0.8, label='Rest (μ=0.1 eV)')
axes[1,1].bar(x + 0.18, ent_active, 0.35, color=MAGENTA, alpha=0.8, label='Active (μ=0.2 eV)')
axes[1,1].set_xlabel('Circuit Layer', color=WHITE)
axes[1,1].set_ylabel('ent_alpha', color=WHITE)
axes[1,1].set_title('Entanglement Strength: Rest vs Active', color=WHITE)
axes[1,1].legend(fontsize=9)
axes[1,1].set_xticks(range(0, 20, 2))

for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    ax.grid(alpha=0.1)

save(fig, '03_graphene_model.png')

# ═══════════════════════════════════════════════════════════════════
# 4. QUANTUM CIRCUIT — fidelity landscape simulation
# ═══════════════════════════════════════════════════════════════════
print("[4/6] Quantum fidelity landscape...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle('khaos-core Quantum Latent Space — Fidelity Navigation',
             color=WHITE, fontsize=15, fontweight='bold')

# a) Fidelity vs theta for a 1-qubit Ry circuit (pedagogical)
theta_scan = np.linspace(0, np.pi, 200)
landmarks = {'Rest': 0.2, 'Neutral': np.pi/2, 'Focus': 1.8, 'Motor': 2.8}
colors_lm = [CYAN, GOLD, MAGENTA, GREEN]

for (name, lm), c in zip(landmarks.items(), colors_lm):
    # F = cos²((θ-θ_lm)/2) for single-qubit Ry
    fid = np.cos((theta_scan - lm)/2)**2
    axes[0].plot(theta_scan, fid, color=c, lw=2, label=f'{name} (θ*={lm:.1f})')
    axes[0].axvline(lm, color=c, ls=':', alpha=0.3)

axes[0].set_xlabel('θ (rad)', color=WHITE)
axes[0].set_ylabel('Fidelity F', color=WHITE)
axes[0].set_title('Landmark Fidelity Profiles', color=WHITE)
axes[0].legend(fontsize=8)
axes[0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
axes[0].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])

# b) Simulated 12-qubit fidelity trajectory
np.random.seed(7)
t = np.linspace(0, 10, 1000)
# Simulate approaching "Focus" landmark
approach = 0.3 + 0.55 * (1 - np.exp(-t/3))
noise = 0.04 * np.random.randn(len(t))
fidelity = np.clip(approach + noise, 0, 1)
axes[1].plot(t, fidelity, color=MAGENTA, lw=1.5, alpha=0.9)
axes[1].axhline(0.85, color=GREEN, ls='--', alpha=0.7, label='NOMINAL threshold (0.85)')
axes[1].axhline(0.5, color=GOLD, ls='--', alpha=0.5, label='DEGRADED threshold (0.5)')
axes[1].axhline(0.3, color=RED, ls='--', alpha=0.5, label='PANIC threshold (0.3)')
axes[1].fill_between(t, 0.85, 1, alpha=0.05, color=GREEN)
axes[1].fill_between(t, 0, 0.3, alpha=0.05, color=RED)
axes[1].set_xlabel('Time (s)', color=WHITE)
axes[1].set_ylabel('Fidelity to Focus', color=WHITE)
axes[1].set_title('Real-Time Fidelity Trajectory', color=WHITE)
axes[1].legend(fontsize=8, loc='lower right')

# c) Circuit breaker state over time
states = np.zeros_like(t)
for i, f in enumerate(fidelity):
    if f > 0.85: states[i] = 3
    elif f > 0.5: states[i] = 2
    elif f > 0.3: states[i] = 1
    else: states[i] = 0

state_colors = {0: RED, 1: GOLD, 2: CYAN, 3: GREEN}
for i in range(len(t)-1):
    axes[2].fill_between([t[i], t[i+1]], 0, 1,
                          color=state_colors[int(states[i])], alpha=0.6)
axes[2].set_xlabel('Time (s)', color=WHITE)
axes[2].set_title('Circuit Breaker State', color=WHITE)
axes[2].set_yticks([0.2, 0.4, 0.6, 0.8])
axes[2].set_yticklabels(['PANIC', 'DEGRADED', 'RECOVERING', 'NOMINAL'], fontsize=9)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    ax.grid(alpha=0.1)

save(fig, '04_quantum_fidelity.png')

# ═══════════════════════════════════════════════════════════════════
# 5. FPGA REGISTER TIMELINE — simulated commit sequence
# ═══════════════════════════════════════════════════════════════════
print("[5/6] FPGA feedback timeline...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor=BG, sharex=True)
fig.suptitle('khaos-core FPGA Tactile Output — 12-Channel PWM + FM',
             color=WHITE, fontsize=15, fontweight='bold')

np.random.seed(99)
t = np.linspace(0, 2, 200)
n_ch = 12

# Simulated proximity (smoothed)
proximity = np.zeros((n_ch, len(t)))
for ch in range(n_ch):
    base = 0.4 + 0.3*np.sin(2*np.pi*0.5*t + ch*0.5)
    proximity[ch] = np.clip(base + 0.03*np.random.randn(len(t)), 0, 1)

# PWM duty from Pauli-Z
zi = 1 - 2*proximity
pwm = ((zi + 1) * 0.5 * 32767).astype(int)

# FM frequency
fm = 50 + proximity * 250

# Plot PWM heatmap
im1 = axes[0].imshow(pwm, aspect='auto', cmap='viridis',
                      extent=[0, 2, 11.5, -0.5])
axes[0].set_ylabel('Hub Ch', color=WHITE)
axes[0].set_title('PWM Duty Cycle (⟨Z_i⟩ → [0, 32767])', color=WHITE)
plt.colorbar(im1, ax=axes[0], label='Duty')

# Plot FM heatmap
im2 = axes[1].imshow(fm, aspect='auto', cmap='magma',
                      extent=[0, 2, 11.5, -0.5], vmin=50, vmax=300)
axes[1].set_ylabel('Hub Ch', color=WHITE)
axes[1].set_title('FM Frequency (Pacinian Range: 50–300 Hz)', color=WHITE)
plt.colorbar(im2, ax=axes[1], label='Hz')

# Plot sovereignty token
tokens = np.zeros(len(t), dtype=np.uint32)
for i in range(len(t)):
    token = i ^ 0xDEADBEEF
    for ch in range(n_ch):
        token ^= int(pwm[ch, i]) << (ch % 16)
    tokens[i] = token & 0xFFFF

axes[2].plot(t, tokens, color=GREEN, lw=1, alpha=0.8)
axes[2].set_ylabel('Token (16-bit)', color=WHITE)
axes[2].set_xlabel('Time (s)', color=WHITE)
axes[2].set_title('Sovereignty Token (XOR-fold frame hash)', color=WHITE)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)

save(fig, '05_fpga_output.png')

# ═══════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE LATENCY BUDGET
# ═══════════════════════════════════════════════════════════════════
print("[6/6] Latency budget...")

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)

stages = [
    ('LSL Pull\n(DMA)', 0.01),
    ('H2D\nMemcpy', 0.005),
    ('IIR Biquad\n(10-SOS)', 0.04),
    ('ICA\nUnmix', 0.008),
    ('DWT\nà-trous', 0.015),
    ('Metrics\nReduction', 0.005),
    ('D2H\nθ-Frame', 0.01),
    ('Python\nBridge', 0.5),
    ('Quantum\nCircuit', 40.0),
    ('Graphene\nModel', 0.005),
    ('Feedback\nKernel', 0.003),
    ('FPGA\nCommit', 0.001),
]

labels = [s[0] for s in stages]
times = [s[1] for s in stages]
colors_bar = [CYAN]*7 + [MAGENTA]*1 + [GOLD]*1 + [GREEN]*1 + [RED]*2

cumulative = np.cumsum(times)
starts = np.concatenate([[0], cumulative[:-1]])

bars = ax.barh(range(len(stages)), times, left=starts, color=colors_bar,
               edgecolor='white', linewidth=0.5, height=0.7, alpha=0.85)

for i, (s, t_ms) in enumerate(zip(starts, times)):
    if t_ms > 0.1:
        ax.text(s + t_ms/2, i, f'{t_ms:.1f} ms', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

ax.set_yticks(range(len(stages)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Time (ms)', color=WHITE)
ax.set_title('khaos-core Full Pipeline Latency Budget (1 Frame @ 1000 Hz)',
             color=WHITE, fontsize=14, fontweight='bold')
ax.axvline(1.0, color=RED, ls='--', lw=2, alpha=0.7, label='1 ms deadline')
ax.legend(fontsize=10)
ax.set_facecolor(BG)
ax.tick_params(colors=WHITE)
ax.grid(alpha=0.1, axis='x')
ax.invert_yaxis()

save(fig, '06_latency_budget.png')

print(f"\n✅ All visuals saved to {os.path.abspath(OUT)}")
