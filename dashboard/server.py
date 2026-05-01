"""
server.py — KHAOS Telemetry WebSocket Server
===================================================
FastAPI + WebSockets. Simulates the kernel pipeline at 60 Hz.

Data emitted per frame:
  - eeg[64]           : 64-channel EEG amplitudes (µV, bandpass-filtered)
  - theta[12]         : hub channel θ angles (rad, 0→π)
  - fidelity          : quantum fidelity to current landmark (0→1)
  - confidence        : signal confidence from DSP metrics (0→1)
  - entropy           : entropy estimate (0→1)
  - bp_index          : Bereitschaftspotential index (0→1)
  - pwm_duty[12]      : FPGA PWM duty per hub (0→32767)
  - fm_freq[12]       : FPGA FM frequency per hub (50→300 Hz)
  - sov_token         : sovereignty token (16-bit)
  - circuit_state     : NOMINAL|DEGRADED|PANIC|RECOVERING
  - mu_eV             : graphene chemical potential (eV)
  - ent_alpha[20]     : entanglement vector per circuit layer
  - frame_id          : monotonic frame counter
  - timestamp_ms      : server timestamp

Run:  uvicorn server:app --host 0.0.0.0 --port 8765 --log-level warning
"""

import asyncio
import json
import math
import time
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

app = FastAPI(title="KHAOS telemetry")

# ── Serve index.html ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html", media_type="text/html")


# ── Simulated kernel state ────────────────────────────────────────────────────

class KernelSimulator:
    """
    Simulates the KHAOS pipeline output at 60 Hz.
    Models realistic EEG dynamics, quantum fidelity transitions,
    and circuit breaker state machine.
    """

    def __init__(self):
        self.frame_id = 0
        self.t = 0.0
        self.dt = 1.0 / 60.0  # 60 Hz

        # Circuit breaker state machine
        self.state = "NOMINAL"
        self.state_timer = 0.0
        self.panic_cooldown = 0.0

        # Fidelity dynamics
        self.fidelity_target = 0.82
        self.fidelity = 0.5
        self.fidelity_vel = 0.0

        # EEG phase offsets (persistent per channel)
        self.eeg_phases = np.random.uniform(0, 2 * np.pi, 64)
        self.eeg_freqs = np.random.uniform(8, 30, 64)  # within passband

        # Theta smoothing
        self.theta_smooth = np.full(12, np.pi / 2)

        # Noise injection for demo
        self.noise_level = 0.0
        self.noise_target = 0.0
        self.spike_timer = 0.0

        # Graphene mu
        self.mu_eV = 0.1

    def step(self) -> dict:
        self.t += self.dt
        self.frame_id += 1

        # ── Noise dynamics (random spikes for demo) ───────────────────────
        self.spike_timer -= self.dt
        if self.spike_timer <= 0:
            # Random noise events
            if random.random() < 0.03:  # 3% chance per frame of spike
                self.noise_target = random.uniform(0.7, 0.95)
                self.spike_timer = random.uniform(2.0, 5.0)
            else:
                self.noise_target = random.uniform(0.05, 0.3)
                self.spike_timer = random.uniform(1.0, 3.0)

        self.noise_level += (self.noise_target - self.noise_level) * 0.5

        # ── EEG simulation (64 channels) ──────────────────────────────────
        eeg = np.zeros(64)
        for ch in range(64):
            # Clean signal: sum of alpha + beta components
            alpha = 15.0 * np.sin(2 * np.pi * self.eeg_freqs[ch] * self.t
                                   + self.eeg_phases[ch])
            beta = 8.0 * np.sin(2 * np.pi * (self.eeg_freqs[ch] * 1.7) * self.t
                                 + self.eeg_phases[ch] * 0.7)
            noise = self.noise_level * 40.0 * np.random.randn()
            eeg[ch] = alpha + beta + noise

        # ── Theta angles (12 hubs) ────────────────────────────────────────
        theta_raw = np.zeros(12)
        for hub in range(12):
            # Simulate beta/mu power ratio varying over time
            beta_power = 5.0 + 3.0 * np.sin(0.3 * self.t + hub * 0.5)
            mu_power = 5.0 + 3.0 * np.cos(0.2 * self.t + hub * 0.3)
            beta_power *= (1 + self.noise_level * np.random.randn() * 0.5)
            mu_power *= (1 + self.noise_level * np.random.randn() * 0.5)
            beta_power = max(beta_power, 0.01)
            mu_power = max(mu_power, 0.01)
            ratio = beta_power / (mu_power + beta_power + 1e-8)
            theta_raw[hub] = 2.0 * np.arcsin(np.sqrt(np.clip(ratio, 0, 1)))

        # Smooth theta
        self.theta_smooth += (theta_raw - self.theta_smooth) * 0.15
        theta = self.theta_smooth.copy()

        # ── Confidence & entropy ──────────────────────────────────────────
        confidence = np.clip(1.0 - self.noise_level * 1.2, 0, 1)
        entropy = np.clip(0.3 + self.noise_level * 0.6 + 0.1 * np.sin(0.5 * self.t), 0, 1)
        bp_index = np.clip(0.5 + 0.3 * np.sin(0.15 * self.t) - self.noise_level * 0.3, 0, 1)

        # ── Quantum fidelity ──────────────────────────────────────────────
        # Spring-damper model toward target, perturbed by noise
        if self.state == "PANIC":
            self.fidelity_target = 0.15
        elif self.state == "RECOVERING":
            self.fidelity_target = 0.7 + 0.2 * (1 - np.exp(-self.state_timer / 3))
        else:
            self.fidelity_target = 0.75 + 0.15 * np.sin(0.1 * self.t)
            self.fidelity_target -= self.noise_level * 0.4

        spring = 2.0 * (self.fidelity_target - self.fidelity) - 0.8 * self.fidelity_vel
        self.fidelity_vel += spring * self.dt
        self.fidelity += self.fidelity_vel * self.dt
        self.fidelity += np.random.randn() * 0.01
        self.fidelity = np.clip(self.fidelity, 0, 1)

        # ── Circuit breaker state machine ─────────────────────────────────
        self.state_timer += self.dt

        if self.noise_level > 0.6:
            self.state = "PANIC"
            self.fidelity_target = 0.1
            confidence = 0.1
        elif self.state == "PANIC" and self.noise_level < 0.4:
            self.state = "RECOVERING"
            self.state_timer = 0
        elif self.state == "RECOVERING" and self.fidelity > 0.8:
            self.state = "NOMINAL"
            self.state_timer = 0
        elif self.state == "NOMINAL" and self.noise_level > 0.4:
            self.state = "DEGRADED"
        elif self.state == "DEGRADED" and self.noise_level < 0.2:
            self.state = "NOMINAL"

        self.panic_cooldown = max(0, self.panic_cooldown - self.dt)

        # ── FPGA output ───────────────────────────────────────────────────
        global_scale = 0.0 if self.state == "PANIC" else 1.0
        pwm_duty = np.zeros(12, dtype=int)
        fm_freq = np.zeros(12)

        for ch in range(12):
            proximity = np.clip(self.fidelity * (0.8 + 0.2 * np.sin(self.t + ch)), 0, 1)
            if self.state != "PANIC":
                zi = 1.0 - 2.0 * proximity
                pwm_duty[ch] = int(np.clip((zi + 1.0) * 0.5 * 32767, 0, 32767))
                fm_freq[ch] = 200.0 + proximity * 800.0 # Subimos a 200Hz - 1000Hz
            # else: stays 0

        # Sovereignty token
        token = self.frame_id ^ 0xDEADBEEF
        for ch in range(12):
            token ^= int(pwm_duty[ch]) << (ch % 16)
        sov_token = token & 0xFFFF

        # ── Graphene model ────────────────────────────────────────────────
        self.mu_eV = 0.1 + 0.08 * np.sin(0.2 * self.t) * (1 - self.noise_level)
        ent_alpha = np.zeros(20)
        for layer in range(20):
            freq = 250.0 / (2 ** (layer + 1))
            sigma = abs(self.mu_eV) * 0.01 / max(freq * 0.001, 0.0001)
            ent_alpha[layer] = np.tanh(sigma * 500)

        return {
            "eeg": eeg.round(2).tolist(),
            "theta": theta.round(4).tolist(),
            "fidelity": round(float(self.fidelity), 4),
            "confidence": round(float(confidence), 4),
            "entropy": round(float(entropy), 4),
            "bp_index": round(float(bp_index), 4),
            "noise_level": round(float(self.noise_level), 4),
            "pwm_duty": pwm_duty.tolist(),
            "fm_freq": fm_freq.round(1).tolist(),
            "sov_token": sov_token,
            "circuit_state": self.state,
            "global_scale": global_scale,
            "mu_eV": round(self.mu_eV, 4),
            "ent_alpha": ent_alpha.round(4).tolist(),
            "frame_id": self.frame_id,
            "timestamp_ms": round(time.time() * 1000),
        }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    sim = KernelSimulator()
    print("[khaos-telemetry] Client connected")

    async def broadcast():
        try:
            while True:
                frame = sim.step()
                await websocket.send_text(json.dumps(frame))
                await asyncio.sleep(1.0 / 60.0)
        except Exception as e:
            print(f"[broadcast] Error: {e}")

    async def listen():
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("type") == "hardware_feedback":
                    # Potentiometer controls noise target directly
                    sim.noise_target = data["noise_level"]
        except Exception as e:
            print(f"[listen] Error: {e}")

    await asyncio.gather(broadcast(), listen())
