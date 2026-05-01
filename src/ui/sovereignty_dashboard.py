"""
sovereignty_dashboard.py — Real-Time Sovereignty Dashboard (Dual-Mode)
══════════════════════════════════════════════════════════════════════════════
Renders three panels updated via matplotlib FuncAnimation:

  Panel 1 — Bloch sphere (3-D)
    Maps q[0]→θ, q[1]→φ on the Bloch sphere.

  Panel 2 — 12-qubit bar chart
    Colour coding: green (0.33–0.67), amber (±edge), red (saturated).

  Panel 3 — Ethics gate status banner
    "ETHICS_GATE  PASS / BLOCK" + rolling event log.

Operating Modes
───────────────
  DEMO mode (default)
    Uses SyntheticMuse2Adapter for hardware-free development.
    Animation at ~10 Hz.
    Activated by: mode="demo" or when no shared memory is available.

  PROD mode
    Reads NeuralPhaseVector from the C++ kernel via POSIX shared memory
    (multiprocessing.shared_memory) or a UNIX domain socket fallback.
    Bloch sphere renders at 60 FPS independently of the 1000 Hz EEG rate
    (visual decimation: display every 17th kernel frame).
    Activated by: mode="prod", shm_name=<shared memory block name>

    NeuralPhaseVector layout (from include/khaos_bridge.h):
      float  theta[MAX_QUBITS=16]   — rotation angles [0, 2π]
      float  confidence             — [0, 1]
      float  entropy_estimate       — [0, 1]
      float  bp_index               — band-power index
      float  alpha_blend            — α-blending factor
      uint64 timestamp_ns           — UTC nanoseconds

    Total struct size: 16*4 + 4*4 + 8 = 88 bytes (padded to 96 for alignment)

Usage
─────
  # DEMO mode
  python sovereignty_dashboard.py

  # PROD mode
  python sovereignty_dashboard.py --mode prod --shm khaos_npv

  # Embedded:
  from src.ui.sovereignty_dashboard import SovereigntyDashboard
  dash = SovereigntyDashboard(mode="prod", shm_name="khaos_npv")
  dash.run()
"""

from __future__ import annotations

import argparse
import math
import queue
import struct
import threading
import time
from enum import Enum
from typing import Optional

import matplotlib
matplotlib.use("TkAgg" if __import__("sys").platform != "linux" else "Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers projection)

# ── Palette (matches NIGHTWATCH / KHAOS dark theme) ─────────────────────
BG       = "#060a0e"
GREEN    = "#00ff88"
AMBER    = "#ffaa00"
RED      = "#ff2222"
BLUE     = "#00ddff"
DIM      = "#1a3a2a"
TEXT_DIM = "#3a5a4a"
GRID_COL = "#0a2a1a"

QUBIT_LABELS = [
    "α-TP9", "α-AF7", "α-AF8", "α-TP10",
    "θ-TP9", "θ-AF7", "θ-AF8", "θ-TP10",
    "Coh-L", "Coh-R",
    "FAA",   "Engage",
]

REFRESH_HZ  = 10   # target update rate (DEMO mode)
PROD_FPS    = 60   # Bloch sphere FPS in PROD mode
GATE_LOG_N  = 60   # rolling gate event log length

# NeuralPhaseVector struct format (from khaos_bridge.h)
# float theta[16], float confidence, float entropy_estimate,
# float bp_index, float alpha_blend, uint64_t timestamp_ns
# + 4 bytes padding to reach 96-byte alignment
_NPV_FORMAT = "=16f4fQ4x"   # little-endian, 16 floats + 4 floats + uint64 + 4 pad
_NPV_SIZE   = struct.calcsize(_NPV_FORMAT)  # should be 96 bytes
_NPV_N_THETA = 16


class DashboardMode(str, Enum):
    DEMO = "demo"
    PROD = "prod"


# ── PROD mode: NeuralPhaseVector reader ───────────────────────────────────────

class NeuralPhaseVectorReader:
    """Reads NeuralPhaseVector structs from POSIX shared memory.

    The C++ kernel writes NeuralPhaseVector to a shared memory block at 1000 Hz.
    This reader polls at PROD_FPS (60 Hz) and decimates — displaying the most
    recently written frame rather than every frame.

    Falls back to a UNIX domain socket (path=/tmp/khaos_npv.sock) if shared
    memory is unavailable.

    Parameters
    ----------
    shm_name : str  — POSIX shared memory block name (e.g. 'khaos_npv')
    timeout  : float — seconds to wait for the block to appear
    """

    def __init__(self, shm_name: str = "khaos_npv", timeout: float = 5.0):
        self._shm_name = shm_name
        self._timeout  = timeout
        self._shm      = None
        self._sock     = None
        self._mode     = None
        self._lock     = threading.Lock()
        self._last_qubits = np.full(12, 0.5)

    def connect(self) -> bool:
        """Try shared memory first, then UNIX socket fallback.

        Returns True if connected.
        """
        # Try POSIX shared memory
        try:
            from multiprocessing.shared_memory import SharedMemory
            self._shm  = SharedMemory(name=self._shm_name, create=False)
            self._mode = "shm"
            print(f"[NPVReader] Connected via shared memory '{self._shm_name}' "
                  f"({self._shm.size} bytes)")
            return True
        except Exception as e:
            print(f"[NPVReader] Shared memory unavailable: {e}")

        # Try UNIX socket
        import socket, os
        sock_path = f"/tmp/{self._shm_name}.sock"
        if os.path.exists(sock_path):
            try:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.connect(sock_path)
                self._sock.settimeout(0.05)
                self._mode = "socket"
                print(f"[NPVReader] Connected via UNIX socket {sock_path}")
                return True
            except Exception as e:
                print(f"[NPVReader] UNIX socket unavailable: {e}")

        print("[NPVReader] No C++ kernel connection available — returning zeros.")
        return False

    def read_qubits(self) -> np.ndarray:
        """Read the latest NeuralPhaseVector and return 12 qubit values ∈ [0,1].

        Performs visual decimation: always returns the most recent frame,
        regardless of how many frames the C++ kernel has written since last call.
        """
        raw = self._read_raw()
        if raw is None:
            return self._last_qubits.copy()

        try:
            fields = struct.unpack(_NPV_FORMAT, raw[:_NPV_SIZE])
            theta_vec = np.array(fields[:_NPV_N_THETA], dtype=np.float64)
            # First 12 angles are the main register; scale [0,2π] → [0,1]
            qubits = theta_vec[:12] / (2 * math.pi)
            qubits = np.clip(qubits, 0.0, 1.0)
            with self._lock:
                self._last_qubits = qubits
            return qubits
        except Exception as e:
            print(f"[NPVReader] Unpack error: {e}")
            return self._last_qubits.copy()

    def _read_raw(self) -> Optional[bytes]:
        """Read _NPV_SIZE bytes from the active transport."""
        if self._mode == "shm" and self._shm is not None:
            try:
                return bytes(self._shm.buf[:_NPV_SIZE])
            except Exception:
                return None
        elif self._mode == "socket" and self._sock is not None:
            try:
                return self._sock.recv(_NPV_SIZE)
            except Exception:
                return None
        return None

    def close(self) -> None:
        if self._shm is not None:
            try: self._shm.close()
            except Exception: pass
        if self._sock is not None:
            try: self._sock.close()
            except Exception: pass


# ── Bloch sphere helper ────────────────────────────────────────────────────────

def _bloch_coords(theta_rad: float, phi_rad: float):
    """Convert (θ, φ) polar angles to Cartesian (x, y, z) on unit sphere."""
    x = math.sin(theta_rad) * math.cos(phi_rad)
    y = math.sin(theta_rad) * math.sin(phi_rad)
    z = math.cos(theta_rad)
    return x, y, z


def _draw_bloch_sphere(ax: "Axes3D"):
    """Render the static wireframe sphere once."""
    ax.set_facecolor(BG)
    ax.set_box_aspect([1, 1, 1])

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 13)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(u.size), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=DIM, linewidth=0.3, alpha=0.5)

    # Axes arrows
    for dx, dy, dz, lbl in [
        (1.3, 0, 0, "X"), (0, 1.3, 0, "Y"), (0, 0, 1.3, "|0⟩"),
        (0, 0, -1.3, "|1⟩"),
    ]:
        ax.quiver(0, 0, 0, dx, dy, dz, color=TEXT_DIM, linewidth=0.6,
                  arrow_length_ratio=0.08)
        ax.text(dx * 1.05, dy * 1.05, dz * 1.05, lbl,
                color=TEXT_DIM, fontsize=7, fontfamily="monospace")

    # Equatorial circle
    phi = np.linspace(0, 2 * np.pi, 60)
    ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi),
            color=DIM, linewidth=0.5, linestyle=":")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COL)
    ax.set_title("Bloch Sphere  q[0]×q[1]", color=GREEN,
                 fontsize=8, fontfamily="monospace", pad=6)


# ── Main dashboard ─────────────────────────────────────────────────────────────

class SovereigntyDashboard:
    """Real-time KHAOS sovereignty dashboard.

    Parameters
    ----------
    adapter     : Muse2Adapter or SyntheticMuse2Adapter (optional)
    extractor   : Muse2FeatureExtractor (optional)
    gate        : EthicsGate (optional)
    refresh_hz  : animation frame rate (default 10)
    """

    def __init__(self, adapter=None, extractor=None, gate=None,
                 refresh_hz: int = REFRESH_HZ,
                 mode: str = "demo",
                 shm_name: str = "khaos_npv"):
        """
        Parameters
        ----------
        adapter, extractor, gate : used in DEMO mode (all optional)
        refresh_hz : animation interval in Hz (DEMO mode default: 10)
        mode       : 'demo' or 'prod'
        shm_name   : shared memory block name for PROD mode
        """
        self._adapter   = adapter
        self._extractor = extractor
        self._gate      = gate
        self._mode      = DashboardMode(mode)

        fps = PROD_FPS if self._mode == DashboardMode.PROD else refresh_hz
        self._interval  = int(1000 / fps)

        # PROD: NeuralPhaseVector reader
        self._npv_reader: Optional[NeuralPhaseVectorReader] = None
        if self._mode == DashboardMode.PROD:
            self._npv_reader = NeuralPhaseVectorReader(shm_name=shm_name)
            connected = self._npv_reader.connect()
            if not connected:
                print("[SovereigntyDashboard] PROD mode: kernel not found, "
                      "falling back to DEMO.")
                self._mode = DashboardMode.DEMO

        # Shared state (updated by data thread, read by animation)
        self._qubits_01  = np.full(12, 0.5)
        self._gate_pass  = True
        self._gate_log: queue.Queue = queue.Queue(maxsize=GATE_LOG_N)
        self._data_lock  = threading.Lock()

        # Matplotlib handles
        self._fig        = None
        self._anim       = None
        self._bloch_arrow = None

    # ── Data acquisition thread ────────────────────────────────────────────

    def _data_loop(self) -> None:
        """Background thread: pull qubit values and gate status."""
        while self._running:
            try:
                qubits = self._fetch_qubits()
                gate_ok = self._check_gate(qubits)
                with self._data_lock:
                    self._qubits_01 = qubits
                    self._gate_pass  = gate_ok
            except Exception as exc:  # noqa: BLE001
                with self._data_lock:
                    self._gate_pass = False
                try:
                    self._gate_log.put_nowait(f"ERROR: {exc}")
                except queue.Full:
                    pass
            time.sleep(1.0 / REFRESH_HZ)

    def _fetch_qubits(self) -> np.ndarray:
        """Return current 12-qubit values ∈ [0, 1].

        PROD mode: reads from NeuralPhaseVector shared memory (C++ kernel).
        DEMO mode: uses SyntheticMuse2Adapter or oscillating fallback.
        """
        # ── PROD mode ──────────────────────────────────────────────────────
        if self._mode == DashboardMode.PROD and self._npv_reader is not None:
            return self._npv_reader.read_qubits()

        # ── DEMO mode ──────────────────────────────────────────────────────
        if self._adapter is None or self._extractor is None:
            t = time.time()
            qubits = np.array([
                0.5 + 0.4 * math.sin(2 * math.pi * 0.2 * t + i * 0.5)
                for i in range(12)
            ])
            return np.clip(qubits, 0, 1)

        if not self._adapter.ready:
            return self._qubits_01

        alpha     = self._adapter.get_filtered_window("alpha")
        theta     = self._adapter.get_filtered_window("theta")
        theta_vec = self._extractor.extract(alpha, theta)
        return theta_vec[:12] / (2 * math.pi)

    def _check_gate(self, qubits: np.ndarray) -> bool:
        """Run gate_pass() and log the result."""
        if self._gate is None:
            return True
        try:
            theta_vec = qubits * 2 * math.pi
            theta_tiled = np.tile(theta_vec, 20)   # (240,)
            self._gate.gate_pass(theta_tiled, label="dashboard")
            msg = f"{time.strftime('%H:%M:%S')} PASS  norm={np.linalg.norm(theta_tiled):.2f}"
            try:
                self._gate_log.put_nowait(msg)
            except queue.Full:
                try: self._gate_log.get_nowait()
                except queue.Empty: pass
                self._gate_log.put_nowait(msg)
            return True
        except Exception as exc:  # noqa: BLE001
            msg = f"{time.strftime('%H:%M:%S')} BLOCK {exc}"
            try: self._gate_log.put_nowait(msg)
            except queue.Full: pass
            return False

    # ── Figure construction ────────────────────────────────────────────────

    def _build_figure(self):
        mode_tag = "PROD  ·  C++ kernel" if self._mode == DashboardMode.PROD \
            else "DEMO  ·  synthetic"
        self._fig = plt.figure(figsize=(16, 8), facecolor=BG)
        self._fig.suptitle(
            f"KHAOS  ·  Sovereignty Dashboard  ·  [{mode_tag}]  ·  KGL v6.0",
            color=GREEN, fontsize=11, fontweight="bold",
            fontfamily="monospace", y=0.98)

        gs = gridspec.GridSpec(
            2, 3, figure=self._fig,
            hspace=0.45, wspace=0.35,
            left=0.05, right=0.97,
            top=0.93,  bottom=0.05)

        # Panel 1 — Bloch sphere (spans 2 rows, col 0)
        self._ax_bloch = self._fig.add_subplot(gs[:, 0], projection="3d")
        _draw_bloch_sphere(self._ax_bloch)

        # Panel 2 — Qubit bars (row 0, col 1-2)
        self._ax_bars = self._fig.add_subplot(gs[0, 1:])
        self._ax_bars.set_facecolor(BG)
        self._ax_bars.set_title("12-Qubit Neural State", color=GREEN,
                                fontsize=9, fontfamily="monospace")
        self._ax_bars.set_xlim(-0.5, 11.5)
        self._ax_bars.set_ylim(0, 1)
        self._ax_bars.set_xticks(range(12))
        self._ax_bars.set_xticklabels(QUBIT_LABELS, fontsize=6,
                                       color=TEXT_DIM, rotation=30, ha="right")
        self._ax_bars.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        self._ax_bars.tick_params(colors=TEXT_DIM, labelsize=6)
        for spine in self._ax_bars.spines.values():
            spine.set_edgecolor(DIM)
        self._ax_bars.axhline(y=0.33, color=DIM, linewidth=0.5, linestyle="--")
        self._ax_bars.axhline(y=0.67, color=DIM, linewidth=0.5, linestyle="--")
        self._ax_bars.set_ylabel("Qubit value [0,1]", color=TEXT_DIM, fontsize=7)

        # Panel 3 — Gate status + log (row 1, col 1-2)
        self._ax_gate = self._fig.add_subplot(gs[1, 1:])
        self._ax_gate.set_facecolor(BG)
        self._ax_gate.axis("off")

        # Pre-create bar patches
        self._bars = self._ax_bars.bar(
            range(12), [0.5] * 12,
            color=GREEN, width=0.7, alpha=0.85)

        # Bloch state vector (will be replaced each frame)
        self._bloch_arrow = None

        # Gate status text
        self._gate_text = self._ax_gate.text(
            0.5, 0.85, "ETHICS_GATE  INITIALISING",
            ha="center", va="top", color=AMBER,
            fontsize=18, fontweight="bold", fontfamily="monospace",
            transform=self._ax_gate.transAxes)

        # Gate log text
        self._log_text = self._ax_gate.text(
            0.02, 0.70, "",
            ha="left", va="top", color=TEXT_DIM,
            fontsize=6.5, fontfamily="monospace",
            transform=self._ax_gate.transAxes)

        # Footnote
        self._fig.text(
            0.5, 0.01,
            "Raw EEG destroyed at DSP boundary  ·  "
            "Only 12-qubit feature vector crosses sovereignty gate  ·  "
            "Audit log: SHA-256 chained",
            ha="center", color=TEXT_DIM, fontsize=6.5, fontfamily="monospace")

    # ── Animation frame ────────────────────────────────────────────────────

    def _animate(self, _frame: int):
        with self._data_lock:
            qubits   = self._qubits_01.copy()
            gate_ok  = self._gate_pass

        # ── Panel 1: update Bloch arrow ────────────────────────────────────
        if self._bloch_arrow is not None:
            self._bloch_arrow.remove()
            self._bloch_arrow = None

        theta_bloch = qubits[0] * math.pi        # q[0] → θ ∈ [0, π]
        phi_bloch   = qubits[1] * 2 * math.pi    # q[1] → φ ∈ [0, 2π]
        bx, by, bz  = _bloch_coords(theta_bloch, phi_bloch)
        self._bloch_arrow = self._ax_bloch.quiver(
            0, 0, 0, bx, by, bz,
            color=RED, linewidth=2.0, arrow_length_ratio=0.12, zorder=10)

        # State vector label
        self._ax_bloch.set_xlabel(
            f"θ={math.degrees(theta_bloch):.1f}°  φ={math.degrees(phi_bloch):.1f}°",
            color=TEXT_DIM, fontsize=6.5, labelpad=2)

        # ── Panel 2: update bars ───────────────────────────────────────────
        for i, (bar, val) in enumerate(zip(self._bars, qubits)):
            bar.set_height(float(val))
            if 0.33 <= val <= 0.67:
                col = GREEN
            elif 0.10 <= val < 0.33 or 0.67 < val <= 0.90:
                col = AMBER
            else:
                col = RED
            bar.set_color(col)

        # ── Panel 3: gate status ───────────────────────────────────────────
        if gate_ok:
            self._gate_text.set_text("ETHICS_GATE  ✓  PASS")
            self._gate_text.set_color(GREEN)
        else:
            self._gate_text.set_text("ETHICS_GATE  ✗  BLOCK")
            self._gate_text.set_color(RED)

        # Drain gate log queue
        log_lines = []
        while True:
            try:
                log_lines.append(self._gate_log.get_nowait())
            except queue.Empty:
                break

        # Rebuild rolling log display (keep last 8 lines)
        if not hasattr(self, "_log_buffer"):
            self._log_buffer = []
        self._log_buffer.extend(log_lines)
        self._log_buffer = self._log_buffer[-8:]
        self._log_text.set_text("\n".join(self._log_buffer))

        return list(self._bars) + [self._gate_text, self._log_text]

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self, block: bool = True) -> None:
        """Start the dashboard.  Blocks by default until window is closed."""
        self._build_figure()

        # Start data thread
        self._running = True
        self._data_thread = threading.Thread(
            target=self._data_loop, name="SovDash-Data", daemon=True)
        self._data_thread.start()

        self._anim = FuncAnimation(
            self._fig,
            self._animate,
            interval=self._interval,
            blit=False,    # 3-D axes don't support blit
            cache_frame_data=False,
        )

        if block:
            try:
                plt.show()
            finally:
                self._running = False
        else:
            plt.ion()
            plt.show()

    def stop(self) -> None:
        """Signal the data thread to stop and release resources."""
        self._running = False
        if self._npv_reader is not None:
            self._npv_reader.close()

    def save_snapshot(self, path: str = "sovereignty_snapshot.png") -> None:
        """Save the current figure to disk (headless-safe)."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=150, bbox_inches="tight",
                              facecolor=BG)
            print(f"[SovereigntyDashboard] Snapshot saved: {path}")


# ── Self-test (headless snapshot) ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description="KHAOS Sovereignty Dashboard")
    parser.add_argument("--mode",    default="demo", choices=["demo", "prod"],
                        help="'demo' (synthetic) or 'prod' (C++ kernel via shm)")
    parser.add_argument("--shm",     default="khaos_npv",
                        help="Shared memory block name for PROD mode")
    parser.add_argument("--test",    action="store_true",
                        help="Headless snapshot test (no window)")
    args = parser.parse_args()

    if args.test or args.mode == "demo":
        print("=== sovereignty_dashboard self-test (headless snapshot) ===")
        matplotlib.use("Agg")   # force headless

    if args.mode == "prod" and not args.test:
        dash = SovereigntyDashboard(mode="prod", shm_name=args.shm)
        dash.run()
        sys.exit(0)

    dash = SovereigntyDashboard(mode="demo")
    dash._build_figure()

    # Inject synthetic qubit values
    with dash._data_lock:
        dash._qubits_01 = np.array([
            0.60, 0.45, 0.55, 0.50,   # alpha TP9/AF7/AF8/TP10
            0.35, 0.40, 0.38, 0.42,   # theta
            0.72, 0.68,               # coherence L/R
            0.55,                     # FAA
            0.48,                     # engagement
        ])
        dash._gate_pass = True

    # Run one animation frame
    dash._animate(0)

    out = Path(tempfile.mkdtemp()) / "sovereignty_snapshot.png"
    dash.save_snapshot(str(out))

    import os
    assert os.path.getsize(str(out)) > 1000, "Snapshot too small — render failed"
    print(f"Snapshot written to: {out}")
    print("Self-test passed. ✓")
