"""
arduino_driver.py — khaos-core Arduino Serial Feedback Driver
=============================================================
Drop-in replacement for the FPGA PCIe driver during prototyping.
Speaks the same binary protocol as khaos_tactile.ino over serial.

Usage:
    from arduino_driver import ArduinoFeedbackDriver

    driver = ArduinoFeedbackDriver(port="COM5")  # or /dev/ttyUSB0
    driver.connect()
    driver.send_frame(
        pwm_duty=[16000, 8000, 24000, 12000],
        fm_freq=[120.0, 80.0, 200.0, 150.0],
        sovereignty_token=0xA1B2,
        circuit_state="NOMINAL"
    )
    driver.disconnect()
"""

import struct
import time
import threading
import serial


# Maps circuit_state string to protocol byte
STATE_MAP = {
    "NOMINAL": 0,
    "DEGRADED": 1,
    "PANIC": 2,
    "RECOVERING": 3,
}

# Safety constants (mirrored from safety_constants.h)
MAX_PWM_RAW = 32767
MAX_FM_HZ = 300.0
MIN_FM_HZ = 50.0
WATCHDOG_TIMEOUT_S = 0.1


class ArduinoFeedbackDriver:
    """
    Serial driver that sends tactile feedback frames to the Arduino
    at up to 60 Hz. Implements the same logical interface as the
    FPGA BAR0 driver but over UART.

    Protocol: 14 bytes per frame (see khaos_tactile.ino header).
    """

    def __init__(self, port: str = "COM5", baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False
        self.frame_count = 0
        self.ack_count = 0
        self._lock = threading.Lock()
        self._reader_thread = None
        self._running = False

    def connect(self) -> bool:
        """Open serial connection and wait for Arduino ready signal."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.05,
                write_timeout=0.05,
            )
            time.sleep(2.0)  # Arduino resets on serial connect

            # Read ready signal
            ready = self.ser.readline().decode(errors="ignore").strip()
            if "KHAOS-TACTILE" in ready:
                print(f"[arduino-driver] Connected: {ready}")
                self.connected = True

                # Start ACK reader thread
                self._running = True
                self._reader_thread = threading.Thread(
                    target=self._read_acks, daemon=True
                )
                self._reader_thread.start()
                return True
            else:
                print(f"[arduino-driver] Unexpected response: {ready}")
                self.ser.close()
                return False

        except serial.SerialException as e:
            print(f"[arduino-driver] Connection failed: {e}")
            return False

    def disconnect(self):
        """Zero all outputs and close connection."""
        if self.connected:
            self.send_frame(
                pwm_duty=[0, 0, 0, 0],
                fm_freq=[0, 0, 0, 0],
                sovereignty_token=0,
                circuit_state="PANIC",
            )
            self._running = False
            time.sleep(0.1)
            self.ser.close()
            self.connected = False
            print(f"[arduino-driver] Disconnected. {self.frame_count} frames sent, {self.ack_count} acked.")

    def send_frame(
        self,
        pwm_duty: list[int],
        fm_freq: list[float],
        sovereignty_token: int,
        circuit_state: str,
    ) -> bool:
        """
        Send one feedback frame to the Arduino.

        Args:
            pwm_duty: list of 4 ints [0, 32767] — per-channel duty
            fm_freq: list of 4 floats [50, 300] Hz — per-channel frequency
            sovereignty_token: uint16 — XOR-fold frame hash
            circuit_state: str — NOMINAL|DEGRADED|PANIC|RECOVERING
        """
        if not self.connected:
            return False

        state_byte = STATE_MAP.get(circuit_state, 0)

        # Scale pwm_duty from [0, 32767] → [0, 255]
        pwm_scaled = []
        for d in pwm_duty[:4]:
            scaled = int((d / MAX_PWM_RAW) * 255)
            scaled = max(0, min(255, scaled))
            pwm_scaled.append(scaled)

        # Scale fm_freq from [50, 300] → [0, 255]
        fm_scaled = []
        for f in fm_freq[:4]:
            scaled = int(((f - MIN_FM_HZ) / (MAX_FM_HZ - MIN_FM_HZ)) * 255)
            scaled = max(0, min(255, scaled))
            fm_scaled.append(scaled)

        # Build frame
        # Bytes 0-1: sync
        # Bytes 2-3: token (uint16 LE)
        # Byte 4: state
        # Bytes 5-8: pwm[0..3]
        # Bytes 9-12: fm[0..3]
        # Byte 13: checksum (XOR of bytes 2-12)
        payload = struct.pack(
            "<HB4B4B",
            sovereignty_token & 0xFFFF,
            state_byte,
            *pwm_scaled,
            *fm_scaled,
        )

        checksum = 0
        for b in payload:
            checksum ^= b

        frame = bytes([0xAA, 0x55]) + payload + bytes([checksum])

        with self._lock:
            try:
                self.ser.write(frame)
                self.frame_count += 1
                return True
            except serial.SerialException:
                return False

    def _read_acks(self):
        """Background thread that reads ACK bytes from Arduino."""
        while self._running:
            try:
                if self.ser and self.ser.in_waiting >= 3:
                    data = self.ser.read(3)
                    if data[0] == 0xAC:
                        self.ack_count += 1
            except (serial.SerialException, OSError):
                break
            time.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════
# Integration helper: bridge from khaos-core telemetry to Arduino
# ═══════════════════════════════════════════════════════════════════

def bridge_telemetry_to_arduino(ws_url: str = "ws://localhost:8765/ws",
                                 serial_port: str = "COM5"):
    """
    Connects the khaos-core telemetry WebSocket to an Arduino.
    Reads frames from the dashboard server and forwards to hardware.
    """
    import asyncio
    import json
    import websockets

    driver = ArduinoFeedbackDriver(port=serial_port)
    if not driver.connect():
        print("[bridge] Failed to connect to Arduino")
        return

    async def run():
        async with websockets.connect(ws_url) as ws:
            print(f"[bridge] Connected to {ws_url}")
            print(f"[bridge] Forwarding to Arduino on {serial_port}")
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    driver.send_frame(
                        pwm_duty=data["pwm_duty"][:4],
                        fm_freq=data["fm_freq"][:4],
                        sovereignty_token=data["sov_token"],
                        circuit_state=data["circuit_state"],
                    )
            except KeyboardInterrupt:
                pass
            finally:
                driver.disconnect()

    asyncio.run(run())


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
    bridge_telemetry_to_arduino(serial_port=port)
