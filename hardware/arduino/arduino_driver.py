import struct
import time
import threading
import serial
import asyncio
import json
import websockets

STATE_MAP = {
    "NOMINAL": 0,
    "DEGRADED": 1,
    "PANIC": 2,
    "RECOVERING": 3,
}

MAX_PWM_RAW = 32767
MAX_FM_HZ = 300.0
MIN_FM_HZ = 50.0

class ArduinoFeedbackDriver:
    def __init__(self, port: str = "COM5", baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False
        self.frame_count = 0
        self.ack_count = 0
        self.last_noise = 0.0
        self._lock = threading.Lock()
        self._reader_thread = None
        self._running = False

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=0.05)
            time.sleep(2.0)
            self.connected = True
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_acks, daemon=True)
            self._reader_thread.start()
            print(f"[arduino-driver] Connected to {self.port}")
            return True
        except Exception as e:
            print(f"[arduino-driver] Connection failed: {e}")
            return False

    def disconnect(self):
        if self.connected:
            self._running = False
            time.sleep(0.1)
            self.ser.close()
            self.connected = False

    def send_frame(self, pwm_duty, fm_freq, sovereignty_token, circuit_state):
        if not self.connected: return False
        state_byte = STATE_MAP.get(circuit_state, 0)
        pwm_scaled = [max(0, min(255, int((d / MAX_PWM_RAW) * 255))) for d in pwm_duty[:4]]
        fm_scaled = [max(0, min(255, int(((f - MIN_FM_HZ) / (MAX_FM_HZ - MIN_FM_HZ)) * 255))) for f in fm_freq[:4]]
        payload = struct.pack("<HB4B4B", sovereignty_token & 0xFFFF, state_byte, *pwm_scaled, *fm_scaled)
        checksum = 0
        for b in payload: checksum ^= b
        frame = bytes([0xAA, 0x55]) + payload + bytes([checksum])
        with self._lock:
            try:
                self.ser.write(frame)
                self.frame_count += 1
                return True
            except: return False

    def _read_acks(self):
        while self._running:
            try:
                if self.ser and self.ser.in_waiting >= 4:
                    data = self.ser.read(4)
                    if data[0] == 0xAC:
                        self.ack_count += 1
                        self.last_noise = data[1] / 255.0
            except: break
            time.sleep(0.005)

async def bridge_telemetry_to_arduino(ws_url, serial_port):
    driver = ArduinoFeedbackDriver(port=serial_port)
    if not driver.connect(): return
    async with websockets.connect(ws_url) as ws:
        print(f"[bridge] Linked {ws_url} <-> {serial_port}")
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(msg)
                    driver.send_frame(data["pwm_duty"], data["fm_freq"], data["sov_token"], data["circuit_state"])
                except asyncio.TimeoutError: pass
                
                feedback = {"type": "hardware_feedback", "noise_level": driver.last_noise}
                await ws.send(json.dumps(feedback))
                await asyncio.sleep(1.0/60.0)
        except Exception as e:
            print(f"[bridge] Error: {e}")
        finally:
            driver.disconnect()

if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    asyncio.run(bridge_telemetry_to_arduino("ws://localhost:8765/ws", port))
