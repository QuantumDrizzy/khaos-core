import ctypes
import time
import serial
import sys

# --- CONFIGURACIÓN ---
PORT = "COM3"
BAUD = 115200
THRESHOLD_DIST = 70.0
GRACE_PERIOD = 3.0

def lock_workstation():
    print("\n\n[!!!] NEURAL LINK LOST. LOCKING SYSTEM...")
    ctypes.windll.user32.LockWorkStation()

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   KHAOS-CORE NEURAL SECURITY MONITOR     ║")
    print("║         [VERBOSE DEBUG MODE]             ║")
    print("╚══════════════════════════════════════════╝")
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0) # Aumentado timeout
        print(f"[*] Conectado a {PORT}. Esperando datos...")
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    presence_timer = 0
    
    while True:
        try:
            if ser.in_waiting > 0:
                # Leer línea y limpiar
                raw_line = ser.readline()
                line = raw_line.decode('utf-8', errors='ignore').strip()
                
                # DEBUG: Imprimir todo lo que llega
                if line:
                    print(f"DEBUG IN: '{line}'")
                
                if "Dist:" in line:
                    try:
                        d_str = line.split("Dist:")[1].split("cm")[0].strip()
                        dist = float(d_str)
                        
                        if dist > THRESHOLD_DIST:
                            presence_timer += 0.05
                        else:
                            presence_timer = 0
                            
                        if presence_timer >= GRACE_PERIOD:
                            lock_workstation()
                            presence_timer = 0
                    except:
                        pass
            
            time.sleep(0.01) # Loop más rápido
            
        except KeyboardInterrupt:
            print("\n[!] Saliendo...")
            break
        except Exception as e:
            print(f"\n[!] Error en loop: {e}")
            break

    ser.close()

if __name__ == "__main__":
    main()
