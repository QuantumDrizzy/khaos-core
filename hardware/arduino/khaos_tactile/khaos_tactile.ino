/*
 * khaos_tactile.ino — [ULTIMATE HYBRID] Khaos-Core BCI Bridge
 * =========================================================
 * 1. FIDELITY MODE (Default): Smooth arm & sound tracking.
 * 2. MEMORY MODE (Secret): Triggered by holding hand <5cm for 3s.
 */

#include <Servo.h>

const uint8_t BUZZER    = 6;
const uint8_t TRIG      = 7;
const uint8_t ECHO      = 8;
const uint8_t LED_GREEN = 9;
const uint8_t LED_RED   = 10;
const uint8_t SERVO_PIN = 12;

Servo neuralArm;
float smoothFid = 0.0;
bool modeMemory = false;
unsigned long modeSwitchStart = 0;

void setup() {
    Serial.begin(115200);
    pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
    pinMode(BUZZER, OUTPUT); pinMode(LED_GREEN, OUTPUT); pinMode(LED_RED, OUTPUT);
    neuralArm.attach(SERVO_PIN);
    randomSeed(analogRead(A5));
    
    // Cyberpunk Boot
    for(int i=0; i<3; i++) {
        digitalWrite(LED_GREEN, HIGH); tone(BUZZER, 800 + i*200, 50); delay(100);
        digitalWrite(LED_GREEN, LOW); delay(50);
    }
    Serial.println(">>> KHAOS HYBRID FIRMWARE V3.0 ONLINE <<<");
}

float getDist() {
    digitalWrite(TRIG, LOW); delayMicroseconds(2);
    digitalWrite(TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    long dur = pulseIn(ECHO, HIGH, 20000);
    return (dur == 0) ? 100.0 : (dur * 0.0343) / 2.0;
}

void loop() {
    if (!modeMemory) {
        runFidelityMode();
    } else {
        runMemoryMode();
    }
}

// ── MODE 1: FIDELITY TRACKING ─────────────────────────────────────
void runFidelityMode() {
    float d = getDist();
    
    // Mode Switch Detection: Hold hand < 5cm for 3 seconds
    if (d < 5.0) {
        if (modeSwitchStart == 0) modeSwitchStart = millis();
        if (millis() - modeSwitchStart > 3000) {
            // Switch to Memory Mode
            tone(BUZZER, 2000, 500);
            digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_RED, HIGH);
            delay(1000);
            modeMemory = true;
            modeSwitchStart = 0;
            return;
        }
        // Visual feedback for charging switch
        digitalWrite(LED_GREEN, (millis()/100)%2);
    } else {
        modeSwitchStart = 0;
    }

    float targetFid = constrain(1.0 - ((d - 5.0) / 30.0), 0.0, 1.0);
    smoothFid += (targetFid - smoothFid) * 0.25;

    // Haptics & Visuals
    if (smoothFid > 0.05) {
        tone(BUZZER, 200 + (uint16_t)(smoothFid * 1400.0));
    } else {
        noTone(BUZZER);
    }
    
    neuralArm.write((int)(smoothFid * 180.0));
    
    if (smoothFid > 0.7) { digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_RED, LOW); }
    else { digitalWrite(LED_GREEN, LOW); digitalWrite(LED_RED, HIGH); }
    
    delay(30);
}

// ── MODE 2: MEMORY CHALLENGE ──────────────────────────────────────
void runMemoryMode() {
    int pattern[3];
    Serial.println("\n[MEMORY MODE] Initializing...");
    
    // Flash LEDs to indicate mode change
    for(int i=0; i<3; i++) { digitalWrite(LED_RED, HIGH); delay(100); digitalWrite(LED_RED, LOW); delay(100); }
    
    // Play Pattern
    for (int i = 0; i < 3; i++) {
        pattern[i] = random(0, 3);
        if (pattern[i] == 0) { tone(BUZZER, 300, 400); neuralArm.write(10); }
        else if (pattern[i] == 1) { tone(BUZZER, 800, 400); neuralArm.write(90); }
        else { tone(BUZZER, 1600, 400); neuralArm.write(175); }
        delay(600); noTone(BUZZER); delay(200);
    }

    // Wait for Input
    int userStep = 0;
    unsigned long startTime = millis();
    
    while (userStep < 3 && (millis() - startTime < 15000)) {
        float d = getDist();
        int input = -1;
        if (d < 10.0) input = 2;
        else if (d > 16.0 && d < 28.0) input = 1;
        else if (d > 45.0 && d < 80.0) input = 0;

        if (input != -1) {
            if (input == pattern[userStep]) {
                tone(BUZZER, 1200, 100); userStep++; delay(1000);
            } else {
                tone(BUZZER, 100, 800); delay(1000);
                modeMemory = false; return; // Fail -> Exit
            }
        }
        delay(50);
    }

    if (userStep >= 3) {
        // Success
        digitalWrite(LED_GREEN, HIGH);
        for(int i=0; i<3; i++) { tone(BUZZER, 800+i*400, 150); delay(200); }
        delay(2000);
    }
    
    modeMemory = false; // Back to Fidelity Mode
}
