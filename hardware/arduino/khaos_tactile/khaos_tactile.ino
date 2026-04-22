/*
 * KĦAOS-CORE MASTER HYBRID FIRMWARE v3.2
 * --------------------------------------
 * - Default: Neural Presence (Proximity)
 * - Secret: Hold <5cm for 3s to trigger Memory Test
 * - Telemetry: "Dist: Xcm" (Used by Python for Security Lock)
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
    neuralArm.write(0);
    randomSeed(analogRead(A5));
    
    // Boot Beep
    tone(BUZZER, 1000, 100); delay(200);
    Serial.println("KHAOS HYBRID v3.2 ONLINE");
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
        float d = getDist();
        
        // --- TELEMETRY FOR PYTHON (CRITICAL) ---
        Serial.print("Dist: "); Serial.print(d, 1); Serial.println("cm");
        
        // Mode Switch: Hold < 5cm for 3s
        if (d < 5.0 && d > 0.1) {
            if (modeSwitchStart == 0) modeSwitchStart = millis();
            if (millis() - modeSwitchStart > 3000) {
                modeMemory = true; modeSwitchStart = 0; return;
            }
            digitalWrite(LED_GREEN, (millis()/100)%2); // Feedback de carga
        } else { modeSwitchStart = 0; }

        float targetFid = constrain(1.0 - ((d - 5.0) / 30.0), 0.0, 1.0);
        smoothFid += (targetFid - smoothFid) * 0.25;

        // Feedback
        if (smoothFid > 0.05) tone(BUZZER, 200 + (uint16_t)(smoothFid * 1400.0));
        else noTone(BUZZER);
        
        neuralArm.write((int)(smoothFid * 180.0));
        
        if (smoothFid > 0.7) { digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_RED, LOW); }
        else { digitalWrite(LED_GREEN, LOW); digitalWrite(LED_RED, HIGH); }

        delay(40);
    } else {
        runMemoryMode();
    }
}

void runMemoryMode() {
    Serial.println(">>> ENTERING MEMORY CHALLENGE <<<");
    int pattern[3];
    for (int i = 0; i < 3; i++) {
        pattern[i] = random(0, 3);
        if (pattern[i] == 0) { tone(BUZZER, 300, 400); neuralArm.write(10); }
        else if (pattern[i] == 1) { tone(BUZZER, 800, 400); neuralArm.write(90); }
        else { tone(BUZZER, 1600, 400); neuralArm.write(175); }
        delay(600); noTone(BUZZER); delay(200);
    }
    
    Serial.println("YOUR TURN...");
    int userStep = 0;
    unsigned long startTime = millis();
    while (userStep < 3 && (millis() - startTime < 10000)) {
        float d = getDist();
        int input = -1;
        if (d < 10.0) input = 2;
        else if (d > 16.0 && d < 28.0) input = 1;
        else if (d > 45.0 && d < 80.0) input = 0;
        
        if (input != -1) {
            if (input == pattern[userStep]) { 
                tone(BUZZER, 1200, 100); userStep++; delay(1000); 
            } else { 
                tone(BUZZER, 100, 800); delay(1000); modeMemory = false; return; 
            }
        }
        delay(50);
    }
    
    if (userStep >= 3) {
        Serial.println(">>> CHALLENGE PASSED! <<<");
        digitalWrite(LED_GREEN, HIGH);
        for(int i=0; i<3; i++) { tone(BUZZER, 800+i*400, 150); delay(200); }
        delay(1000);
    }
    modeMemory = false;
}
