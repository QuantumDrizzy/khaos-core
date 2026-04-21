/*
 * khaos_tactile.ino — khaos-core Arduino Tactile Feedback Controller
 * ==================================================================
 * Receives binary frames from the PC over Serial (115200 baud).
 * Drives vibration motors, buzzers, and LEDs from the 37-sensor kit.
 *
 * Pin Map (adjust to your wiring):
 *   PWM outputs (vibration motors / buzzers):
 *     Pin 3  → Hub 0 (motor vibration)
 *     Pin 5  → Hub 1 (motor vibration)
 *     Pin 6  → Hub 2 (buzzer passive)
 *     Pin 9  → Hub 3 (motor vibration)
 *
 *   Digital outputs (circuit breaker LEDs):
 *     Pin 10 → GREEN  LED (NOMINAL)
 *     Pin 11 → YELLOW LED (DEGRADED / RECOVERING)
 *     Pin 12 → RED    LED (PANIC)
 *     Pin 13 → Onboard LED (heartbeat)
 *
 * Protocol (binary, little-endian):
 *   Byte 0:      0xAA (sync byte)
 *   Byte 1:      0x55 (sync byte)
 *   Byte 2-3:    sovereignty_token (uint16)
 *   Byte 4:      circuit_state (0=NOMINAL, 1=DEGRADED, 2=PANIC, 3=RECOVERING)
 *   Byte 5-8:    pwm_duty[0..3] (uint8 each, scaled 0-255)
 *   Byte 9-12:   fm_freq[0..3]  (uint8 each, scaled 0-255 → 50-300 Hz)
 *   Byte 13:     checksum (XOR of bytes 2-12)
 *   Total: 14 bytes per frame @ 60 Hz = 840 bytes/sec
 *
 * Safety:
 *   - Watchdog: if no valid frame in 100ms → all outputs zeroed
 *   - PANIC state: all PWM zeroed regardless of duty values
 *   - Max PWM clamped to 200/255 (78%) for motor safety
 */

#define SYNC_0        0xAA
#define SYNC_1        0x55
#define FRAME_SIZE    14
#define WATCHDOG_MS   100
#define MAX_PWM       200    // 78% max duty for motor safety
#define N_CHANNELS    4

// PWM output pins (must be PWM-capable: 3, 5, 6, 9, 10, 11)
const uint8_t PWM_PINS[N_CHANNELS] = {3, 5, 6, 9};

// Status LED pins
const uint8_t LED_GREEN  = 10;
const uint8_t LED_YELLOW = 11;
const uint8_t LED_RED    = 12;
const uint8_t LED_HEART  = 13;

// State
uint8_t rx_buf[FRAME_SIZE];
uint8_t rx_idx = 0;
unsigned long last_valid_frame = 0;
unsigned long last_heartbeat = 0;
bool heartbeat_state = false;

// Current output state
uint8_t current_pwm[N_CHANNELS] = {0};
uint8_t current_state = 0;  // NOMINAL
uint16_t current_token = 0;
uint32_t frame_count = 0;

void setup() {
    Serial.begin(115200);

    for (int i = 0; i < N_CHANNELS; i++) {
        pinMode(PWM_PINS[i], OUTPUT);
        analogWrite(PWM_PINS[i], 0);
    }

    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_HEART, OUTPUT);

    // Boot sequence: flash all LEDs
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_YELLOW, HIGH);
    digitalWrite(LED_RED, HIGH);
    delay(300);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, LOW);

    // Start in NOMINAL
    digitalWrite(LED_GREEN, HIGH);
    last_valid_frame = millis();

    // Send ready signal
    Serial.println("[KHAOS-TACTILE] READY v1.0");
}

void zero_outputs() {
    for (int i = 0; i < N_CHANNELS; i++) {
        analogWrite(PWM_PINS[i], 0);
        current_pwm[i] = 0;
    }
}

void update_leds(uint8_t state) {
    digitalWrite(LED_GREEN,  state == 0 ? HIGH : LOW);  // NOMINAL
    digitalWrite(LED_YELLOW, (state == 1 || state == 3) ? HIGH : LOW);  // DEGRADED/RECOVERING
    digitalWrite(LED_RED,    state == 2 ? HIGH : LOW);  // PANIC
}

void apply_frame() {
    uint16_t token = rx_buf[2] | (rx_buf[3] << 8);
    uint8_t state  = rx_buf[4];

    current_token = token;
    current_state = state;

    // Update LEDs
    update_leds(state);

    if (state == 2) {
        // PANIC — zero everything regardless
        zero_outputs();
        return;
    }

    // Apply PWM values
    for (int i = 0; i < N_CHANNELS; i++) {
        uint8_t duty = rx_buf[5 + i];
        // Clamp for motor safety
        if (duty > MAX_PWM) duty = MAX_PWM;
        current_pwm[i] = duty;
        analogWrite(PWM_PINS[i], duty);
    }

    // FM frequencies: for buzzer channels, use tone()
    // Pin 6 (hub 2) is buzzer — map 0-255 to 50-300 Hz
    uint8_t fm_raw = rx_buf[9 + 2];  // fm_freq[2] for buzzer channel
    if (fm_raw > 0 && current_pwm[2] > 10) {
        uint16_t freq = 50 + (uint16_t)fm_raw;  // 50-305 Hz
        tone(PWM_PINS[2], freq);
    } else {
        noTone(PWM_PINS[2]);
    }

    frame_count++;
}

uint8_t compute_checksum() {
    uint8_t cs = 0;
    for (int i = 2; i < FRAME_SIZE - 1; i++) {
        cs ^= rx_buf[i];
    }
    return cs;
}

void loop() {
    unsigned long now = millis();

    // ── Read serial data ─────────────────────────────────────────
    while (Serial.available()) {
        uint8_t b = Serial.read();

        if (rx_idx == 0 && b != SYNC_0) continue;
        if (rx_idx == 1 && b != SYNC_1) { rx_idx = 0; continue; }

        rx_buf[rx_idx++] = b;

        if (rx_idx == FRAME_SIZE) {
            rx_idx = 0;

            // Validate checksum
            if (rx_buf[FRAME_SIZE - 1] == compute_checksum()) {
                apply_frame();
                last_valid_frame = now;

                // ACK: send frame count back
                Serial.write(0xAC);
                Serial.write((frame_count >> 0) & 0xFF);
                Serial.write((frame_count >> 8) & 0xFF);
            }
        }
    }

    // ── Watchdog: no valid frame in 100ms → zero all ─────────────
    if (now - last_valid_frame > WATCHDOG_MS) {
        zero_outputs();
        // Flash red LED as warning
        if ((now / 200) % 2 == 0) {
            digitalWrite(LED_RED, HIGH);
            digitalWrite(LED_GREEN, LOW);
        } else {
            digitalWrite(LED_RED, LOW);
        }
    }

    // ── Heartbeat LED (500ms toggle) ─────────────────────────────
    if (now - last_heartbeat > 500) {
        heartbeat_state = !heartbeat_state;
        digitalWrite(LED_HEART, heartbeat_state ? HIGH : LOW);
        last_heartbeat = now;
    }
}
