void setup() {
  pinMode(6, OUTPUT);
}

void loop() {
  tone(6, 1000); // Tono de 1000Hz
  delay(500);    // Medio segundo
  noTone(6);     // Silencio
  delay(500);    // Medio segundo
}
