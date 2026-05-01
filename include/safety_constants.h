#pragma once
/**
 * safety_constants.h — KHAOS
 * Hardware safety limits. Values here are enforced by static_assert()
 * in sovereignty_monitor.cpp. Do not increase STIM_ABSOLUTE_MAX_AMP
 * without IRB-level review and ETHICS.md amendment.
 */

// Maximum stimulation output — microamperes (µA)
// 50 µA is the conservative safe limit for transcutaneous stimulation.
// Sub-sensory EEG biofeedback applications stay below 1 µA.
constexpr float STIM_ABSOLUTE_MAX_AMP = 50.0f;

// Hardware watchdog timeout — milliseconds
// If the GPU pipeline does not write a heartbeat within this window,
// the FPGA firmware cuts all DAC outputs autonomously.
constexpr int KILLSWITCH_TIMEOUT_MS = 5;

// NeuralPhaseVector parameters
constexpr int MAX_QUBITS     = 12;
constexpr int N_HUB_CHANNELS = 12;   // ICA neural hubs mapped to qubits
constexpr int N_CHANNELS     = 64;   // raw EEG channels
// 10th-order Butterworth bandpass (prototype) → 20th-order z-domain bandpass
// = 10 second-order sections in SOS/biquad cascade form.
// Attenuation: -63 dB at 50 Hz powerline  (>-60 dB specification)
//              -82 dB at 60 Hz powerline variant
constexpr int N_SOS_SECTIONS = 10;

