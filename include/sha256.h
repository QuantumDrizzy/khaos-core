#pragma once
/**
 * sha256.h — minimal SHA-256 interface
 *
 * In production, link against libcrypto (OpenSSL) or liboqs hash primitives.
 * This header provides the declaration expected by sovereignty_monitor.cpp.
 *
 * Replace the stub implementation in src/security/sha256.cpp with a real one.
 */
#include <array>
#include <cstdint>
#include <cstddef>

std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len);

