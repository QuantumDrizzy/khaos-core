/**
 * sovereignty_monitor.cpp
 * khaos-core — Cognitive Sovereignty Guardian
 *
 * Responsibilities:
 *   1. Guard the dissipation gate: block any non-user-initiated alpha application
 *   2. Maintain an append-only, SHA-256-chained audit log
 *   3. Verify log chain integrity on startup
 *   4. Register and verify the hardware kill-switch heartbeat
 *
 * Build requirement: ETHICS_COMPLIANT must be defined (set by CMakeLists.txt).
 * This file will not compile without it — intentionally.
 */

#ifndef ETHICS_COMPLIANT
#  error "sovereignty_monitor.cpp requires ETHICS_COMPLIANT to be defined. \
See ETHICS.md and CMakeLists.txt. This is not a configurable option."
#endif

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── SHA-256 (vendored, no external dep for this core file) ────────────────────
// Minimal implementation — replace with libcrypto in production builds.
#include "sha256.h"   // expects: std::array<uint8_t,32> sha256(const uint8_t*, size_t)

// ── Safety constants (from include/safety_constants.h) ────────────────────────
#include "safety_constants.h"
// Expects:
//   constexpr float STIM_ABSOLUTE_MAX_AMP   (microamperes)
//   constexpr int   KILLSWITCH_TIMEOUT_MS

static_assert(STIM_ABSOLUTE_MAX_AMP <= 50.0f,
    "Stimulation ceiling exceeds 50 µA. See ETHICS.md §II for amendment process.");

// =============================================================================
// Types
// =============================================================================

using Timestamp = std::chrono::time_point<std::chrono::system_clock,
                                          std::chrono::nanoseconds>;
using Hash256   = std::array<uint8_t, 32>;

/**
 * Every event that touches the alpha-blend or the stimulation pipeline
 * produces one LogEntry. Entries are chained: each includes the hash
 * of the previous entry, making retroactive tampering detectable.
 */
struct LogEntry {
    // ── Identity ──────────────────────────────────────────────────────────────
    uint64_t    sequence_id;          // monotonically increasing, never reused
    int64_t     timestamp_ns;         // nanoseconds since Unix epoch

    // ── Event payload ─────────────────────────────────────────────────────────
    enum class EventType : uint8_t {
        SESSION_START        = 0x01,
        SESSION_END          = 0x02,
        DISSIPATION_APPLIED  = 0x10,  // alpha > 0 applied to theta-frame
        DISSIPATION_BLOCKED  = 0x11,  // attempted without user intent — ALARM
        PHASE_TRANSITION     = 0x20,  // circuit breaker state change
        CALIBRATION_COMPLETE = 0x30,
        KILLSWITCH_TRIGGERED = 0x40,
        KILLSWITCH_TIMEOUT   = 0x41,
        INTEGRITY_VIOLATION  = 0xF0,  // log chain broken — ALARM
        INCIDENT             = 0xFF,  // see ETHICS.md §V
    } event_type;

    float       alpha_applied;        // 0.0 if not a dissipation event
    float       confidence;           // NeuralPhaseVector.confidence at event time
    float       entropy;              // S(ρ) at event time
    bool        user_initiated;       // true = user intent confirmed via BP index
    char        note[128];            // human-readable context, null-terminated

    // ── Chain ─────────────────────────────────────────────────────────────────
    Hash256     prev_hash;            // SHA-256 of the previous serialized entry
    Hash256     self_hash;            // SHA-256 of this entry with self_hash zeroed

    // ── Software provenance ───────────────────────────────────────────────────
    uint32_t    version_hash;         // KHAOS_VERSION_HASH truncated to 32 bits
};

static_assert(sizeof(LogEntry) <= 512,
    "LogEntry is larger than expected — review for accidental field growth.");

// =============================================================================
// Internal helpers
// =============================================================================

namespace {

Hash256 hash_entry(const LogEntry& e) {
    // Hash with self_hash zeroed so the hash is of the data, not itself
    LogEntry copy = e;
    copy.self_hash.fill(0);
    return sha256(reinterpret_cast<const uint8_t*>(&copy), sizeof(copy));
}

std::string hex(const Hash256& h) {
    std::ostringstream ss;
    for (uint8_t b : h) ss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    return ss.str();
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

// =============================================================================
// SovereigntyMonitor
// =============================================================================

class SovereigntyMonitor {
public:
    // -------------------------------------------------------------------------
    // Construction & initialization
    // -------------------------------------------------------------------------

    explicit SovereigntyMonitor(std::filesystem::path log_path)
        : log_path_(std::move(log_path))
        , sequence_(0)
        , session_active_(false)
        , killswitch_registered_(false)
    {
        // Verify existing log chain before accepting any new entries
        if (std::filesystem::exists(log_path_)) {
            verify_chain_on_startup();
        }
    }

    ~SovereigntyMonitor() {
        if (session_active_) end_session();
    }

    // -------------------------------------------------------------------------
    // Session lifecycle
    // -------------------------------------------------------------------------

    void begin_session() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (session_active_) {
            throw std::runtime_error("SovereigntyMonitor: session already active");
        }
        if (!killswitch_registered_) {
            throw std::runtime_error(
                "SovereigntyMonitor: hardware kill-switch must be registered "
                "before starting a session. See ETHICS.md §III.");
        }
        session_active_ = true;
        append_entry(make_entry(LogEntry::EventType::SESSION_START,
                                /*alpha=*/0.0f, /*confidence=*/0.0f,
                                /*entropy=*/0.0f, /*user_initiated=*/true,
                                "Session started"));
    }

    void end_session() {
        std::lock_guard<std::mutex> lock(mutex_);
        session_active_ = false;
        append_entry(make_entry(LogEntry::EventType::SESSION_END,
                                0.0f, 0.0f, 0.0f, true, "Session ended"));
    }

    // -------------------------------------------------------------------------
    // The dissipation gate — the core sovereignty enforcement point
    // -------------------------------------------------------------------------

    /**
     * Call this BEFORE applying any non-zero alpha to the theta-frame.
     *
     * @param alpha          The alpha value about to be applied (0 = no dissipation)
     * @param confidence     Current NeuralPhaseVector confidence
     * @param entropy        Current S(ρ)
     * @param user_initiated True only if the user's BP index confirms active intent
     *                       toward the rest landmark in this frame.
     *
     * @returns true  — the dissipation is permitted; log entry written.
     * @returns false — BLOCKED. Log entry written as DISSIPATION_BLOCKED.
     *                  Caller MUST NOT apply the alpha. A DISSIPATION_BLOCKED
     *                  event triggers the incident response protocol.
     */
    [[nodiscard]]
    bool request_dissipation(float alpha, float confidence, float entropy,
                              bool user_initiated)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (alpha <= 0.0f) return true;  // nothing to gate

        if (!session_active_) {
            // Should never happen — defensive check
            incident("Dissipation requested outside active session");
            return false;
        }

        if (!user_initiated) {
            // ETHICS.md §II: dissipation without user intent is prohibited
            append_entry(make_entry(
                LogEntry::EventType::DISSIPATION_BLOCKED,
                alpha, confidence, entropy, false,
                "BLOCKED: dissipation without confirmed user intent"));

            trigger_incident(
                "Autonomous dissipation attempt detected. "
                "System requires manual restart. See ETHICS.md §V.");

            return false;
        }

        // Permitted — log it
        append_entry(make_entry(
            LogEntry::EventType::DISSIPATION_APPLIED,
            alpha, confidence, entropy, true,
            "User-initiated dissipation applied"));

        return true;
    }

    // -------------------------------------------------------------------------
    // Kill-switch registration and heartbeat
    // -------------------------------------------------------------------------

    /**
     * Must be called before begin_session().
     * Registers the callback that will be invoked if the software heartbeat
     * to the FPGA times out.
     *
     * In production, fpga_cut_all_outputs is a direct PCIe register write
     * to the stimulation FPGA's emergency-stop register.
     */
    void register_killswitch(std::function<void()> fpga_cut_all_outputs) {
        std::lock_guard<std::mutex> lock(mutex_);
        killswitch_callback_ = std::move(fpga_cut_all_outputs);
        killswitch_registered_ = true;
        last_heartbeat_ns_.store(now_ns());
    }

    /**
     * Call once per frame from the GPU pipeline host loop.
     * The FPGA watchdog independently monitors this — but the software
     * also tracks it to log timeouts.
     */
    void heartbeat() {
        last_heartbeat_ns_.store(now_ns());
    }

    /**
     * Must be called from a watchdog thread, e.g. every 1 ms.
     * If > KILLSWITCH_TIMEOUT_MS have elapsed without a heartbeat,
     * triggers the hardware cut and logs the event.
     */
    void check_heartbeat() {
        int64_t elapsed_ms = (now_ns() - last_heartbeat_ns_.load()) / 1'000'000LL;
        if (elapsed_ms > KILLSWITCH_TIMEOUT_MS) {
            trigger_killswitch_timeout();
        }
    }

    // -------------------------------------------------------------------------
    // Phase transition logging (circuit breaker state changes)
    // -------------------------------------------------------------------------

    void log_phase_transition(const char* from_state, const char* to_state,
                               float confidence, float entropy) {
        std::lock_guard<std::mutex> lock(mutex_);
        char note[128];
        std::snprintf(note, sizeof(note), "%s -> %s", from_state, to_state);
        append_entry(make_entry(LogEntry::EventType::PHASE_TRANSITION,
                                0.0f, confidence, entropy, true, note));
    }

    void log_calibration_complete(float quality_score) {
        std::lock_guard<std::mutex> lock(mutex_);
        char note[128];
        std::snprintf(note, sizeof(note),
                      "Calibration complete. Quality score: %.3f", quality_score);
        append_entry(make_entry(LogEntry::EventType::CALIBRATION_COMPLETE,
                                0.0f, quality_score, 0.0f, true, note));
    }

    // -------------------------------------------------------------------------
    // Chain integrity verification (called on startup)
    // -------------------------------------------------------------------------

    struct VerificationResult {
        bool     ok;
        uint64_t entries_checked;
        uint64_t broken_at_sequence;  // valid only if !ok
        std::string detail;
    };

    VerificationResult verify_chain() const {
        std::ifstream f(log_path_, std::ios::binary);
        if (!f) return {false, 0, 0, "Cannot open log file"};

        Hash256 expected_prev;
        expected_prev.fill(0);  // genesis: prev_hash of first entry is all-zeros

        LogEntry entry{};
        uint64_t count = 0;

        while (f.read(reinterpret_cast<char*>(&entry), sizeof(LogEntry))) {
            // Check prev_hash matches what we computed from the previous entry
            if (entry.prev_hash != expected_prev) {
                return {false, count, entry.sequence_id,
                        "Chain broken: prev_hash mismatch at sequence "
                        + std::to_string(entry.sequence_id)};
            }
            // Check self_hash
            Hash256 computed = hash_entry(entry);
            if (computed != entry.self_hash) {
                return {false, count, entry.sequence_id,
                        "Entry hash invalid at sequence "
                        + std::to_string(entry.sequence_id)};
            }
            expected_prev = entry.self_hash;
            ++count;
        }

        return {true, count, 0, "OK — " + std::to_string(count) + " entries verified"};
    }

private:
    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    LogEntry make_entry(LogEntry::EventType type,
                        float alpha, float confidence, float entropy,
                        bool user_initiated, const char* note) const
    {
        LogEntry e{};
        e.sequence_id   = sequence_++;
        e.timestamp_ns  = now_ns();
        e.event_type    = type;
        e.alpha_applied = alpha;
        e.confidence    = confidence;
        e.entropy       = entropy;
        e.user_initiated = user_initiated;
        std::strncpy(e.note, note, sizeof(e.note) - 1);
        e.version_hash  = KHAOS_VERSION_HASH & 0xFFFFFFFF;
        e.prev_hash     = last_hash_;
        e.self_hash.fill(0);  // zeroed before hashing
        e.self_hash     = hash_entry(e);
        return e;
    }

    void append_entry(const LogEntry& entry) {
        // Append-only: open in append+binary mode. Never seek, never truncate.
        std::ofstream f(log_path_,
                        std::ios::binary | std::ios::app | std::ios::out);
        if (!f) {
            throw std::runtime_error(
                "SovereigntyMonitor: cannot write to audit log at "
                + log_path_.string());
        }
        f.write(reinterpret_cast<const char*>(&entry), sizeof(LogEntry));
        f.flush();  // fsync equivalent — ensure durability before returning
        last_hash_ = entry.self_hash;
    }

    void verify_chain_on_startup() {
        auto result = verify_chain();
        if (!result.ok) {
            // Log the violation (to a separate incident file — the main log may
            // be compromised)
            std::ofstream incident_file(log_path_.string() + ".incident",
                                        std::ios::app);
            incident_file << "[" << now_ns() << "] INTEGRITY VIOLATION: "
                          << result.detail << "\n";
            incident_file.flush();

            throw std::runtime_error(
                "KHAOS INTEGRITY VIOLATION: audit log chain is broken.\n"
                "Detail: " + result.detail + "\n"
                "The system will not start. See ETHICS.md §V.");
        }
        // Replay log to find the last hash (needed to continue the chain)
        replay_for_last_hash();
    }

    void replay_for_last_hash() {
        std::ifstream f(log_path_, std::ios::binary);
        if (!f) return;

        LogEntry entry{};
        last_hash_.fill(0);
        sequence_ = 0;
        while (f.read(reinterpret_cast<char*>(&entry), sizeof(LogEntry))) {
            last_hash_ = entry.self_hash;
            sequence_  = entry.sequence_id + 1;
        }
    }

    void trigger_killswitch_timeout() {
        // Log before cutting — if this write fails, the FPGA has already cut anyway
        {
            std::lock_guard<std::mutex> lock(mutex_);
            append_entry(make_entry(
                LogEntry::EventType::KILLSWITCH_TIMEOUT,
                0.0f, 0.0f, 0.0f, false,
                "Heartbeat timeout — FPGA cut triggered"));
        }
        if (killswitch_callback_) killswitch_callback_();
    }

    void trigger_incident(const char* reason) {
        // Called with mutex_ already held
        append_entry(make_entry(
            LogEntry::EventType::INCIDENT,
            0.0f, 0.0f, 0.0f, false, reason));

        // Signal the runtime to halt stimulation and alert user
        incident_flag_.store(true);

        if (killswitch_callback_) killswitch_callback_();
    }

    void incident(const char* reason) {
        // Called without lock — acquires internally
        std::lock_guard<std::mutex> lock(mutex_);
        trigger_incident(reason);
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    std::filesystem::path       log_path_;
    mutable std::mutex          mutex_;
    mutable uint64_t            sequence_;
    Hash256                     last_hash_{};
    bool                        session_active_;
    bool                        killswitch_registered_;
    std::function<void()>       killswitch_callback_;
    std::atomic<int64_t>        last_heartbeat_ns_{0};
    std::atomic<bool>           incident_flag_{false};

public:
    // Public read of incident flag — runtime checks this each frame
    bool incident_active() const { return incident_flag_.load(); }
};
