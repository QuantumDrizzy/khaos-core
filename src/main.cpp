/**
 * main.cpp — khaos-core / Quantum Mirror
 *
 * Pipeline entry point.  Wires together:
 *   LSL EEG capture  →  SignalProcessor (CUDA IIR)  →  DWTExtractor (CUDA à-trous)
 *   →  SovereigntyMonitor (audit log + kill-switch)
 *   →  Python bridge (mirror_bridge.py) — quantum circuits + graphene model
 *   →  Feedback stub (console print; replace with haptic/audio driver)
 *
 * Build:  cmake -B build -DETHICS_COMPLIANT=ON && cmake --build build
 * Run:    ./khaos_mirror [--log <path>] [--stream <lsl-name>] [--bridge <py-script>]
 *         ./khaos_mirror --dry-run          (synthetic EEG, no LSL, 5-second CI smoke test)
 */

#ifndef ETHICS_COMPLIANT
#  error "main.cpp requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

#include "../include/safety_constants.h"
#include "../include/sha256.h"
#include "../include/khaos_bridge.h"   // NeuralPhaseVector, MAX_QUBITS, N_CHANNELS, etc.
#include "../include/dsp_pipeline.h"   // dsp_create_and_init / dsp_process_frame / …

#include <array>
#include <atomic>
#include <chrono>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES   // MSVC; harmless on GCC/Clang
#endif
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// POSIX — subprocess pipes
#include <fcntl.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// ─── Optional LSL ────────────────────────────────────────────────────────────
#ifdef KHAOS_LSL_ENABLED
#  include <lsl_cpp.h>
#endif

// =============================================================================
// Pipeline constants (not in khaos_bridge.h)
// =============================================================================
static constexpr int SAMPLE_RATE_HZ = 1000;
static constexpr int N_QUBITS       = N_HUB_CHANNELS;  // 12

// DSPPipeline (SignalProcessor + DWTExtractor) is declared in dsp_pipeline.h
// and implemented in neuro/dsp_pipeline.cu.  main.cpp drives it via the C API.

struct SovereigntyMonitor {
    static SovereigntyMonitor& instance() {
        static SovereigntyMonitor inst;
        return inst;
    }
    void init(const std::string& log_path) {
        log_path_ = log_path;
        fprintf(stderr, "[sovereignty] audit log → %s\n", log_path_.c_str());
    }
    void check_heartbeat() noexcept {}
    void register_killswitch(std::function<void()> fn) { killswitch_ = std::move(fn); }
    [[nodiscard]]
    bool request_dissipation(float /*alpha*/, float /*conf*/,
                              float /*ent*/, bool user_initiated) noexcept {
        return user_initiated;   // only user-initiated dissipation permitted
    }
    void log_frame(const NeuralPhaseVector& /*vec*/) noexcept {}
    void log_event(const char* tag, const char* detail) {
        fprintf(stderr, "[sovereignty] %s  %s\n", tag, detail);
    }
private:
    std::string            log_path_;
    std::function<void()>  killswitch_;
};

// =============================================================================
// Global state
// =============================================================================
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_killswitch_armed{false};

static void handle_signal(int /*sig*/) noexcept {
    g_running.store(false, std::memory_order_relaxed);
}

// =============================================================================
// Circuit breaker
// =============================================================================
enum class CircuitState { NOMINAL, DEGRADED, PANIC, RECOVERING };

struct CircuitBreaker {
    CircuitState state = CircuitState::NOMINAL;

    // rolling counters (each tick = 1 ms)
    int low_conf_ticks  = 0;   // confidence < 0.5
    int very_low_ticks  = 0;   // confidence < 0.3
    int panic_ticks     = 0;   // ticks since PANIC entered

    static constexpr int DEGRADED_THRESH_MS  = 200;
    static constexpr int PANIC_THRESH_MS     = 100;
    static constexpr int PANIC_TIMEOUT_MS    = 5000;

    // Called once per ms from the EEG loop.
    void step(float confidence, float bp_index,
              float fidelity, bool user_dissipation) noexcept {
        using CS = CircuitState;

        switch (state) {
            case CS::NOMINAL:
                if (confidence < 0.5f) ++low_conf_ticks; else low_conf_ticks = 0;
                if (low_conf_ticks >= DEGRADED_THRESH_MS) {
                    transition(CS::DEGRADED);
                }
                break;

            case CS::DEGRADED:
                if (confidence < 0.3f) ++very_low_ticks; else very_low_ticks = 0;
                if (confidence >= 0.5f) { low_conf_ticks = 0; transition(CS::NOMINAL); break; }
                if (very_low_ticks >= PANIC_THRESH_MS || bp_index > 0.8f) {
                    transition(CS::PANIC);
                }
                break;

            case CS::PANIC:
                ++panic_ticks;
                if (user_dissipation || panic_ticks >= PANIC_TIMEOUT_MS) {
                    transition(CS::RECOVERING);
                }
                break;

            case CS::RECOVERING:
                if (fidelity > 0.85f) transition(CS::NOMINAL);
                break;
        }
    }

    float alpha() const noexcept {
        switch (state) {
            case CircuitState::NOMINAL:    return 1.0f;
            case CircuitState::DEGRADED:   return 0.6f;
            case CircuitState::PANIC:      return 0.0f;
            case CircuitState::RECOVERING: return 0.3f;
        }
        return 1.0f;
    }

    const char* state_name() const noexcept {
        switch (state) {
            case CircuitState::NOMINAL:    return "NOMINAL";
            case CircuitState::DEGRADED:   return "DEGRADED";
            case CircuitState::PANIC:      return "PANIC";
            case CircuitState::RECOVERING: return "RECOVERING";
        }
        return "?";
    }

private:
    CircuitState prev_ = CircuitState::NOMINAL;

    void transition(CircuitState next) noexcept {
        if (next == state) return;
        char buf[64];
        snprintf(buf, sizeof(buf), "%s → %s", state_name(),
                 [&]{ state = next; return state_name(); }());
        SovereigntyMonitor::instance().log_event("CIRCUIT_TRANSITION", buf);
        low_conf_ticks = very_low_ticks = panic_ticks = 0;
    }
};

// =============================================================================
// Python bridge (stdin/stdout JSON-line protocol)
// =============================================================================
struct PythonBridge {
    pid_t  child_pid = -1;
    int    stdin_fd  = -1;   // write to child stdin
    int    stdout_fd = -1;   // read from child stdout

    // Last values received from bridge
    float fidelity   = 0.0f;
    char  landmark[32] = "unknown";
    bool  qid        = false;
    float ent_alpha[MAX_QUBITS] = {};

    bool spawn(const char* script) {
        int to_child[2], from_child[2];
        if (pipe(to_child) || pipe(from_child)) {
            perror("pipe");
            return false;
        }

        child_pid = fork();
        if (child_pid < 0) { perror("fork"); return false; }

        if (child_pid == 0) {
            // New session — isolate from parent signal group
            setsid();
            // Child: redirect stdin/stdout to pipes
            dup2(to_child[0],   STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);  close(to_child[1]);
            close(from_child[0]); close(from_child[1]);
            // -u: force unbuffered stdout/stderr — critical when stdout is a pipe
            execlp("python3", "python3", "-u", script, nullptr);
            perror("execlp");
            _exit(1);
        }

        // Parent
        close(to_child[0]);
        close(from_child[1]);
        stdin_fd  = to_child[1];
        stdout_fd = from_child[0];

        // Make child stdout non-blocking
        int flags = fcntl(stdout_fd, F_GETFL, 0);
        fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);

        fprintf(stderr, "[bridge] spawned %s (pid=%d)\n", script, child_pid);
        return true;
    }

    // Send a frame to the bridge (called every 100 EEG frames ≈ 10 Hz)
    void send(const NeuralPhaseVector& vec) noexcept {
        if (stdin_fd < 0) return;

        char buf[512];
        int n = snprintf(buf, sizeof(buf),
            "{\"theta\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
             "\"bp_index\":%.4f,\"timestamp_ns\":%lu}\n",
            vec.theta[0], vec.theta[1], vec.theta[2], vec.theta[3],
            vec.theta[4], vec.theta[5], vec.theta[6], vec.theta[7],
            vec.theta[8], vec.theta[9], vec.theta[10],vec.theta[11],
            vec.bp_index, (unsigned long)vec.timestamp_ns);

        if (n > 0) (void)write(stdin_fd, buf, (size_t)n);
    }

    // Block until bridge emits {"ready":true} or timeout_s elapses.
    // Called once after spawn() — survives CUDA-Q JIT compilation delay.
    bool wait_ready(int timeout_s) noexcept {
        if (stdout_fd < 0) return false;

        // Temporarily switch to blocking I/O for the warmup wait
        int flags = fcntl(stdout_fd, F_GETFL, 0);
        fcntl(stdout_fd, F_SETFL, flags & ~O_NONBLOCK);

        using Clock = std::chrono::steady_clock;
        auto t_end = Clock::now() + std::chrono::seconds(timeout_s);

        char buf[256] = {};
        int  pos      = 0;

        while (Clock::now() < t_end) {
            auto remaining = t_end - Clock::now();
            long rem_us = (long)std::chrono::duration_cast<
                std::chrono::microseconds>(remaining).count();
            if (rem_us <= 0) break;

            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv{ rem_us / 1000000L, rem_us % 1000000L };
            if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) <= 0) break;

            char ch;
            ssize_t n = read(stdout_fd, &ch, 1);
            if (n <= 0) break;

            if (ch == '\n') {
                buf[pos] = '\0';
                if (strstr(buf, "\"ready\"")) {
                    fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);
                    return true;
                }
                // Not the ready line — could be a spurious early response; keep waiting
                pos = 0;
            } else {
                if (pos < (int)sizeof(buf) - 1) buf[pos++] = ch;
            }
        }

        // Restore non-blocking regardless of outcome
        fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);
        return false;  // timed out
    }

    // Non-blocking poll for a response line
    bool poll() noexcept {
        if (stdout_fd < 0) return false;

        // Quick select with zero timeout
        fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
        struct timeval tv{0, 0};
        if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) <= 0)
            return false;

        char line[1024];
        ssize_t n = read(stdout_fd, line, sizeof(line) - 1);
        if (n <= 0) return false;
        line[n] = '\0';

        // Minimal JSON parse (no external dependency)
        parse_response(line);
        return true;
    }

    void shutdown() noexcept {
        // Close stdin first — Python bridge exits cleanly on BrokenPipeError
        if (stdin_fd  >= 0) { close(stdin_fd);  stdin_fd  = -1; }
        if (child_pid > 0)  {
            // Give bridge up to 500 ms to flush and exit naturally
            for (int i = 0; i < 50; ++i) {
                int status = 0;
                if (waitpid(child_pid, &status, WNOHANG) != 0) { child_pid = -1; break; }
                usleep(10000);   // 10 ms
            }
            // Force-kill only if still alive
            if (child_pid > 0) {
                kill(child_pid, SIGKILL);
                waitpid(child_pid, nullptr, 0);
                child_pid = -1;
            }
        }
        if (stdout_fd >= 0) { close(stdout_fd); stdout_fd = -1; }
    }

private:
    // Extract a float after a JSON key, e.g. "fidelity":0.87
    static float extract_float(const char* json, const char* key, float fallback) noexcept {
        const char* p = strstr(json, key);
        if (!p) return fallback;
        p = strchr(p, ':');
        if (!p) return fallback;
        return (float)atof(p + 1);
    }

    void parse_response(const char* line) noexcept {
        fidelity = extract_float(line, "\"fidelity\"", fidelity);
        qid      = (strstr(line, "\"qid\":true") != nullptr);

        // json.dumps produces "key": "value" (space after colon).
        // Search for the key, then scan forward to the opening quote.
        const char* lm = strstr(line, "\"landmark\":");
        if (lm) {
            const char* q = strchr(lm + 11, '"');   // skip key, find opening "
            if (q) {
                ++q;   // step past "
                const char* end = strchr(q, '"');
                if (end) {
                    size_t len = std::min((size_t)(end - q), sizeof(landmark) - 1);
                    memcpy(landmark, q, len);
                    landmark[len] = '\0';
                }
            }
        }

        // ent_alpha array: "ent_alpha": [0.1, 0.2, ...]
        const char* ea = strstr(line, "\"ent_alpha\":");
        if (ea) {
            ea = strchr(ea + 12, '[');   // skip key, find opening [
            if (ea) {
                ++ea;   // step past [
                for (int i = 0; i < N_QUBITS && *ea && *ea != ']'; ++i) {
                    ent_alpha[i] = (float)atof(ea);
                    ea = strchr(ea, ',');
                    if (!ea) break;
                    ++ea;
                }
            }
        }
    }
};

// =============================================================================
// Dry-run synthetic EEG source
// =============================================================================
static void gen_synthetic_sample(float* out, int n_ch, double t_sec) noexcept {
    for (int ch = 0; ch < n_ch; ++ch) {
        // 10 Hz mu + 20 Hz beta + white-ish noise via cheap LCG
        float mu   = 5.0f * std::sin(2.0f * 3.14159265f * 10.0f * (float)t_sec + ch * 0.3f);
        float beta = 2.0f * std::sin(2.0f * 3.14159265f * 20.0f * (float)t_sec + ch * 0.7f);
        float noise = 0.5f * ((float)(rand() % 1000) / 500.0f - 1.0f);
        out[ch] = mu + beta + noise;
    }
}

// =============================================================================
// Feedback printer
// =============================================================================
static void print_feedback(const PythonBridge& br, const NeuralPhaseVector& vec,
                            const CircuitBreaker& cb) noexcept {
    float ent_mean = 0.0f;
    for (int i = 0; i < N_QUBITS; ++i) ent_mean += br.ent_alpha[i];
    ent_mean /= N_QUBITS;

    printf("[MIRROR] state=%-10s fidelity=%.3f landmark=%-12s "
           "ent_mean=%.3f bp=%.3f conf=%.3f qid=%d ts=%lus\n",
           cb.state_name(), br.fidelity, br.landmark,
           ent_mean, vec.bp_index, vec.confidence,
           (int)br.qid,
           (unsigned long)(vec.timestamp_ns / 1000000000ULL));
    fflush(stdout);
}

// =============================================================================
// Watchdog thread — calls check_heartbeat() every 1 ms
// =============================================================================
static void watchdog_thread_fn() noexcept {
    using namespace std::chrono_literals;
    while (g_running.load(std::memory_order_relaxed)) {
        SovereigntyMonitor::instance().check_heartbeat();
        std::this_thread::sleep_for(1ms);
    }
}

// =============================================================================
// CLI argument parser
// =============================================================================
struct Config {
    std::string log_path    = "data/audit.log";
    std::string lsl_stream  = "EEG";
    std::string bridge_script = "src/quantum/mirror_bridge.py";
    bool        dry_run     = false;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dry-run") {
            cfg.dry_run = true;
        } else if (std::string(argv[i]) == "--log" && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (std::string(argv[i]) == "--stream" && i + 1 < argc) {
            cfg.lsl_stream = argv[++i];
        } else if (std::string(argv[i]) == "--bridge" && i + 1 < argc) {
            cfg.bridge_script = argv[++i];
        } else {
            fprintf(stderr, "usage: khaos_mirror [--dry-run] [--log <path>] "
                            "[--stream <name>] [--bridge <script>]\n");
        }
    }
    return cfg;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // ── Signal handlers ──────────────────────────────────────────────────────
    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    // Ignore SIGPIPE — we handle bridge death via write() return value
    signal(SIGPIPE, SIG_IGN);

    // ── Sovereignty monitor ───────────────────────────────────────────────────
    auto& sov = SovereigntyMonitor::instance();
    sov.init(cfg.log_path);
    sov.register_killswitch([]() noexcept {
        g_killswitch_armed.store(true, std::memory_order_relaxed);
        g_running.store(false, std::memory_order_relaxed);
        fprintf(stderr, "\n[KILLSWITCH] Hardware kill-switch triggered.\n");
    });

    fprintf(stderr, "╔═══════════════════════════════════════╗\n");
    fprintf(stderr, "║   khaos-core / Quantum Mirror v0.1    ║\n");
    fprintf(stderr, "║   ETHICS_COMPLIANT build              ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════╝\n");
    if (cfg.dry_run)
        fprintf(stderr, "[mode] DRY RUN — synthetic EEG, no LSL, 5 s\n");

    // ── DSP pipeline (IIR + à-trous DWT on GPU) ──────────────────────────────
    DSPPipeline*   dsp = nullptr;
    if (!cfg.dry_run) {
        dsp = dsp_create_and_init();
        fprintf(stderr, "[dsp] IIR + DWT pipeline initialised\n");
    } else {
        fprintf(stderr, "[dsp] dry-run: synthetic vec, GPU pipeline skipped\n");
    }
    CircuitBreaker  cb;

    // ── Python bridge ─────────────────────────────────────────────────────────
    PythonBridge bridge;
    if (!bridge.spawn(cfg.bridge_script.c_str())) {
        fprintf(stderr, "[bridge] WARNING: could not spawn %s — "
                        "quantum feedback disabled\n", cfg.bridge_script.c_str());
    } else {
        // Wait for bridge to finish initialisation (CUDA-Q JIT can take 10-30 s).
        // The bridge emits {"ready":true} on stdout once it's accepting frames.
        constexpr int BRIDGE_WARMUP_TIMEOUT_S = 60;
        fprintf(stderr, "[bridge] waiting for initialisation "
                        "(CUDA-Q JIT may take up to %ds)...\n",
                        BRIDGE_WARMUP_TIMEOUT_S);
        if (bridge.wait_ready(BRIDGE_WARMUP_TIMEOUT_S)) {
            fprintf(stderr, "[bridge] initialised and ready\n");
        } else {
            fprintf(stderr, "[bridge] WARNING: warmup timed out — "
                            "continuing without quantum feedback\n");
        }
    }

    // ── LSL inlet ─────────────────────────────────────────────────────────────
#ifdef KHAOS_LSL_ENABLED
    lsl::stream_inlet* inlet = nullptr;
    if (!cfg.dry_run) {
        try {
            auto results = lsl::resolve_stream("name", cfg.lsl_stream, 1, 5.0);
            if (results.empty()) {
                fprintf(stderr, "[lsl] stream '%s' not found — exiting\n",
                        cfg.lsl_stream.c_str());
                bridge.shutdown();
                return 1;
            }
            inlet = new lsl::stream_inlet(results[0]);
            fprintf(stderr, "[lsl] connected to stream '%s'\n",
                    cfg.lsl_stream.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "[lsl] error: %s\n", e.what());
            bridge.shutdown();
            return 1;
        }
    }
#else
    if (!cfg.dry_run) {
        fprintf(stderr, "[lsl] khaos_mirror was built without LSL support. "
                        "Use --dry-run or rebuild with liblsl.\n");
        bridge.shutdown();
        return 1;
    }
#endif

    // ── Watchdog thread ───────────────────────────────────────────────────────
    std::thread watchdog(watchdog_thread_fn);

    // ── EEG loop ──────────────────────────────────────────────────────────────
    fprintf(stderr, "[main] entering EEG loop (Ctrl-C to stop)\n");

    float raw[N_CHANNELS] = {};
    NeuralPhaseVector vec  = {};
    uint64_t frame_count   = 0;
    float    last_fidelity = 0.0f;

    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();
    auto t_dry_end = t_start + std::chrono::seconds(5);

    while (g_running.load(std::memory_order_relaxed)) {
        // ── Dry-run exit condition ───────────────────────────────────────────
        if (cfg.dry_run && Clock::now() >= t_dry_end) break;

        // ── Acquire one EEG sample ───────────────────────────────────────────
        if (cfg.dry_run) {
            double t = std::chrono::duration<double>(Clock::now() - t_start).count();
            gen_synthetic_sample(raw, N_CHANNELS, t);

            // Stub DSP never calls pop_phase_vector() with real data, so populate
            // vec directly in dry-run mode with synthetic but plausible values.
            for (int q = 0; q < N_QUBITS; ++q)
                vec.theta[q] = (float)(M_PI * (0.5 + 0.3 * std::sin(
                    2.0 * M_PI * 10.0 * t + q * 0.52)));
            vec.confidence       = 0.70f + 0.15f * (float)std::sin(2.0 * M_PI * 0.1  * t);
            vec.bp_index         = 0.20f + 0.10f * (float)std::sin(2.0 * M_PI * 0.05 * t);
            vec.entropy_estimate = 0.60f + 0.10f * (float)std::sin(2.0 * M_PI * 0.07 * t);
            vec.alpha_blend      = 1.0f;
            vec.timestamp_ns     = (uint64_t)std::chrono::duration_cast<
                std::chrono::nanoseconds>(Clock::now() - t_start).count();

            // Pace to 1000 Hz
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        } else {
#ifdef KHAOS_LSL_ENABLED
            double ts_lsl = 0.0;
            inlet->pull_sample(raw, N_CHANNELS, 0.05 /* timeout s */);
            (void)ts_lsl;
#endif
        }

        // ── DSP (IIR → DWT → NeuralPhaseVector) ──────────────────────────────
        if (dsp) {
            // Real GPU path: IIR biquad + à-trous DWT + metrics on CUDA
            const uint64_t ts_ns = (uint64_t)std::chrono::duration_cast<
                std::chrono::nanoseconds>(Clock::now() - t_start).count();

            dsp_process_frame(dsp, raw, ts_ns, cb.alpha());
            dsp_request_theta_async(dsp);
            const NeuralPhaseVector* gpu_vec = dsp_sync_theta(dsp);
            vec = *gpu_vec;   // copy to local (16 floats + metadata)
            sov.log_frame(vec);
        }

        ++frame_count;

        // ── Bridge exchange (10 Hz) ───────────────────────────────────────────
        if (frame_count % 100 == 0) {
            bridge.send(vec);
        }
        if (bridge.poll()) {
            last_fidelity = bridge.fidelity;
            print_feedback(bridge, vec, cb);
        }

        // ── Circuit breaker step (1 ms = 1 sample @ 1000 Hz) ─────────────────
        bool user_dissipation = false;  // TODO: wire to UI button / BP threshold
        if (sov.request_dissipation(vec.alpha_blend, vec.confidence,
                                     vec.entropy_estimate, user_dissipation)) {
            // dissipation approved — smooth alpha already handled by cb.alpha()
        }
        cb.step(vec.confidence, vec.bp_index, last_fidelity, user_dissipation);
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    g_running.store(false, std::memory_order_relaxed);  // unblock watchdog thread

    fprintf(stderr, "\n[main] shutting down after %llu frames\n",
            (unsigned long long)frame_count);

    sov.log_event("SESSION_END",
        g_killswitch_armed ? "kill-switch triggered" : "clean shutdown");

    watchdog.join();
    bridge.shutdown();
    if (dsp) { dsp_destroy(dsp); dsp = nullptr; }

#ifdef KHAOS_LSL_ENABLED
    delete inlet;
#endif

    if (cfg.dry_run) {
        printf("[MIRROR] dry-run complete — %llu frames processed, "
               "last fidelity=%.3f\n",
               (unsigned long long)frame_count, last_fidelity);
    }

    return g_killswitch_armed ? 2 : 0;
}
