/**
 * main.cpp — KHAOS / Quantum Mirror  (closed-loop integration)
 *
 * Full pipeline:
 *
 *   LSLConnector  ──►  SignalProcessor (CUDA IIR + à-trous DWT)
 *                             │
 *                     NeuralPhaseVector (θ-Frame, 10 Hz)
 *                             │
 *                      Python bridge  (CUDA-Q + graphene model)
 *                             │
 *                   ent_alpha[N_HUB_CHANNELS]  (quantum proximity)
 *                             │
 *                     FeedbackEngine (CUDA PWM + FM modulation)
 *                             │
 *                    TactileFeedbackOutput  →  FPGA register bank
 *                             │
 *                   SovereigntyMonitor  (audit log, kill-switch)
 *
 * Build:  cmake -B build -DETHICS_COMPLIANT=ON && cmake --build build
 * Run:    ./khaos_mirror [--log <path>] [--stream <lsl-name>] [--bridge <py-script>]
 *         ./khaos_mirror --dry-run   (synthetic EEG, no hardware, 5-second smoke test)
 *
 * --dry-run notes:
 *   • LSLConnector started in synthetic mode (10 Hz μ + 25 Hz β + noise).
 *   • GPU DSP pipeline is bypassed; NeuralPhaseVector is synthesised from
 *     the frame index so the bridge + feedback paths still exercise normally.
 *   • FeedbackEngine is fully active (CUDA modulation kernel runs as usual).
 *   • Exits after DRY_RUN_SECONDS.
 */

#ifndef ETHICS_COMPLIANT
#  error "main.cpp requires ETHICS_COMPLIANT. See docs/ETHICS.md."
#endif

// ── khaos headers ────────────────────────────────────────────────────────────
#include "../include/safety_constants.h"
#include "../include/sha256.h"
#include "../include/khaos_bridge.h"      // NeuralPhaseVector, MAX_QUBITS, N_CHANNELS
#include "../include/dsp_pipeline.h"      // dsp_create_and_init / dsp_process_frame / …
#include "../include/lsl_connector.h"     // EEGFrameSlot, LSLHandle, lsl_*
#include "../include/feedback_engine.h"   // TactileFeedbackOutput, FeedbackHandle, feedback_*
#include "../include/fpga_driver.h"       // FPGAHandle, FPGAStatus, fpga_*

// ── stdlib ───────────────────────────────────────────────────────────────────
#include <array>
#include <atomic>
#include <chrono>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
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

// ── POSIX (subprocess pipes, signal handling, real-time threads) ─────────────
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// =============================================================================
// Pipeline constants
// =============================================================================

static constexpr int    SAMPLE_RATE_HZ      = 1000;
static constexpr int    N_QUBITS            = N_HUB_CHANNELS;  // 12
static constexpr int    BRIDGE_DOWNSAMPLE   = 100;  // send θ-Frame every 100 ms
static constexpr int    RING_DRAIN_LIMIT    = 8;    // max frames popped per iter
static constexpr int    DRY_RUN_SECONDS     = 5;    // duration of --dry-run

// Feedback max duty (mirrors FEEDBACK_MAX_DUTY in feedback_engine.cu)
static constexpr float  FEEDBACK_MAX_DUTY_F = 32767.0f;

// =============================================================================
// SovereigntyMonitor (stub — real implementation in sovereignty_monitor.cpp)
// =============================================================================

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

    void register_killswitch(std::function<void()> fn) {
        killswitch_ = std::move(fn);
    }

    [[nodiscard]]
    bool request_dissipation(float /*alpha*/, float /*conf*/,
                              float /*ent*/, bool user_initiated) noexcept {
        return user_initiated;
    }

    void log_frame(const NeuralPhaseVector& /*vec*/) noexcept {}

    void log_event(const char* tag, const char* detail) {
        fprintf(stderr, "[sovereignty] %-24s  %s\n", tag, detail);
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
// FPGA shared frame (EEG thread → FPGA driver thread)
// =============================================================================
//
// The EEG loop publishes each feedback frame here and signals via eventfd.
// The FPGA driver thread consumes one frame per eventfd read.
//
// Memory ordering: the eventfd write (release) / read (acquire) pair provides
// the required happens-before relationship between the frame copy and the
// fpga_write_frame() call.  No mutex needed for this single-producer,
// single-consumer pattern.

struct FPGASharedFrame {
    std::atomic<uint32_t>    seq{0};      // odd = write in progress (producer fence)
    TactileFeedbackOutput    frame{};     // most-recent feedback output
};

static FPGASharedFrame g_fpga_frame;

// =============================================================================
// Thread real-time helpers
// =============================================================================

/**
 * Pin the calling thread to @p cpu_core and optionally elevate to SCHED_FIFO.
 *
 * @param cpu_core   Core to pin to (≥0), or -1 to skip affinity.
 * @param sched_prio SCHED_FIFO priority [1,99], or 0/negative to skip.
 * @param label      Short label for log messages (e.g. "main", "fpga").
 */
static void set_thread_realtime(int cpu_core, int sched_prio, const char* label)
{
    pthread_t self = pthread_self();

    if (cpu_core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET((size_t)cpu_core, &cpuset);
        int rc = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
        if (rc != 0)
            fprintf(stderr, "[rt] WARNING: %s: pthread_setaffinity_np(core=%d) "
                            "failed (rc=%d)\n", label, cpu_core, rc);
        else
            fprintf(stderr, "[rt] %s thread pinned to Core %d\n", label, cpu_core);
    }

    if (sched_prio >= 1) {
        struct sched_param sp{};
        sp.sched_priority = sched_prio;
        int rc = pthread_setschedparam(self, SCHED_FIFO, &sp);
        if (rc != 0)
            fprintf(stderr, "[rt] WARNING: %s: SCHED_FIFO prio=%d failed "
                            "(rc=%d) — running SCHED_OTHER\n",
                    label, sched_prio, rc);
        else
            fprintf(stderr, "[rt] %s thread SCHED_FIFO prio=%d\n",
                    label, sched_prio);
    }
}

// =============================================================================
// FPGA driver thread
// =============================================================================

struct FPGADriverContext {
    FPGAHandle* fpga;
    int         event_fd;   // eventfd signalled by EEG loop after each frame
    int         epoll_fd;   // epoll set watching event_fd
};

/**
 * FPGA driver thread function (Core 2, SCHED_FIFO 80).
 *
 * Waits for the EEG loop to signal event_fd, then:
 *   1. Reads (and drains) the eventfd counter.
 *   2. Copies the shared frame under an atomic sequence check.
 *   3. Calls fpga_write_frame() — fire-and-forget register write + COMMIT.
 *   4. Every STATUS_CHECK_INTERVAL cycles: calls fpga_read_status() and
 *      arms the kill-switch if FAULT or SOV_FAIL is set.
 *
 * Exits when g_running goes false.
 */
static void fpga_driver_thread_fn(FPGADriverContext ctx) noexcept
{
    // Pin this thread to Core 2, SCHED_FIFO prio 80
    set_thread_realtime(2, 80, "fpga");
    fprintf(stderr, "[fpga] driver thread started\n");

    static constexpr int STATUS_CHECK_INTERVAL = 10;  // check STATUS every N commits
    int cycle = 0;

    struct epoll_event ev{};

    while (g_running.load(std::memory_order_relaxed)) {

        // ── Wait for EEG loop to signal a new frame ───────────────────────────
        int n = epoll_wait(ctx.epoll_fd, &ev, 1, /*timeout_ms=*/200);
        if (n <= 0) {
            // Timeout (200 ms) or EINTR — loop back and check g_running
            continue;
        }

        // ── Drain the eventfd counter (may have accumulated multiple signals) ─
        uint64_t efd_val = 0;
        ssize_t _rd = read(ctx.event_fd, &efd_val, sizeof(efd_val));
        (void)_rd;

        // ── Snapshot the shared frame ─────────────────────────────────────────
        // The eventfd read (acquire) synchronises with the eventfd write
        // (release) in the EEG loop, so frame is guaranteed up-to-date here.
        TactileFeedbackOutput local_frame;
        memcpy(&local_frame, &g_fpga_frame.frame, sizeof(local_frame));

        // ── Write frame to FPGA register bank ────────────────────────────────
        fpga_write_frame(ctx.fpga, &local_frame);
        ++cycle;

        // ── Deferred STATUS check (avoids blocking the commit path) ──────────
        if (cycle % STATUS_CHECK_INTERVAL == 0) {
            FPGAStatus s = fpga_read_status(ctx.fpga);
            if (s.fault) {
                fprintf(stderr,
                    "[fpga] FATAL: STATUS fault detected "
                    "(raw=0x%04X, frame_ctr=%u) — arming kill-switch\n",
                    s.raw, s.frame_ctr);
                SovereigntyMonitor::instance().log_event(
                    "FPGA_FAULT",
                    (s.raw & FPGA_STATUS_SOV_FAIL) ?
                        "sovereignty token mismatch" : "hardware fault");
                g_killswitch_armed.store(true, std::memory_order_relaxed);
                g_running.store(false, std::memory_order_relaxed);
            }
            if (s.overrun) {
                fprintf(stderr, "[fpga] OVERRUN detected (frame_ctr=%u)\n",
                        s.frame_ctr);
            }
        }
    }

    fprintf(stderr, "[fpga] driver thread exiting after %d commits\n", cycle);
}

// =============================================================================
// Circuit breaker
// =============================================================================

enum class CircuitState { NOMINAL, DEGRADED, PANIC, RECOVERING };

struct CircuitBreaker {
    CircuitState state = CircuitState::NOMINAL;

    int low_conf_ticks  = 0;
    int very_low_ticks  = 0;
    int panic_ticks     = 0;

    static constexpr int DEGRADED_THRESH_MS  = 200;
    static constexpr int PANIC_THRESH_MS     = 100;
    static constexpr int PANIC_TIMEOUT_MS    = 5000;

    void step(float confidence, float bp_index,
              float fidelity, bool user_dissipation) noexcept {
        using CS = CircuitState;
        switch (state) {
            case CS::NOMINAL:
                if (confidence < 0.5f) ++low_conf_ticks; else low_conf_ticks = 0;
                if (low_conf_ticks >= DEGRADED_THRESH_MS) transition(CS::DEGRADED);
                break;
            case CS::DEGRADED:
                if (confidence < 0.3f) ++very_low_ticks; else very_low_ticks = 0;
                if (confidence >= 0.5f) { low_conf_ticks = 0; transition(CS::NOMINAL); break; }
                if (very_low_ticks >= PANIC_THRESH_MS || bp_index > 0.8f)
                    transition(CS::PANIC);
                break;
            case CS::PANIC:
                ++panic_ticks;
                if (user_dissipation || panic_ticks >= PANIC_TIMEOUT_MS)
                    transition(CS::RECOVERING);
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
    int    stdin_fd  = -1;
    int    stdout_fd = -1;

    float fidelity              = 0.0f;
    char  landmark[32]          = "unknown";
    bool  qid                   = false;
    float ent_alpha[MAX_QUBITS] = {};   // quantum proximity per hub channel

    bool spawn(const char* script) {
        int to_child[2], from_child[2];
        if (pipe(to_child) || pipe(from_child)) { perror("pipe"); return false; }

        child_pid = fork();
        if (child_pid < 0) { perror("fork"); return false; }

        if (child_pid == 0) {
            setsid();
            dup2(to_child[0],   STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);  close(to_child[1]);
            close(from_child[0]); close(from_child[1]);
            execlp("python3", "python3", "-u", script, nullptr);
            perror("execlp");
            _exit(1);
        }

        close(to_child[0]);
        close(from_child[1]);
        stdin_fd  = to_child[1];
        stdout_fd = from_child[0];

        int flags = fcntl(stdout_fd, F_GETFL, 0);
        fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);

        fprintf(stderr, "[bridge] spawned %s (pid=%d)\n", script, child_pid);
        return true;
    }

    // Send θ-Frame to the bridge (called every BRIDGE_DOWNSAMPLE frames = 10 Hz)
    void send(const NeuralPhaseVector& vec) noexcept {
        if (stdin_fd < 0) return;
        char buf[512];
        int n = snprintf(buf, sizeof(buf),
            "{\"theta\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
             "\"bp_index\":%.4f,\"timestamp_ns\":%lu}\n",
            vec.theta[0],  vec.theta[1],  vec.theta[2],  vec.theta[3],
            vec.theta[4],  vec.theta[5],  vec.theta[6],  vec.theta[7],
            vec.theta[8],  vec.theta[9],  vec.theta[10], vec.theta[11],
            vec.bp_index, (unsigned long)vec.timestamp_ns);
        if (n > 0) {
            // Bridge may die at any time; write failure is detected on the
            // next bridge.poll() (returns false) and cleaned up by shutdown().
            ssize_t _wr = write(stdin_fd, buf, (size_t)n);
            (void)_wr;
        }
    }

    // Block until bridge emits {"ready":true} or timeout_s elapses
    bool wait_ready(int timeout_s) noexcept {
        if (stdout_fd < 0) return false;

        int flags = fcntl(stdout_fd, F_GETFL, 0);
        fcntl(stdout_fd, F_SETFL, flags & ~O_NONBLOCK);

        using Clock = std::chrono::steady_clock;
        auto t_end  = Clock::now() + std::chrono::seconds(timeout_s);
        char buf[256] = {};
        int  pos = 0;

        while (Clock::now() < t_end) {
            auto remaining = t_end - Clock::now();
            long rem_us = (long)std::chrono::duration_cast<
                std::chrono::microseconds>(remaining).count();
            if (rem_us <= 0) break;

            fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
            struct timeval tv{ rem_us / 1000000L, rem_us % 1000000L };
            if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) <= 0) break;

            char ch;
            if (read(stdout_fd, &ch, 1) <= 0) break;

            if (ch == '\n') {
                buf[pos] = '\0';
                if (strstr(buf, "\"ready\"")) {
                    fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);
                    return true;
                }
                pos = 0;
            } else {
                if (pos < (int)sizeof(buf) - 1) buf[pos++] = ch;
            }
        }
        fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);
        return false;
    }

    // Non-blocking poll for a response line
    bool poll() noexcept {
        if (stdout_fd < 0) return false;
        fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd, &fds);
        struct timeval tv{0, 0};
        if (select(stdout_fd + 1, &fds, nullptr, nullptr, &tv) <= 0)
            return false;
        char line[1024];
        ssize_t n = read(stdout_fd, line, sizeof(line) - 1);
        if (n <= 0) return false;
        line[n] = '\0';
        parse_response(line);
        return true;
    }

    void shutdown() noexcept {
        if (stdin_fd  >= 0) { close(stdin_fd);  stdin_fd  = -1; }
        if (child_pid > 0) {
            for (int i = 0; i < 50; ++i) {
                int status = 0;
                if (waitpid(child_pid, &status, WNOHANG) != 0) { child_pid = -1; break; }
                usleep(10000);
            }
            if (child_pid > 0) { kill(child_pid, SIGKILL); waitpid(child_pid, nullptr, 0); child_pid = -1; }
        }
        if (stdout_fd >= 0) { close(stdout_fd); stdout_fd = -1; }
    }

private:
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

        const char* lm = strstr(line, "\"landmark\":");
        if (lm) {
            const char* q = strchr(lm + 11, '"');
            if (q) {
                ++q;
                const char* end = strchr(q, '"');
                if (end) {
                    size_t len = std::min((size_t)(end - q), sizeof(landmark) - 1);
                    memcpy(landmark, q, len);
                    landmark[len] = '\0';
                }
            }
        }

        const char* ea = strstr(line, "\"ent_alpha\":");
        if (ea) {
            ea = strchr(ea + 12, '[');
            if (ea) {
                ++ea;
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
// Sovereignty: sign one closed-loop frame to the audit log
// =============================================================================

/**
 * Log a structured closed-loop frame event.
 *
 * Called every BRIDGE_DOWNSAMPLE samples (~10 Hz) when the bridge responds.
 * Encodes: frame counter, fidelity, active landmark, circuit state,
 * neural confidence / BP index / entropy, mean feedback PWM & FM, α blend.
 *
 * The SovereigntyMonitor implementation (sovereignty_monitor.cpp) SHA-256-
 * chains these events to the audit log for post-session forensic review.
 */
static void sovereignty_sign_closed_loop(
    SovereigntyMonitor&              sov,
    uint64_t                         frame_count,
    const PythonBridge&              bridge,
    const NeuralPhaseVector&         vec,
    const CircuitBreaker&            cb,
    const TactileFeedbackOutput&     out)
{
    float pwm_mean = 0.0f, fm_mean = 0.0f;
    for (int c = 0; c < N_HUB_CHANNELS; ++c) {
        pwm_mean += (float)out.pwm_duty[c] / FEEDBACK_MAX_DUTY_F * 100.0f;
        fm_mean  += out.fm_freq_hz[c];
    }
    pwm_mean /= N_HUB_CHANNELS;
    fm_mean  /= N_HUB_CHANNELS;

    char buf[320];
    snprintf(buf, sizeof(buf),
        "frame=%llu fidelity=%.3f landmark=%s circuit=%s "
        "conf=%.3f bp=%.3f entropy=%.3f "
        "pwm_mean=%.1f%% fm_mean=%.1fHz alpha=%.2f cycle=%u",
        (unsigned long long)frame_count,
        bridge.fidelity, bridge.landmark, cb.state_name(),
        vec.confidence, vec.bp_index, vec.entropy_estimate,
        pwm_mean, fm_mean, out.global_scale,
        out.bridge_cycle);

    sov.log_event("CLOSED_LOOP_FRAME", buf);
}

// =============================================================================
// Console status line (10 Hz)
// =============================================================================

static void print_closed_loop_status(
    const PythonBridge&          bridge,
    const NeuralPhaseVector&     vec,
    const CircuitBreaker&        cb,
    const TactileFeedbackOutput& out)
{
    float pwm_mean = 0.0f, fm_mean = 0.0f;
    for (int c = 0; c < N_HUB_CHANNELS; ++c) {
        pwm_mean += (float)out.pwm_duty[c] / FEEDBACK_MAX_DUTY_F * 100.0f;
        fm_mean  += out.fm_freq_hz[c];
    }
    pwm_mean /= N_HUB_CHANNELS;
    fm_mean  /= N_HUB_CHANNELS;

    printf("[MIRROR] %-10s  fid=%.3f  lm=%-12s  "
           "conf=%.2f  bp=%.2f  "
           "pwm=%5.1f%%  fm=%5.1fHz  "
           "α=%.2f  qid=%d\n",
           cb.state_name(), bridge.fidelity, bridge.landmark,
           vec.confidence, vec.bp_index,
           pwm_mean, fm_mean,
           out.global_scale, (int)bridge.qid);
    fflush(stdout);
}

// =============================================================================
// Watchdog thread
// =============================================================================

static void watchdog_thread_fn() noexcept {
    using namespace std::chrono_literals;
    while (g_running.load(std::memory_order_relaxed)) {
        SovereigntyMonitor::instance().check_heartbeat();
        std::this_thread::sleep_for(1ms);
    }
}

// =============================================================================
// Bridge path resolution
// =============================================================================
//
// khaos_mirror lives in <project>/build/.  The bridge script is at
// <project>/src/quantum/mirror_bridge.py.  We resolve the project root at
// runtime via /proc/self/exe so the binary can be run from any CWD.

static std::string resolve_default_bridge_path()
{
    // Read the absolute path of the running executable via procfs (Linux / WSL2).
    char exe[4096] = {};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = '\0';
        std::string p(exe);
        // Strip /khaos_mirror  →  .../build
        auto s1 = p.rfind('/');
        if (s1 != std::string::npos) {
            p = p.substr(0, s1);
            // Strip /build  →  <project-root>
            auto s2 = p.rfind('/');
            if (s2 != std::string::npos) {
                return p.substr(0, s2) + "/src/quantum/mirror_bridge.py";
            }
        }
    }
    // Fallback: run from project root (e.g. during CI)
    return "src/quantum/mirror_bridge.py";
}

// =============================================================================
// CLI argument parser
// =============================================================================

struct Config {
    std::string log_path      = "data/audit.log";
    std::string lsl_stream    = "EEG";
    std::string bridge_script = "";   // resolved in main() via /proc/self/exe
    bool        dry_run       = false;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if      (a == "--dry-run")               { cfg.dry_run = true; }
        else if (a == "--log"    && i+1 < argc)  { cfg.log_path      = argv[++i]; }
        else if (a == "--stream" && i+1 < argc)  { cfg.lsl_stream    = argv[++i]; }
        else if (a == "--bridge" && i+1 < argc)  { cfg.bridge_script = argv[++i]; }
        else {
            fprintf(stderr, "usage: khaos_mirror [--dry-run] [--log <path>] "
                            "[--stream <name>] [--bridge <script>]\n");
        }
    }
    return cfg;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    // Resolve bridge path now so it's absolute and CWD-independent
    if (cfg.bridge_script.empty())
        cfg.bridge_script = resolve_default_bridge_path();

    // ── Signal handlers ───────────────────────────────────────────────────────
    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);   // bridge death handled via write() return value

    // ── Sovereignty monitor ───────────────────────────────────────────────────
    auto& sov = SovereigntyMonitor::instance();
    sov.init(cfg.log_path);
    sov.register_killswitch([]() noexcept {
        g_killswitch_armed.store(true, std::memory_order_relaxed);
        g_running.store(false, std::memory_order_relaxed);
        fprintf(stderr, "\n[KILLSWITCH] Hardware kill-switch triggered.\n");
    });

    fprintf(stderr,
        "╔═══════════════════════════════════════════════╗\n"
        "║   KHAOS / Quantum Mirror v0.1            ║\n"
        "║   ETHICS_COMPLIANT build  (closed-loop)       ║\n"
        "╚═══════════════════════════════════════════════╝\n");
    if (cfg.dry_run)
        fprintf(stderr, "[mode] DRY RUN — synthetic LSL, no GPU DSP, %ds\n",
                DRY_RUN_SECONDS);

    // ── GPU DSP pipeline (IIR + à-trous DWT) ─────────────────────────────────
    DSPPipeline* dsp = nullptr;
    if (!cfg.dry_run) {
        dsp = dsp_create_and_init();
        fprintf(stderr, "[dsp] IIR + DWT pipeline initialised\n");
    } else {
        fprintf(stderr, "[dsp] skipped in dry-run mode\n");
    }

    // ── Tactile feedback engine ───────────────────────────────────────────────
    // Active in all modes: exercises the GPU modulation kernel even in dry-run.
    FeedbackHandle* fb = feedback_create_and_init();
    fprintf(stderr, "[feedback] PWM+FM engine initialised (N_HUB_CHANNELS=%d)\n",
            N_HUB_CHANNELS);

    // ── FPGA PCIe driver ──────────────────────────────────────────────────────
    // Stub mode in dry-run (no /dev/uio0 access); real UIO in production.
    FPGAHandle* fpga = fpga_open(cfg.dry_run ? nullptr : "/dev/uio0");
    fprintf(stderr, "[fpga] register bank open (stub=%s)\n",
            cfg.dry_run ? "yes" : "no");

    // ── Python bridge ─────────────────────────────────────────────────────────
    PythonBridge bridge;
    if (!bridge.spawn(cfg.bridge_script.c_str())) {
        fprintf(stderr, "[bridge] WARNING: could not spawn %s — "
                        "quantum feedback disabled\n", cfg.bridge_script.c_str());
    } else {
        constexpr int BRIDGE_WARMUP_TIMEOUT_S = 60;
        fprintf(stderr, "[bridge] waiting for initialisation "
                        "(CUDA-Q JIT may take up to %ds)…\n",
                BRIDGE_WARMUP_TIMEOUT_S);
        if (bridge.wait_ready(BRIDGE_WARMUP_TIMEOUT_S))
            fprintf(stderr, "[bridge] ready\n");
        else
            fprintf(stderr, "[bridge] WARNING: warmup timed out — "
                            "continuing without quantum feedback\n");
    }

    // ── LSL connector ─────────────────────────────────────────────────────────
    // Started AFTER bridge warmup so the ring buffer does not accumulate
    // frames during the (potentially long) CUDA-Q JIT delay.  The initial
    // jitter measurement is only meaningful from steady-state acquisition.
    LSLHandle* lsl = lsl_create(cfg.lsl_stream.c_str(), "EEG");

    // Pin LSL pull thread to Core 1 at SCHED_FIFO prio 90 (highest priority
    // acquisition thread — must not be preempted between samples at 1000 Hz).
    lsl_set_realtime(lsl, 1, 90);

    lsl_start(lsl, cfg.dry_run ? 1 : 0);
    fprintf(stderr, "[lsl] source: %s\n",
            lsl_is_synthetic(lsl) ? "synthetic generator" : "real LSL stream");

    // ── Main thread affinity (Core 0 — EEG processing loop) ──────────────────
    // Affinity only; SCHED_FIFO not applied to main so it doesn't block the
    // watchdog or FPGA threads at prio 99 / 80.
    set_thread_realtime(0, -1, "main");

    // ── eventfd + epoll for EEG→FPGA signalling ───────────────────────────────
    int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (efd < 0) { perror("[fpga] eventfd"); abort(); }

    int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) { perror("[fpga] epoll_create1"); abort(); }

    {
        struct epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = efd;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, efd, &ev) < 0) {
            perror("[fpga] epoll_ctl");
            abort();
        }
    }

    // ── FPGA driver thread (Core 2, SCHED_FIFO 80) ───────────────────────────
    FPGADriverContext fpga_ctx{ fpga, efd, epfd };
    std::thread fpga_thread(fpga_driver_thread_fn, fpga_ctx);

    // ── Watchdog thread ───────────────────────────────────────────────────────
    std::thread watchdog(watchdog_thread_fn);

    // ── State ─────────────────────────────────────────────────────────────────
    CircuitBreaker     cb;
    NeuralPhaseVector  vec        = {};
    EEGFrameSlot       slot       = {};
    uint64_t           frame_count   = 0;
    float              last_fidelity = 0.0f;

    using Clock = std::chrono::steady_clock;
    auto t_start   = Clock::now();
    auto t_dry_end = t_start + std::chrono::seconds(DRY_RUN_SECONDS);

    fprintf(stderr, "[main] entering closed-loop (Ctrl-C to stop)\n");

    // ── Closed-loop EEG loop ──────────────────────────────────────────────────
    while (g_running.load(std::memory_order_relaxed)) {

        // Dry-run exit condition
        if (cfg.dry_run && Clock::now() >= t_dry_end) break;

        // ── Drain ring buffer ─────────────────────────────────────────────────
        // Typically 0-1 frames available per iteration at 1000 Hz.
        // Cap at RING_DRAIN_LIMIT to prevent GPU stream runaway.
        int drained = 0;
        while (lsl_try_pop(lsl, &slot) && drained < RING_DRAIN_LIMIT) {
            ++drained;
            ++frame_count;

            // ── DSP: IIR biquad → à-trous DWT → NeuralPhaseVector ────────────
            if (dsp) {
                dsp_process_frame(dsp, slot.samples,
                                  slot.timestamp_ns, cb.alpha());

                // Sync θ-Frame to host every BRIDGE_DOWNSAMPLE frames (10 Hz).
                // The GPU has already processed this frame; sync is fast (~5 µs).
                if (frame_count % BRIDGE_DOWNSAMPLE == 0) {
                    dsp_request_theta_async(dsp);
                    const NeuralPhaseVector* gpu_vec = dsp_sync_theta(dsp);
                    vec = *gpu_vec;
                    sov.log_frame(vec);
                }
            } else {
                // Dry-run: synthesise a plausible θ-Frame from frame index
                double t = (double)slot.frame_index / SAMPLE_RATE_HZ;
                for (int q = 0; q < N_QUBITS; ++q)
                    vec.theta[q] = (float)(M_PI * (0.5 + 0.3 * std::sin(
                        2.0 * M_PI * 10.0 * t + q * 0.52)));
                vec.confidence       = 0.70f + 0.15f * (float)std::sin(2.0 * M_PI * 0.1  * t);
                vec.bp_index         = 0.20f + 0.10f * (float)std::sin(2.0 * M_PI * 0.05 * t);
                vec.entropy_estimate = 0.60f + 0.10f * (float)std::sin(2.0 * M_PI * 0.07 * t);
                vec.alpha_blend      = 1.0f;
                vec.timestamp_ns     = slot.timestamp_ns;
            }

            // ── Bridge exchange (10 Hz) ───────────────────────────────────────
            if (frame_count % BRIDGE_DOWNSAMPLE == 0) {
                bridge.send(vec);
            }

            // ── Poll bridge response (non-blocking) ───────────────────────────
            if (bridge.poll()) {
                last_fidelity = bridge.fidelity;

                // ── FeedbackEngine ────────────────────────────────────────────
                // Pass ent_alpha[] as per-channel proximity.
                // When landmark == "flow" and fidelity is high, ent_alpha values
                // rise → PWM duty and FM frequency increase proportionally.
                // global_scale = 0 when circuit breaker is in PANIC → kernel
                // zeroes all outputs, protecting the subject.
                feedback_process(fb,
                    bridge.ent_alpha,
                    cb.alpha(),
                    slot.timestamp_ns);
                const TactileFeedbackOutput* out = feedback_sync_output(fb);

                // ── FPGA: publish frame and wake driver thread ─────────────────
                // Copy frame into the shared buffer; the eventfd write acts as a
                // release fence so the FPGA thread sees the fully committed copy.
                memcpy(&g_fpga_frame.frame, out, sizeof(*out));
                {
                    uint64_t one = 1;
                    ssize_t _wr = write(efd, &one, sizeof(one));
                    (void)_wr;
                }

                // ── Sovereignty: sign closed-loop frame ───────────────────────
                sovereignty_sign_closed_loop(
                    sov, frame_count, bridge, vec, cb, *out);

                // ── Console status (10 Hz) ────────────────────────────────────
                print_closed_loop_status(bridge, vec, cb, *out);
            }

            // ── Circuit breaker step (1 sample = 1 ms) ────────────────────────
            bool user_dissipation = false;  // TODO: wire to UI button / BP thresh
            (void)sov.request_dissipation(vec.alpha_blend, vec.confidence,
                                           vec.entropy_estimate, user_dissipation);
            cb.step(vec.confidence, vec.bp_index, last_fidelity, user_dissipation);
        }

        // ── Yield when ring is empty (avoid busy-spin) ────────────────────────
        if (drained == 0) {
            struct timespec ts{ 0, 100000L };  // 100 µs
            nanosleep(&ts, nullptr);
        }
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    g_running.store(false, std::memory_order_relaxed);

    fprintf(stderr, "\n[main] shutdown after %llu frames  last_fidelity=%.3f\n",
            (unsigned long long)frame_count, last_fidelity);

    sov.log_event("SESSION_END",
        g_killswitch_armed ? "kill-switch triggered" : "clean shutdown");

    watchdog.join();

    // ── FPGA thread shutdown ──────────────────────────────────────────────────
    // g_running is already false; wake the epoll_wait so it exits promptly.
    {
        uint64_t one = 1;
        ssize_t _wr = write(efd, &one, sizeof(one));
        (void)_wr;
    }
    fpga_thread.join();
    close(epfd);
    close(efd);
    fpga_close(fpga);
    fpga = nullptr;

    bridge.shutdown();

    lsl_stop(lsl);
    lsl_print_stats(lsl);
    lsl_destroy(lsl);

    if (fb)  { feedback_destroy(fb);  fb  = nullptr; }
    if (dsp) { dsp_destroy(dsp);      dsp = nullptr; }

    if (cfg.dry_run) {
        printf("[MIRROR] dry-run complete — %llu frames  last_fidelity=%.3f\n",
               (unsigned long long)frame_count, last_fidelity);
    }

    return g_killswitch_armed ? 2 : 0;
}
