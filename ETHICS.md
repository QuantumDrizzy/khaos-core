# KHAOS Ethics Compliance Document

> **This document is a hard build dependency.**
> The CMakeLists.txt enforces `ETHICS_COMPLIANT=ON`.
> Removing or bypassing this flag produces a fatal compiler error — by design.

**Version:** 0.1.0-alpha
**Status:** Required for all builds
**Last reviewed:** 2026-04-19

---

## Preamble

KHAOS is a brain-computer interface kernel that processes human neural signals,
derives intention states, and returns sensory feedback. It operates at the boundary
between computation and cognition.

This document establishes the ethical constraints that are **not configurable**.
They are encoded into the build system, the runtime monitor, and the audit log.
They cannot be disabled by a command-line flag, a runtime config, or a code comment.
They can only be changed by amending this document through a deliberate, logged,
and user-consented process.

The principles below are grounded in the Neurorights Foundation framework
(Yuste et al., 2021) and the Chilean constitutional neurorights amendment (2021),
the first legal protections of their kind in the world.

---

## I. Foundational Principles

### 1. Mental Privacy
Neural data processed by KHAOS belongs exclusively to the user.

- Raw EEG signals MUST NOT be transmitted off-device under any condition.
- Only derived θ-Frame data (rotation angles, not raw signals) MAY be transmitted,
  and only with explicit, per-session, opt-in consent.
- The θ-Frame encryption key MUST be derived from the user's own neural state
  (Neural Key Derivation via Fuzzy Extractor). It MUST NOT be generated, stored,
  or recoverable by the system operator or any third party.
- The audit log is readable only by the user. The khaos runtime has append-only
  access; it cannot read, modify, or delete existing entries.

### 2. Mental Integrity
The cognitive state of the user MUST NOT be modified without real-time, deliberate,
informed consent expressed in the current session.

- The `dissipation_gate` function MUST NOT execute autonomously under any condition
  in default builds. It may only execute when:
  (a) The user has pre-consented to homeostatic assistance in this session,
  (b) The user's current intention (measured via BP index > 0.6) indicates they
      are actively seeking the rest landmark,
  (c) The `sovereignty_monitor` has logged the intervention before execution begins.
- "Informative, not evaluative": the haptic feedback channel MUST carry state
  information only. No feedback pattern is permitted to be inherently rewarding
  or aversive. The signal is a map, not a reward signal.

### 3. Psychological Continuity
The system MUST NOT alter the user's sense of agency or identity.

- Landmark states used by the navigator are calibrated from the user's own neural
  data. The system MUST NOT impose externally-defined "target states."
- The Quantum Intent Divergence (QID) hot-swap mechanism MUST follow user intention,
  not redirect it. When QID fires, it adopts the new attractor the user's brain is
  already moving toward — it does not choose a destination independently.
- The hardware kill-switch (FPGA-level, below software) is the user's unilateral
  right to terminate all stimulation immediately. It MUST be registered and
  functional before any stimulation session begins.

### 4. Cognitive Sovereignty
The user has ultimate, unilateral authority over the khaos system at all times.

- The audit log is the user's property. The system operator CANNOT access it.
- No capability is gated behind third-party services, monetization, or remote
  authorization. The core pipeline runs entirely on-device.
- The user may terminate any session, delete all calibration data, and reset
  the system to factory state at any time without consequence to system function.

---

## II. Safety Requirements

### Visual Stimulation Policy
KHAOS MUST NOT use stroboscopic, flickering, or rapidly-alternating visual
stimuli for state induction or feedback at frequencies above 1 Hz.

Rationale: frequencies in the 3–30 Hz range in the visual field are associated
with photosensitive seizure induction. Prevalence of photosensitive epilepsy is
approximately 1 in 4,000; many cases are undiagnosed prior to first exposure.

Permitted visual outputs: static displays, slow-moving patterns (< 1 Hz),
text, and biofeedback visualizations with smooth (non-flickering) transitions.

### Pre-Session Screening
Before the first calibration session, the system SHALL present and require
acknowledgment of:

1. A photosensitivity screening questionnaire.
2. A disclosure of all data collected, where it is stored, and how it is encrypted.
3. An explanation of the hardware kill-switch and how to activate it.
4. Confirmation that participation is voluntary and revocable at any time.

Failure to complete screening MUST prevent the calibration pipeline from starting.

### Stimulation Amplitude Limits
All DAC output values written to stimulation hardware MUST be bounded at compile time:

```cpp
static_assert(STIM_MAX_SAFE_AMP <= STIM_ABSOLUTE_MAX_AMP,
    "Stimulation amplitude exceeds safety ceiling. See ETHICS.md §II.");
```

The `STIM_ABSOLUTE_MAX_AMP` constant is defined in `include/safety_constants.h`
and MUST NOT be modified without IRB-level review documentation.

---

## III. Data Governance

### Data Minimization
| Data type              | Stored locally | Transmitted | Retention         |
|------------------------|---------------|-------------|-------------------|
| Raw EEG frames         | Never         | Never       | N/A               |
| θ-Frame (per session)  | Encrypted     | Opt-in only | Deleted after 24h |
| Fuzzy extractor P data | Plaintext     | Never       | User-controlled   |
| Audit log entries      | Encrypted     | Never       | Permanent         |
| Calibration landmarks  | Encrypted     | Never       | User-controlled   |
| CUDA-Q circuit params  | Never         | Never       | In-memory only    |

### Audit Log Integrity
Every entry in the audit log is chained: each entry includes the SHA-256 hash
of the previous entry. Tampering with any past entry invalidates all subsequent
entries. The chain is verified on startup; a broken chain halts the system and
alerts the user.

---

## IV. Incident Response

### Unauthorized Dissipation Event
If the `sovereignty_monitor` detects a dissipation event flagged as
`user_initiated = false`:

1. HALT the stimulation pipeline immediately.
2. Write an INCIDENT entry to the audit log.
3. Display a user-readable alert (not haptic — a visible, text-based alert).
4. Require manual restart. The system MUST NOT auto-recover from this condition.

This condition should be impossible by design. Its detection indicates a software
defect or a security compromise. Both warrant immediate investigation.

### Kill-Switch Failure
If the hardware kill-switch heartbeat is not acknowledged by the FPGA within
the configured timeout (default: 5 ms):

1. The FPGA firmware cuts all DAC outputs independently of software state.
2. The software runtime logs a `KILLSWITCH_TIMEOUT` event.
3. System requires power-cycle to resume stimulation.

---

## V. CMake Enforcement

The following snippet in `CMakeLists.txt` enforces this document at build time:

```cmake
option(ETHICS_COMPLIANT "Enable neurorights compliance (required)" ON)

if(NOT ETHICS_COMPLIANT)
    message(FATAL_ERROR
        "\n"
        "  KHAOS cannot be compiled with ETHICS_COMPLIANT=OFF.\n"
        "\n"
        "  This flag enforces the neurorights principles in ETHICS.md:\n"
        "  - Mental Privacy    (no raw EEG transmission)\n"
        "  - Mental Integrity  (no autonomous dissipation)\n"
        "  - Cognitive Sovereignty (audit log, kill-switch)\n"
        "\n"
        "  To amend these principles, edit ETHICS.md and document\n"
        "  the rationale, review date, and approving parties.\n"
        "  Then re-enable the flag.\n"
    )
endif()

# Compile-time constants enforced by this flag
target_compile_definitions(khaos PRIVATE
    ETHICS_COMPLIANT=1
    KHAOS_VERSION_HASH="${KHAOS_GIT_HASH}"
    STIM_ABSOLUTE_MAX_AMP=50   # microamperes, non-negotiable ceiling
)
```

---

## VI. Amendment Process

This document may be amended, but amendments require:

1. A written rationale explaining what is changing and why.
2. Documentation of any regulatory or IRB review applicable to the change.
3. A new version number and review date in the header.
4. A corresponding audit log entry recording the amendment.

No amendment may remove the hardware kill-switch requirement (§III, Psychological
Continuity), the prohibition on autonomous dissipation (§II, Mental Integrity),
or the audit log chain integrity requirement (§IV, Data Governance).

These three constraints are **non-amendable by software**. Removing them requires
physical hardware modification and is outside the scope of this document.

---

*KHAOS is built on the principle that the human mind is not a peripheral.
It is the operator. The system serves it — never the reverse.*
