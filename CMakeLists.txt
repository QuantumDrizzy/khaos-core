cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

# =============================================================================
# khaos-core — Quantum BCI Kernel
# =============================================================================
project(khaos_core
    VERSION 0.1.0
    DESCRIPTION "Real-time EEG → quantum circuit BCI kernel"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# ██████╗  Ethics Compliance Gate — READ ETHICS.md BEFORE DISABLING
# =============================================================================
option(ETHICS_COMPLIANT
    "Enforce neurorights compliance (REQUIRED — see docs/ETHICS.md)" ON)

if(NOT ETHICS_COMPLIANT)
    message(FATAL_ERROR
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║          khaos-core WILL NOT BUILD                          ║\n"
        "║          ETHICS_COMPLIANT is set to OFF                     ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  This flag enforces the neurorights principles in           ║\n"
        "║  docs/ETHICS.md:                                            ║\n"
        "║                                                              ║\n"
        "║   • Mental Privacy    — no raw EEG leaves the device        ║\n"
        "║   • Mental Integrity  — no autonomous dissipation gate      ║\n"
        "║   • Cognitive Sovereignty — audit log + hardware kill-switch║\n"
        "║                                                              ║\n"
        "║  To amend these principles:                                  ║\n"
        "║   1. Edit docs/ETHICS.md with rationale and review date.    ║\n"
        "║   2. Document any applicable regulatory review.             ║\n"
        "║   3. Re-enable ETHICS_COMPLIANT=ON.                         ║\n"
        "║                                                              ║\n"
        "║  khaos-core cannot be compiled without this flag.           ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )
endif()

# Capture git hash for audit log provenance
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE KHAOS_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
else()
    set(KHAOS_GIT_HASH "unknown")
endif()
message(STATUS "khaos build hash: ${KHAOS_GIT_HASH}")

# =============================================================================
# Build type defaults
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# CUDA configuration
# =============================================================================
find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")

# Detect GPU architecture — default to native (auto-detect from installed GPU)
# Override with: cmake -DCMAKE_CUDA_ARCHITECTURES=89  (Ada Lovelace / RTX 4xxx-5xxx)
#                cmake -DCMAKE_CUDA_ARCHITECTURES=86  (Ampere / RTX 3xxx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
    message(STATUS "CUDA architecture: native (auto-detect)")
endif()

# =============================================================================
# CUDA-Q (cudaq) detection
# =============================================================================
find_package(cudaq QUIET
    HINTS $ENV{CUDAQ_INSTALL_PREFIX} /usr/local/cudaq)

if(cudaq_FOUND)
    message(STATUS "CUDA-Q found: ${cudaq_VERSION}")
    set(KHAOS_CUDAQ ON)
else()
    message(WARNING
        "CUDA-Q not found. The quantum simulation backend will be disabled.\n"
        "Install CUDA-Q: https://developer.nvidia.com/cuda-q\n"
        "Or set CUDAQ_INSTALL_PREFIX to the install directory.")
    set(KHAOS_CUDAQ OFF)
endif()

# =============================================================================
# Open Quantum Safe — CRYSTALS-Kyber / Dilithium (post-quantum crypto)
# =============================================================================
find_package(liboqs QUIET
    HINTS $ENV{LIBOQS_PREFIX} /usr/local)

if(liboqs_FOUND OR TARGET OQS::oqs)
    message(STATUS "liboqs found — post-quantum crypto enabled")
    set(KHAOS_PQC ON)
else()
    find_library(LIBOQS_LIB oqs HINTS /usr/local/lib /usr/lib)
    find_path(LIBOQS_INCLUDE oqs/oqs.h HINTS /usr/local/include /usr/include)
    if(LIBOQS_LIB AND LIBOQS_INCLUDE)
        add_library(OQS::oqs UNKNOWN IMPORTED)
        set_target_properties(OQS::oqs PROPERTIES
            IMPORTED_LOCATION "${LIBOQS_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBOQS_INCLUDE}")
        set(KHAOS_PQC ON)
        message(STATUS "liboqs found via manual path: ${LIBOQS_LIB}")
    else()
        message(WARNING
            "liboqs not found. Neural key Kyber-1024 encryption will be disabled.\n"
            "Install: https://github.com/open-quantum-safe/liboqs")
        set(KHAOS_PQC OFF)
    endif()
endif()

# =============================================================================
# Lab Streaming Layer (LSL) — EEG data capture
# =============================================================================
find_package(LSL QUIET HINTS $ENV{LSL_ROOT})
if(LSL_FOUND OR TARGET LSL::lsl)
    message(STATUS "LSL found")
    set(KHAOS_LSL ON)
else()
    find_library(LSL_LIB lsl HINTS /usr/local/lib)
    find_path(LSL_INCLUDE lsl_cpp.h HINTS /usr/local/include)
    if(LSL_LIB AND LSL_INCLUDE)
        add_library(LSL::lsl UNKNOWN IMPORTED)
        set_target_properties(LSL::lsl PROPERTIES
            IMPORTED_LOCATION "${LSL_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LSL_INCLUDE}")
        set(KHAOS_LSL ON)
        message(STATUS "LSL found via manual path: ${LSL_LIB}")
    else()
        message(WARNING
            "liblsl not found. LSL EEG capture will be unavailable.\n"
            "Install: https://github.com/sccn/liblsl")
        set(KHAOS_LSL OFF)
    endif()
endif()

# =============================================================================
# Compiler flags — optimized for minimum latency
# =============================================================================

# Common CXX flags
set(KHAOS_CXX_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND KHAOS_CXX_FLAGS
        -O3                     # full optimization
        -march=native           # use all available CPU ISA extensions
        # -fno-exceptions removed: sovereignty_monitor.cpp uses std::runtime_error
        -fno-omit-frame-pointer # keep for profiling / perf
        -Wall -Wextra
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND KHAOS_CXX_FLAGS -DNDEBUG)
    endif()
endif()

# CUDA flags — minimize kernel launch latency
set(KHAOS_CUDA_FLAGS "")
list(APPEND KHAOS_CUDA_FLAGS
    --use_fast_math             # fused multiply-add, fast sqrt/trig
    --generate-line-info        # preserve source line info for profiling
    -O3
    --ptxas-options=-v          # verbose PTX assembler (register usage)
    --maxrregcount=64           # cap register usage to maximise occupancy
)
# Real-time: prefer low latency over high throughput
list(APPEND KHAOS_CUDA_FLAGS --Wreorder)

# =============================================================================
# Core library target
# =============================================================================
add_library(khaos_core STATIC
    # DSP pipeline — dsp_pipeline.cu is the single CUDA TU; it #includes
    # signal_processor.cu and dwt.cu so all three kernels compile together
    # and link cleanly without duplicate symbols.
    src/neuro/dsp_pipeline.cu

    # Closed-loop feedback engine — tactile PWM/FM modulation kernel
    # Included as a separate CUDA TU (does not include signal_processor.cu)
    src/neuro/feedback_engine.cu

    # Sovereignty layer
    src/security/sovereignty_monitor.cpp
    src/security/sha256.cpp

    # LSL connector — async ring-buffer fill + jitter measurement
    # Plain C++ (no nvcc): includes lsl_cpp.h when KHAOS_LSL_ENABLED
    src/neuro/lsl_connector.cpp
)

target_include_directories(khaos_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})

target_compile_options(khaos_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${KHAOS_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${KHAOS_CUDA_FLAGS}>)

# Handle empty git hash (no git repo or no commits)
if(KHAOS_GIT_HASH STREQUAL "" OR NOT KHAOS_GIT_HASH)
    set(KHAOS_GIT_HASH "0")
endif()

# ETHICS_COMPLIANT propagates to all translation units
# NOTE: STIM_ABSOLUTE_MAX_AMP and KILLSWITCH_TIMEOUT_MS are defined in
#       include/safety_constants.h — do NOT redefine here as -D macros
#       or they will collide with the constexpr declarations.
target_compile_definitions(khaos_core PUBLIC
    ETHICS_COMPLIANT=1
    KHAOS_VERSION_HASH=0x${KHAOS_GIT_HASH}
    $<$<BOOL:${KHAOS_CUDAQ}>:KHAOS_CUDAQ_ENABLED>
    $<$<BOOL:${KHAOS_PQC}>:KHAOS_PQC_ENABLED>
    $<$<BOOL:${KHAOS_LSL}>:KHAOS_LSL_ENABLED>
)

# Enable CUDA separable compilation (needed for device-side function calls
# across .cu translation units, e.g. signal_processor → dwt)
set_target_properties(khaos_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Link optional dependencies
target_link_libraries(khaos_core PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    $<$<BOOL:${KHAOS_CUDAQ}>:cudaq::cudaq>
    $<$<BOOL:${KHAOS_PQC}>:OQS::oqs>
    $<$<BOOL:${KHAOS_LSL}>:LSL::lsl>
)

# =============================================================================
# Main executable
# =============================================================================
add_executable(khaos_mirror
    src/main.cpp)

target_link_libraries(khaos_mirror PRIVATE khaos_core)
target_compile_options(khaos_mirror PRIVATE ${KHAOS_CXX_FLAGS})

# =============================================================================
# Tests  (only built when source files exist)
# =============================================================================
include(CTest)
enable_testing()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/bench/bench_latency.cu)
    add_executable(bench_latency tests/bench/bench_latency.cu)
    target_link_libraries(bench_latency PRIVATE khaos_core CUDA::cudart)
    target_compile_options(bench_latency PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${KHAOS_CUDA_FLAGS}>)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/unit/test_sovereignty.cpp)
    add_executable(test_sovereignty tests/unit/test_sovereignty.cpp)
    target_link_libraries(test_sovereignty PRIVATE khaos_core)
    add_test(NAME sovereignty COMMAND test_sovereignty)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/unit/test_iir_filter.cu)
    # Standalone TU: includes signal_processor.cu directly → no khaos_core link.
    add_executable(test_iir_filter tests/unit/test_iir_filter.cu)
    target_include_directories(test_iir_filter PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_iir_filter PRIVATE ETHICS_COMPLIANT=1)
    target_compile_options(test_iir_filter PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${KHAOS_CUDA_FLAGS}>)
    target_link_libraries(test_iir_filter PRIVATE CUDA::cudart)
    set_target_properties(test_iir_filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME iir_filter COMMAND test_iir_filter)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/unit/test_dwt.cu)
    # Standalone TU: includes signal_processor.cu + dwt.cu directly.
    add_executable(test_dwt tests/unit/test_dwt.cu)
    target_include_directories(test_dwt PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(test_dwt PRIVATE ETHICS_COMPLIANT=1)
    target_compile_options(test_dwt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${KHAOS_CUDA_FLAGS}>)
    target_link_libraries(test_dwt PRIVATE CUDA::cudart)
    set_target_properties(test_dwt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME dwt COMMAND test_dwt)
endif()

# =============================================================================
# Install
# =============================================================================
install(TARGETS khaos_core khaos_mirror
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY include/ DESTINATION include/khaos)
install(FILES docs/ETHICS.md DESTINATION share/khaos)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "  khaos-core ${PROJECT_VERSION} configuration summary")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "  ETHICS_COMPLIANT : ON  (required)")
message(STATUS "  CUDA Toolkit     : ${CUDAToolkit_VERSION}")
message(STATUS "  GPU architecture : sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA-Q backend   : ${KHAOS_CUDAQ}")
message(STATUS "  Post-quantum PQC : ${KHAOS_PQC}")
message(STATUS "  LSL EEG capture  : ${KHAOS_LSL}")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "══════════════════════════════════════════════")
message(STATUS "")
