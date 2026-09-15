#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PASS=0
FAIL=0
SKIP=0

# L3's components are submodules. Uninitialised, they are empty directories
# that colcon skips without complaint, so say so rather than reporting a pass.
if [ ! -f "$PROJECT_ROOT/ai-cpp-l3/shm/CMakeLists.txt" ]; then
    echo "ERROR: ai-cpp-l3 submodules are not initialised."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

echo "========================================="
echo "  ai-cpp-course Test Runner"
echo "  All tests run inside Docker container"
echo "========================================="
echo ""

run_test() {
    local label="$1"
    local cmd="$2"
    echo "--- $label ---"
    if eval "$cmd" 2>&1; then
        echo "$label: PASSED"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        if [ $exit_code -eq 5 ]; then
            echo "$label: NO TESTS COLLECTED (skipped)"
            SKIP=$((SKIP + 1))
        else
            echo "$label: FAILED"
            FAIL=$((FAIL + 1))
        fi
    fi
    echo ""
}

# ---- Phase 1: Python-only tests (no build required) ----
run_test "L5 Python Optimization" \
    "pytest $PROJECT_ROOT/ai-cpp-l5/ -v"

# ---- Phase 2: Tests requiring compiled modules ----
# Source colcon install if available
if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    # install/setup.bash references COLCON_TRACE without a default, which
    # aborts under our own `set -u` (unbound variable).
    set +u
    source "$PROJECT_ROOT/install/setup.bash"
    set -u
fi

run_test "L3 Shared Memory" \
    "PYTHONPATH=$PROJECT_ROOT/ai-cpp-l3:\${PYTHONPATH:-} pytest $PROJECT_ROOT/ai-cpp-l3/test_shm.py $PROJECT_ROOT/ai-cpp-l3/test_integration_shm.py -v"

run_test "L4 Nanobind" \
    "pytest $PROJECT_ROOT/ai-cpp-l4/ -v"

run_test "L6 Measurement" \
    "pytest $PROJECT_ROOT/ai-cpp-l6/ -v"

run_test "L7 GPU" \
    "pytest $PROJECT_ROOT/ai-cpp-l7/ -v"

run_test "L8 Compile-Time Concepts" \
    "pytest $PROJECT_ROOT/ai-cpp-l8/ -v"

run_test "L18 Golden Oracles and Sanitizers" \
    "pytest $PROJECT_ROOT/ai-cpp-l18/ -v"

# L19's default (static) build is SUPPOSED to fail this test -- that failure
# is the lesson. See ai-cpp-l19/README.md.
echo "--- L19 Linking Semantics (static build, expected FAIL) ---"
if pytest "$PROJECT_ROOT/ai-cpp-l19/" -v 2>&1; then
    echo "L19 Linking Semantics: FAILED (static build should not pass this -- see README.md)"
    FAIL=$((FAIL + 1))
else
    echo "L19 Linking Semantics: PASSED (failed as expected for a static build)"
    PASS=$((PASS + 1))
fi
echo ""

# ---- Summary ----
echo "========================================="
echo "  Summary"
echo "========================================="
echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi

exit 0
