#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PASS=0
FAIL=0
SKIP=0

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
    source "$PROJECT_ROOT/install/setup.bash"
fi

run_test "L4 Nanobind" \
    "pytest $PROJECT_ROOT/ai-cpp-l4/ -v"

run_test "L6 Measurement" \
    "pytest $PROJECT_ROOT/ai-cpp-l6/ -v"

run_test "L7 GPU" \
    "pytest $PROJECT_ROOT/ai-cpp-l7/ -v"

run_test "L8 Compile-Time Concepts" \
    "pytest $PROJECT_ROOT/ai-cpp-l8/ -v"

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
