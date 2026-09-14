#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
rm -f /tmp/xlang_test_done

# Start C++ writer in background
"${BUILD_DIR}/cross_language_writer" &
CPP_PID=$!
sleep 0.2

# Run Python verifier
PYTHONPATH="${BUILD_DIR}" python3 "${SCRIPT_DIR}/cross_language_verify.py"
RESULT=$?

# Wait for C++ process to finish (it reads back Python-written data)
wait $CPP_PID 2>/dev/null
CPP_RESULT=$?

# Cleanup
rm -f /tmp/xlang_test_done /dev/shm/xlang_*

if [ $RESULT -ne 0 ]; then
    echo "FAIL: Python verifier failed"
    exit $RESULT
fi

if [ $CPP_RESULT -ne 0 ]; then
    echo "FAIL: C++ read-back failed"
    exit $CPP_RESULT
fi

echo "Cross-language interop test PASSED"
exit 0
