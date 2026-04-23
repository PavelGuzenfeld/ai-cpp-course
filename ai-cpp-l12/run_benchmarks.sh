#!/usr/bin/env bash
# Compile polynomial_flags.cpp three ways and compare wall time.
# Run this from the ai-cpp-l12 directory.
set -euo pipefail

CXX=${CXX:-g++}
SRC=polynomial_flags.cpp

echo "Compiler: $($CXX --version | head -1)"
echo

compile_and_run() {
  local label="$1"; shift
  local flags="$*"
  local bin
  bin=$(mktemp -u /tmp/poly_XXXXXX)
  $CXX $flags -std=c++23 "$SRC" -o "$bin"
  printf "%-40s" "$label"
  "$bin"
  rm -f "$bin"
}

compile_and_run "-O2 (baseline)"                          -O2
compile_and_run "-O3 -march=x86-64-v3"                    -O3 -march=x86-64-v3
compile_and_run "-O3 -ffast-math -march=x86-64-v3"        -O3 -ffast-math -march=x86-64-v3
compile_and_run "-O3 -ffast-math -march=native"           -O3 -ffast-math -march=native

echo
echo "Expected on Haswell+ with GCC 14:"
echo "  - Middle build can be SLOWER than -O2 (scheduling regression)"
echo "  - Third build ~7x faster than -O2 (AVX2 reduction unlocked)"
echo "  - -march=native varies; sometimes unstable due to thermal throttle"
