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

# -march names an ISA, so the baseline differs per architecture and naming
# the wrong one is a compile error, not a no-op. See README A.2.
case "$(uname -m)" in
  x86_64|amd64)  ARCH_FLAG=-march=x86-64-v3 ;;
  aarch64|arm64) ARCH_FLAG=-march=armv8.2-a+simd ;;
  *)             ARCH_FLAG=""
                 echo "No -march baseline known for $(uname -m); running without one." >&2
                 echo >&2 ;;
esac

compile_and_run "-O2 (baseline)"                          -O2
compile_and_run "-O3 $ARCH_FLAG"                          -O3 $ARCH_FLAG
compile_and_run "-O3 -ffast-math $ARCH_FLAG"              -O3 -ffast-math $ARCH_FLAG
compile_and_run "-O3 -ffast-math -march=native"           -O3 -ffast-math -march=native

echo
echo "Expected on Haswell+ with GCC 14:"
echo "  - Middle build can be SLOWER than -O2 (scheduling regression)"
echo "  - Third build ~7x faster than -O2 (AVX2 reduction unlocked)"
echo "  - -march=native varies; sometimes unstable due to thermal throttle"
echo
echo "The shape holds on aarch64, the ratios do not: on an Orin NX the middle"
echo "build matches -O2 rather than regressing, and -ffast-math buys ~4.9x."
