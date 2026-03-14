"""
Benchmarks for Lesson 8: Compile-Time Concepts for Performance.

Compares:
  1. State machine: Python string vs C++ string vs C++ variant
  2. Grayscale conversion: runtime computation vs compile-time LUT
  3. Gamma correction: runtime pow() vs compile-time LUT
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from state_machine_slow import StringStateMachine as PyStringSM

import compile_time_lut
import state_machine


def benchmark_state_machines(iterations: int = 100_000):
    """Benchmark 1: State machine dispatch overhead."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK 1: State Machine ({iterations:,} iterations)")
    print(f"{'=' * 70}")

    # --- Python string-based ---
    sm = PyStringSM()
    start = time.perf_counter()
    for _ in range(iterations):
        sm.update(True, 100.0, 200.0, 50.0, 50.0)
        sm.update(True, 105.0, 205.0, 50.0, 50.0)
        sm.update(False)
        for _ in range(31):
            sm.update(False)
        sm.update(True, 110.0, 210.0, 50.0, 50.0)
        sm.update(False)
        sm.update(True, 115.0, 215.0, 50.0, 50.0)
    py_time_us = (time.perf_counter() - start) * 1e6

    # --- C++ string-based ---
    cpp_str_time_us = state_machine.benchmark_string_sm(iterations)

    # --- C++ variant-based ---
    cpp_var_time_us = state_machine.benchmark_variant_sm(iterations)

    print(f"\n{'Method':<35} {'Time (us)':>12} {'Speedup':>10}")
    print(f"{'-' * 57}")
    print(f"{'Python string state machine':<35} {py_time_us:>12,.0f} {'1.0x':>10}")
    print(
        f"{'C++ string state machine':<35} {cpp_str_time_us:>12,.0f} "
        f"{py_time_us / cpp_str_time_us:>9.1f}x"
    )
    print(
        f"{'C++ variant state machine':<35} {cpp_var_time_us:>12,.0f} "
        f"{py_time_us / cpp_var_time_us:>9.1f}x"
    )
    if cpp_str_time_us > 0:
        print(
            f"\n  variant vs string (C++ only): "
            f"{cpp_str_time_us / cpp_var_time_us:.1f}x faster"
        )


def benchmark_grayscale(iterations: int = 1_000):
    """Benchmark 2: Grayscale conversion — runtime vs compile-time LUT."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK 2: BGR-to-Grayscale ({iterations:,} iterations, 640x480)")
    print(f"{'=' * 70}")

    # Create a synthetic BGR image
    bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # --- Runtime computation ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compile_time_lut.apply_grayscale_runtime(bgr)
    runtime_us = (time.perf_counter() - start) * 1e6

    # --- Compile-time LUT ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compile_time_lut.apply_grayscale_lut(bgr)
    lut_us = (time.perf_counter() - start) * 1e6

    print(f"\n{'Method':<35} {'Time (us)':>12} {'Speedup':>10}")
    print(f"{'-' * 57}")
    print(f"{'Runtime computation':<35} {runtime_us:>12,.0f} {'1.0x':>10}")
    print(
        f"{'Compile-time LUT':<35} {lut_us:>12,.0f} "
        f"{runtime_us / lut_us:>9.1f}x"
    )


def benchmark_gamma(iterations: int = 1_000):
    """Benchmark 3: Gamma correction — runtime pow() vs compile-time LUT."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK 3: Gamma Correction ({iterations:,} iterations, 640x480)")
    print(f"{'=' * 70}")

    gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    gamma = 2.2

    # --- Runtime pow() ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compile_time_lut.apply_gamma_runtime(gray, gamma)
    runtime_us = (time.perf_counter() - start) * 1e6

    # --- Compile-time LUT ---
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compile_time_lut.apply_gamma_lut(gray, gamma)
    lut_us = (time.perf_counter() - start) * 1e6

    print(f"\n{'Method':<35} {'Time (us)':>12} {'Speedup':>10}")
    print(f"{'-' * 57}")
    print(f"{'Runtime pow()':<35} {runtime_us:>12,.0f} {'1.0x':>10}")
    print(
        f"{'Compile-time LUT':<35} {lut_us:>12,.0f} "
        f"{runtime_us / lut_us:>9.1f}x"
    )


def main():
    print("Lesson 8: Compile-Time Concepts for Performance — Benchmarks")
    print("=" * 70)

    benchmark_state_machines(iterations=100_000)
    benchmark_grayscale(iterations=1_000)
    benchmark_gamma(iterations=1_000)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
