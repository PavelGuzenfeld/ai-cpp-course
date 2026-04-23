"""Benchmark __builtin_* vs their C++20/23 std:: alternatives.

Every pair should finish in the same ns/op because the std:: name lowers to
the same __builtin_ underneath on libstdc++.

Run:
    python3 benchmark_builtins.py
"""
import random
import time

import builtins_demo as bd


def measure(label, fn, iters=50):
    # Warm up once to populate branch-target buffers.
    fn()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter_ns()
    per = (t1 - t0) / iters
    print(f"  {label:<32s}  {per:>12.0f} ns/iter")
    return per


def main() -> None:
    rng = random.Random(42)
    v = [rng.getrandbits(64) for _ in range(1 << 16)]

    print("popcount (1M 64-bit values)")
    b = measure("__builtin_popcountll",  lambda: bd.popcount_sum_builtin(v))
    s = measure("std::popcount",         lambda: bd.popcount_sum_std(v))
    print(f"  ratio: {b / s:.2f}x  (expect ~1.0x — same instruction)")

    print("\nbyte swap (XOR-accumulate over the same 1M values)")
    b = measure("__builtin_bswap64",     lambda: bd.bswap_xor_builtin(v))
    s = measure("std::byteswap",         lambda: bd.bswap_xor_std(v))
    print(f"  ratio: {b / s:.2f}x  (expect ~1.0x)")

    print("\nsingle-op dispatches (just to confirm the bindings work)")
    print(f"  popcount(42) builtin={bd.popcount_builtin(42)}  std={bd.popcount_std(42)}")
    print(f"  clz(42)      builtin={bd.clz_builtin(42)}        std={bd.clz_std(42)}")
    print(f"  bswap(1)     builtin=0x{bd.bswap_builtin(1):016x}  std=0x{bd.bswap_std(1):016x}")
    print(f"  apply(Add, 2, 3) builtin={bd.apply_builtin(0, 2, 3)}  std={bd.apply_std(0, 2, 3)}")


if __name__ == "__main__":
    main()
