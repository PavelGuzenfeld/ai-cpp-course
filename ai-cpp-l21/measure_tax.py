"""Fills the tax table in machine_model.md, and the FFI-crossing row that
cannot be measured from C++ alone."""

import sys
import time

sys.path.insert(0, "/workspace/install/lib/python3.10/site-packages")

try:
    import tax_bench as tb
except ImportError:
    print("tax_bench module not built. colcon build --packages-select nanobind-l21")
    raise SystemExit(1)


def _median_ns(fn, repeats=5):
    return sorted(fn() for _ in range(repeats))[repeats // 2]


def ffi_crossing_ns(items=200_000):
    """Per-item cost one-call-per-item minus per-item cost one-call-per-batch.

    Both loops do the same total work on the C++ side, so the difference is
    the boundary itself. Measured from Python because the caller is the half
    that pays.
    """
    noop = tb.noop
    t0 = time.perf_counter_ns()
    for _ in range(items):
        noop()
    per_call = (time.perf_counter_ns() - t0) / items

    t0 = time.perf_counter_ns()
    tb.noop_batch(items)
    per_batched = (time.perf_counter_ns() - t0) / items

    return per_call, per_batched


def python_loop_floor_ns(items=200_000):
    """The empty Python loop, so the crossing number is not quietly claiming
    the interpreter's own per-iteration cost."""
    t0 = time.perf_counter_ns()
    for _ in range(items):
        pass
    return (time.perf_counter_ns() - t0) / items


def main():
    rows = [
        ("syscall floor (getpid)", lambda: tb.syscall_floor_ns()),
        ("clock_gettime, vDSO", lambda: tb.clock_gettime_vdso_ns()),
        ("clock_gettime, forced syscall", lambda: tb.clock_gettime_syscall_ns()),
        ("malloc+free, 64 B", lambda: tb.malloc_small_ns()),
        ("malloc+free+fault, 1 MiB", lambda: tb.malloc_page_faulted_ns()),
        ("mutex lock+unlock, uncontended", lambda: tb.mutex_uncontended_ns()),
        ("pipe round trip, 1 B", lambda: tb.pipe_roundtrip_ns()),
        ("socketpair round trip, 1 B", lambda: tb.socketpair_roundtrip_ns()),
    ]

    print(f"{'crossing':<34} {'ns':>12}")
    print("-" * 48)
    for label, fn in rows:
        print(f"{label:<34} {_median_ns(fn):>12.1f}")

    per_call, per_batched = ffi_crossing_ns()
    loop_floor = python_loop_floor_ns()
    print(f"{'nanobind call, one per item':<34} {per_call:>12.1f}")
    print(f"{'nanobind call, one per batch':<34} {per_batched:>12.4f}")
    print(f"{'(empty Python loop, per iter)':<34} {loop_floor:>12.1f}")
    print()
    print(f"crossing cost = {per_call - per_batched:.1f} ns/item, of which "
          f"{loop_floor:.1f} ns is the Python loop itself")
    print(f"batching {200_000} items into one call removes "
          f"{(per_call - per_batched) / per_call * 100:.1f}% of the per-item cost")


if __name__ == "__main__":
    main()
