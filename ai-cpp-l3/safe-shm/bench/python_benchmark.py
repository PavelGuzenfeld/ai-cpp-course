#!/usr/bin/env python3
"""Performance benchmarks for safe_shm_py Python bindings.

Measures throughput and latency for seqlock, cyclic buffer, time series,
and cross-process round-trip operations.

Usage:
    python3 python_benchmark.py [--iterations N]
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

import safe_shm_py as shm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DEFAULT = 100_000
SHM_PREFIX = "pybench_"


def _shm_name(suffix: str) -> str:
    return f"{SHM_PREFIX}{suffix}_{os.getpid()}"


def _cleanup(names: list[str]) -> None:
    for name in names:
        try:
            shm.shm_remove(name)
        except Exception:
            pass


def _fmt_ops(ops: float) -> str:
    if ops >= 1e6:
        return f"{ops / 1e6:>8.2f} Mops/s"
    if ops >= 1e3:
        return f"{ops / 1e3:>8.2f} Kops/s"
    return f"{ops:>8.2f}  ops/s"


def _fmt_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:>8.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:>8.2f} us"
    return f"{ns:>8.0f} ns"


def _print_row(label: str, total_ns: int, n: int) -> None:
    avg_ns = total_ns / n
    ops = n / (total_ns / 1e9)
    print(f"  {label:<44s}  {_fmt_ops(ops)}  {_fmt_ns(avg_ns)}/op  (n={n})")


def _section(title: str) -> None:
    print()
    print(f"{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# 1. Seqlock write/read throughput (Stamped<double>)
# ---------------------------------------------------------------------------

def bench_seqlock_stamped(n: int) -> list[str]:
    _section("Seqlock Stamped<double> throughput")

    name = _shm_name("seqlock_stamped")
    writer = shm.SeqlockWriterStampedF64(name)
    reader = shm.SeqlockReaderStampedF64(name)

    # -- write throughput --
    val = shm.stamp_f64(1.0, seq=0)
    t0 = time.perf_counter_ns()
    for i in range(n):
        val.sequence = i
        val.data = float(i)
        writer.store(val)
    t1 = time.perf_counter_ns()
    _print_row("SeqlockWriterStampedF64.store()", t1 - t0, n)

    # -- read throughput --
    t0 = time.perf_counter_ns()
    for _ in range(n):
        reader.load()
    t1 = time.perf_counter_ns()
    _print_row("SeqlockReaderStampedF64.load()", t1 - t0, n)

    # -- write+read round-trip (same process) --
    t0 = time.perf_counter_ns()
    for i in range(n):
        val.sequence = i
        val.data = float(i)
        writer.store(val)
        reader.load()
    t1 = time.perf_counter_ns()
    _print_row("store() + load() round-trip", t1 - t0, n)

    return [name]


# ---------------------------------------------------------------------------
# 2. CyclicBuffer insert/read throughput (Stamped<double>, capacity 64)
# ---------------------------------------------------------------------------

def bench_cyclic_buffer(n: int) -> list[str]:
    _section("CyclicBuffer Stamped<double> throughput (capacity=64)")

    name = _shm_name("cyclic")
    writer = shm.CyclicBufferWriterStampedF64(name)
    reader = shm.CyclicBufferReaderStampedF64(name)

    val = shm.StampedF64()

    # -- insert throughput --
    t0 = time.perf_counter_ns()
    for i in range(n):
        val.timestamp_ns = i * 1000
        val.sequence = i
        val.data = float(i) * 0.5
        writer.insert(val)
    t1 = time.perf_counter_ns()
    _print_row("CyclicBufferWriterStampedF64.insert()", t1 - t0, n)

    # -- get_latest throughput --
    t0 = time.perf_counter_ns()
    for _ in range(n):
        reader.get_latest()
    t1 = time.perf_counter_ns()
    _print_row("CyclicBufferReaderStampedF64.get_latest()", t1 - t0, n)

    # -- get(reverse_index) throughput --
    t0 = time.perf_counter_ns()
    for _ in range(n):
        reader.get(0)
    t1 = time.perf_counter_ns()
    _print_row("CyclicBufferReaderStampedF64.get(0)", t1 - t0, n)

    return [name]


# ---------------------------------------------------------------------------
# 3. TimeSeries query latency
# ---------------------------------------------------------------------------

def bench_time_series(n: int) -> list[str]:
    _section("TimeSeries Stamped<double> query latency (capacity=64)")

    name = _shm_name("timeseries")
    writer = shm.CyclicBufferWriterStampedF64(name)
    ts = shm.TimeSeriesStampedF64(name)

    # Pre-fill buffer with 64 elements (monotonic timestamps)
    for i in range(64):
        val = shm.StampedF64()
        val.timestamp_ns = (i + 1) * 10_000  # 10us apart
        val.sequence = i
        val.data = float(i)
        writer.insert(val)

    mid_ts = 32 * 10_000  # target: middle of the buffer
    latest_ts = 64 * 10_000

    # -- find_closest --
    t0 = time.perf_counter_ns()
    for _ in range(n):
        ts.find_closest(mid_ts)
    t1 = time.perf_counter_ns()
    _print_row("find_closest(mid)", t1 - t0, n)

    # -- find_interpolation_pair --
    target_between = 32 * 10_000 + 5_000  # between two entries
    t0 = time.perf_counter_ns()
    for _ in range(n):
        ts.find_interpolation_pair(target_between)
    t1 = time.perf_counter_ns()
    _print_row("find_interpolation_pair(mid)", t1 - t0, n)

    # -- get_latest_if_fresh (fresh case) --
    stale_threshold = 1  # very old, so latest is always fresh
    t0 = time.perf_counter_ns()
    for _ in range(n):
        ts.get_latest_if_fresh(stale_threshold)
    t1 = time.perf_counter_ns()
    _print_row("get_latest_if_fresh() [fresh]", t1 - t0, n)

    # -- get_latest_if_fresh (stale case) --
    future_threshold = latest_ts * 100  # way in the future
    t0 = time.perf_counter_ns()
    for _ in range(n):
        ts.get_latest_if_fresh(future_threshold)
    t1 = time.perf_counter_ns()
    _print_row("get_latest_if_fresh() [stale]", t1 - t0, n)

    return [name]


# ---------------------------------------------------------------------------
# 4. Cross-process round-trip latency
# ---------------------------------------------------------------------------

def _child_writer(shm_name: str, n: int, ready_event, go_event) -> None:
    """Child process: write sequentially, each with monotonic timestamp."""
    writer = shm.SeqlockWriterStampedF64(shm_name)
    ready_event.set()
    go_event.wait()

    for i in range(n):
        val = shm.stamp_f64(float(i), seq=i + 1)
        writer.store(val)


def bench_cross_process(n_rounds: int) -> list[str]:
    _section("Cross-process round-trip latency")

    name = _shm_name("xproc")
    n = min(n_rounds, 10_000)  # keep cross-process test shorter

    ready_event = multiprocessing.Event()
    go_event = multiprocessing.Event()

    child = multiprocessing.Process(
        target=_child_writer, args=(name, n, ready_event, go_event)
    )
    child.start()
    ready_event.wait(timeout=5.0)

    reader = shm.SeqlockReaderStampedF64(name)

    # Let the child write, parent reads and measures end-to-end
    go_event.set()
    child.join(timeout=10.0)
    if child.exitcode is None:
        child.terminate()
        print("  WARNING: child process timed out")
        return [name]

    # Now measure parent read throughput on data written by child
    # (data is already in SHM; measures read path only but proves cross-process)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        reader.load()
    t1 = time.perf_counter_ns()
    _print_row("cross-process SeqlockReader.load()", t1 - t0, n)

    # Measure true round-trip: parent writes timestamp, child reads and writes
    # back, parent reads. We approximate with sequential write-then-read.
    name_rt = _shm_name("xproc_rt")

    def _roundtrip_child(shm_rd: str, shm_wr: str, count: int, rdy, go) -> None:
        r = shm.SeqlockReaderStampedF64(shm_rd)
        w = shm.SeqlockWriterStampedF64(shm_wr)
        rdy.set()
        go.wait()
        last_seq = 0
        replied = 0
        while replied < count:
            val = r.load()
            if val.sequence > last_seq:
                last_seq = val.sequence
                w.store(shm.stamp_f64(val.data, seq=last_seq))
                replied += 1

    ready2 = multiprocessing.Event()
    go2 = multiprocessing.Event()
    child2 = multiprocessing.Process(
        target=_roundtrip_child,
        args=(name_rt + "_a", name_rt + "_b", n, ready2, go2),
    )
    child2.start()
    ready2.wait(timeout=5.0)

    parent_writer = shm.SeqlockWriterStampedF64(name_rt + "_a")
    parent_reader = shm.SeqlockReaderStampedF64(name_rt + "_b")

    go2.set()

    latencies: list[int] = []
    for i in range(1, n + 1):
        t_send = time.perf_counter_ns()
        parent_writer.store(shm.stamp_f64(float(i), seq=i))
        # Spin-wait for child reply
        while True:
            reply = parent_reader.load()
            if reply.sequence >= i:
                break
        t_recv = time.perf_counter_ns()
        latencies.append(t_recv - t_send)

    child2.join(timeout=10.0)
    if child2.exitcode is None:
        child2.terminate()

    total_ns = sum(latencies)
    _print_row("round-trip (write->child->read)", total_ns, n)

    # Percentile stats
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    p999 = latencies[int(len(latencies) * 0.999)]
    print(f"  {'Percentiles:':<44s}  p50={_fmt_ns(p50)}  p99={_fmt_ns(p99)}  p99.9={_fmt_ns(p999)}")

    return [name, name_rt + "_a", name_rt + "_b"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="safe_shm_py performance benchmarks")
    parser.add_argument(
        "--iterations", "-n", type=int, default=N_DEFAULT,
        help=f"Number of iterations per benchmark (default: {N_DEFAULT})",
    )
    args = parser.parse_args()
    n = args.iterations

    print(f"safe_shm_py benchmark  (pid={os.getpid()}, iterations={n})")
    print(f"Python {sys.version}")
    print(f"Timing: time.perf_counter_ns()")

    all_shm_names: list[str] = []

    try:
        all_shm_names += bench_seqlock_stamped(n)
        all_shm_names += bench_cyclic_buffer(n)
        all_shm_names += bench_time_series(n)
        all_shm_names += bench_cross_process(n)
    finally:
        _section("Cleanup")
        _cleanup(all_shm_names)
        # Also clean any stale segments from this prefix
        for seg in shm.shm_list(SHM_PREFIX):
            try:
                shm.shm_remove(seg)
            except Exception:
                pass
        print(f"  Removed {len(all_shm_names)} SHM segments")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
