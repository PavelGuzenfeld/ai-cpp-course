#!/usr/bin/env python3
"""
Zero-Copy Binary Protocol Parsing — Practical addition to Lesson 5.

Demonstrates numpy.frombuffer + struct.pack for zero-copy binary protocol
parsing, a pattern common in drone telemetry systems where tracker_engine
receives MAVLink-like data over shared memory or sockets.

Run:
    python binary_protocol.py
"""

import struct
import sys
import time

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Telemetry packet format
# ---------------------------------------------------------------------------
# Field          Type     Bytes   Description
# -----          ----     -----   -----------
# timestamp_ns   uint64   8       Monotonic nanosecond timestamp
# lat            float64  8       Latitude  (degrees)
# lon            float64  8       Longitude (degrees)
# alt            float32  4       Altitude  (metres AGL)
# roll           float32  4       Roll      (radians)
# pitch          float32  4       Pitch     (radians)
# yaw            float32  4       Yaw       (radians)
#                         --
# Total:                  40 bytes per packet

PACKET_FMT = "<Qddffff"          # little-endian
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 40
assert PACKET_SIZE == 40, f"Expected 40 bytes, got {PACKET_SIZE}"

# Structured dtype that mirrors the C struct layout
PACKET_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("lat",          "<f8"),
    ("lon",          "<f8"),
    ("alt",          "<f4"),
    ("roll",         "<f4"),
    ("pitch",        "<f4"),
    ("yaw",          "<f4"),
])
assert PACKET_DTYPE.itemsize == PACKET_SIZE


# ---------------------------------------------------------------------------
# Pack a stream of packets (simulating a telemetry source)
# ---------------------------------------------------------------------------

def generate_packet_stream(n: int) -> bytes:
    """Create *n* contiguous binary telemetry packets."""
    rng = np.random.default_rng(0)
    buf = bytearray(n * PACKET_SIZE)
    for i in range(n):
        ts = 1_000_000_000 + i * 10_000_000        # 100 Hz
        lat = 32.0 + rng.normal(0, 0.0001)
        lon = -117.0 + rng.normal(0, 0.0001)
        alt = 100.0 + rng.normal(0, 0.5)
        roll = rng.normal(0, 0.05)
        pitch = rng.normal(0, 0.05)
        yaw = rng.uniform(-np.pi, np.pi)
        struct.pack_into(PACKET_FMT, buf, i * PACKET_SIZE,
                         ts, lat, lon, alt, roll, pitch, yaw)
    return bytes(buf)


def generate_packet_stream_numpy(n: int) -> bytes:
    """Faster stream generation using numpy (for large n)."""
    rng = np.random.default_rng(0)
    arr = np.empty(n, dtype=PACKET_DTYPE)
    arr["timestamp_ns"] = np.arange(n, dtype=np.uint64) * 10_000_000 + 1_000_000_000
    arr["lat"] = 32.0 + rng.normal(0, 0.0001, n)
    arr["lon"] = -117.0 + rng.normal(0, 0.0001, n)
    arr["alt"] = (100.0 + rng.normal(0, 0.5, n)).astype(np.float32)
    arr["roll"] = rng.normal(0, 0.05, n).astype(np.float32)
    arr["pitch"] = rng.normal(0, 0.05, n).astype(np.float32)
    arr["yaw"] = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    return arr.tobytes()


# ---------------------------------------------------------------------------
# Parsing approaches
# ---------------------------------------------------------------------------

def parse_struct_unpack(data: bytes, n: int) -> list[tuple]:
    """Parse with struct.unpack — copies every field into Python objects."""
    results = []
    for i in range(n):
        offset = i * PACKET_SIZE
        pkt = struct.unpack_from(PACKET_FMT, data, offset)
        results.append(pkt)
    return results


def parse_numpy_frombuffer(data: bytes, n: int) -> np.ndarray:
    """Parse with np.frombuffer — zero-copy view into the buffer."""
    arr = np.frombuffer(data, dtype=PACKET_DTYPE, count=n)
    return arr


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(func, *args, warmup: int = 2, repeats: int = 10, **kwargs) -> float:
    for _ in range(warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N = 10_000

    print("Zero-Copy Binary Protocol Parsing")
    print(f"  Packet size : {PACKET_SIZE} bytes")
    print(f"  Packet count: {N:,}")
    print(f"  Stream size : {N * PACKET_SIZE / 1024:.1f} KB")
    print()

    # Generate stream
    print("Generating packet stream ... ", end="", flush=True)
    data = generate_packet_stream_numpy(N)
    print(f"done ({len(data):,} bytes)")
    print()

    # --- Parse and verify equivalence ---
    result_struct = parse_struct_unpack(data, N)
    result_numpy = parse_numpy_frombuffer(data, N)

    # Spot-check first packet
    s = result_struct[0]
    n = result_numpy[0]
    assert s[0] == n["timestamp_ns"], "timestamp mismatch"
    assert abs(s[1] - n["lat"]) < 1e-12, "lat mismatch"
    assert abs(s[2] - n["lon"]) < 1e-12, "lon mismatch"
    print("Equivalence check: PASSED (first packet matches)")
    print()

    # --- Zero-copy proof ---
    print("Zero-copy verification:")
    buf = bytearray(data)  # mutable buffer
    view = np.frombuffer(buf, dtype=PACKET_DTYPE)
    print(f"  np.shares_memory(buf, view) = {np.shares_memory(buf, view)}")
    original_ts = view[0]["timestamp_ns"].copy()
    # Mutate the underlying buffer and observe the numpy view change
    new_ts = original_ts + 999
    struct.pack_into("<Q", buf, 0, new_ts)
    print(f"  After mutating buffer: view[0].timestamp_ns changed from "
          f"{original_ts} to {view[0]['timestamp_ns']}")
    print(f"  Confirms zero-copy: {view[0]['timestamp_ns'] == new_ts}")
    print()

    # --- Benchmark ---
    ms_struct = bench(parse_struct_unpack, data, N)
    ms_numpy = bench(parse_numpy_frombuffer, data, N)

    print(f"{'Method':<25} {'Time (ms)':>10} {'Speedup':>10}")
    print("-" * 47)
    print(f"{'struct.unpack (copy)':<25} {ms_struct:>10.3f} {'1.0x':>10}")
    print(f"{'np.frombuffer (zero-copy)':<25} {ms_numpy:>10.3f} {ms_struct/ms_numpy:>9.1f}x")
    print()

    # --- Field access benchmark ---
    print("Field access after parsing (extract all latitudes):")
    def access_struct():
        return [r[1] for r in result_struct]
    def access_numpy():
        return result_numpy["lat"]  # returns a view, no copy

    ms_s = bench(access_struct)
    ms_n = bench(access_numpy)
    print(f"  struct list comprehension : {ms_s:.3f} ms")
    print(f"  numpy field access (view) : {ms_n:.3f} ms  ({ms_s/ms_n:.0f}x faster)")
    print()

    print("Key takeaways:")
    print("  - np.frombuffer creates a zero-copy view of binary data")
    print("  - No per-packet Python object allocation overhead")
    print("  - Field access returns numpy views — still zero-copy")
    print("  - Ideal for high-rate telemetry (MAVLink, custom protocols)")
    print("  - In tracker_engine, shared-memory telemetry uses this pattern")
    print()


if __name__ == "__main__":
    main()
