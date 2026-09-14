#!/usr/bin/env python3
"""CyclicBuffer + TimeSeries Python example.

Demonstrates writing and reading Stamped<double> values through a
cyclic ring buffer, with temporal lookups.

Can be used standalone (same process) or cross-process with C++ writer.
"""

import safe_shm_py as shm

SHM_NAME = "py_cyclic_demo"


def main():
    # Write 20 stamped values
    writer = shm.CyclicBufferWriterStampedF64(SHM_NAME)

    for i in range(20):
        s = shm.StampedF64()
        s.timestamp_ns = i * 1000  # synthetic timestamps
        s.sequence = i
        s.data = float(i) * 1.5
        writer.insert(s)

    print(f"Wrote {writer.total_writes()} items")

    # Read with CyclicBufferReader
    reader = shm.CyclicBufferReaderStampedF64(SHM_NAME)
    print(f"Available: {reader.available()} / capacity {reader.capacity}")

    latest = reader.get_latest()
    print(f"Latest: seq={latest.sequence} data={latest.data:.1f} ts={latest.timestamp_ns}")

    # Read history (reverse index: 0=latest, 1=previous, ...)
    print("\nHistory (newest first):")
    for i in range(min(5, reader.available())):
        item = reader.get(i)
        print(f"  [{i}] seq={item.sequence} data={item.data:.1f}")

    # TimeSeries temporal queries
    ts = shm.TimeSeriesStampedF64(SHM_NAME)

    # Find closest to timestamp 7500 (between seq=7 @ ts=7000 and seq=8 @ ts=8000)
    closest = ts.find_closest(7500)
    if closest is not None:
        print(f"\nClosest to ts=7500: seq={closest.sequence} ts={closest.timestamp_ns}")

    # Interpolation pair
    interp = ts.find_interpolation_pair(7500)
    if interp is not None:
        print(f"Interpolation: before.seq={interp.before.sequence} "
              f"after.seq={interp.after.sequence} alpha={interp.alpha:.2f}")

    # Freshness check
    fresh = ts.get_latest_if_fresh(15000)
    if fresh is not None:
        print(f"Fresh (>= 15000): seq={fresh.sequence}")
    else:
        print("Data is stale (latest < 15000)")

    stale = ts.get_latest_if_fresh(99999)
    if stale is None:
        print("Confirmed stale (latest < 99999)")

    # SHM lifecycle utilities
    print(f"\nSHM exists '{SHM_NAME}': {shm.shm_exists(SHM_NAME)}")
    segments = shm.shm_list("py_")
    print(f"SHM segments with 'py_' prefix: {segments}")


if __name__ == "__main__":
    main()
