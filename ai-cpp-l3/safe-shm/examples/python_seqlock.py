#!/usr/bin/env python3
"""Seqlock Python example: read sensor data written by C++.

Run the C++ writer first:
    ./seqlock_example writer

Then run this reader:
    python3 python_seqlock.py
"""

import safe_shm_py as shm
import time

SHM_NAME = "example_seqlock_sensor"


def main():
    # Write from Python
    writer = shm.SeqlockWriterF64("py_seqlock_demo")
    writer.store(3.14159)
    print(f"Wrote: 3.14159")

    # Read from Python
    reader = shm.SeqlockReaderF64("py_seqlock_demo")
    value = reader.load()
    print(f"Read:  {value}")
    print(f"Sequence: {reader.sequence()}")

    # Stamped values
    stamped = shm.stamp_f64(42.0, seq=1)
    print(f"\nStamped: {stamped}")
    print(f"  timestamp_ns: {stamped.timestamp_ns}")
    print(f"  sequence:     {stamped.sequence}")
    print(f"  data:         {stamped.data}")

    # Monotonic clock
    t1 = shm.monotonic_now_ns()
    time.sleep(0.001)
    t2 = shm.monotonic_now_ns()
    print(f"\nMonotonic clock delta: {(t2 - t1) / 1e6:.2f} ms")


if __name__ == "__main__":
    main()
