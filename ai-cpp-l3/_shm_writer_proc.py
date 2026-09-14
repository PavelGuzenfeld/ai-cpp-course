"""
Helper process for test_integration_shm.py.

Writes one SensorReading into a named shared-memory segment, signals the
reader via a sentinel file, then blocks until the reader signals it is done
-- so the writer's RAII destructor (which unlinks the segment) does not fire
before the reader has had a chance to open it.

Usage: _shm_writer_proc.py <name> <x> <y> <z> <seq> <ready_file> <done_file>
"""

import sys
import time

from shm_roundtrip_native import ShmWriter

TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.02


def main() -> int:
    name, x, y, z, seq, ready_file, done_file = sys.argv[1:8]

    writer = ShmWriter(name)
    writer.write(float(x), float(y), float(z), int(seq))

    with open(ready_file, "w") as f:
        f.write("ready")

    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with open(done_file, "r"):
                return 0
        except FileNotFoundError:
            time.sleep(POLL_INTERVAL_S)

    sys.stderr.write("writer: timed out waiting for reader\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
