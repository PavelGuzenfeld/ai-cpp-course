"""
Helper process for test_integration_shm.py.

Waits for the writer's ready sentinel, opens the same named shared-memory
segment, reads it, and writes the result to out_file as "x,y,z,seq" --
then signals the writer that it is safe to exit.

Usage: _shm_reader_proc.py <name> <ready_file> <done_file> <out_file>
"""

import sys
import time

from shm_roundtrip_native import ShmReader

TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.02


def _wait_for(path: str) -> bool:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with open(path, "r"):
                return True
        except FileNotFoundError:
            time.sleep(POLL_INTERVAL_S)
    return False


def main() -> int:
    name, ready_file, done_file, out_file = sys.argv[1:5]

    if not _wait_for(ready_file):
        sys.stderr.write("reader: timed out waiting for writer\n")
        return 1

    reader = ShmReader(name)
    x, y, z, seq = reader.read()

    with open(out_file, "w") as f:
        f.write(f"{x},{y},{z},{seq}")

    with open(done_file, "w") as f:
        f.write("done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
