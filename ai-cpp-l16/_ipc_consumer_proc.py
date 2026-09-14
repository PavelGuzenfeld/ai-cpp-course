"""
Helper process for the L16 IPC integration test. See _ipc_producer_proc.py.

Usage: _ipc_consumer_proc.py <mode> <socket_path> <payload_size> <ready_file> <done_file> <out_file> <count_file>
"""

import sys
import time

from ipc_native import (
    ShmFrameView,
    close_fd,
    connect_unix_socket,
    recv_bytes,
    recv_fd,
    zero_copy_transfer_count,
)

TIMEOUT_S = 15.0
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
    mode, socket_path, payload_size, ready_file, done_file, out_file, count_file = sys.argv[1:8]
    size = int(payload_size)

    if not _wait_for(ready_file):
        sys.stderr.write("consumer: timed out waiting for producer\n")
        return 1

    sock_fd = connect_unix_socket(socket_path)
    if mode == "zerocopy":
        shm_fd = recv_fd(sock_fd)
        view = ShmFrameView(shm_fd, size)
        received = bytes(view.read())
    else:
        received = bytes(recv_bytes(sock_fd, size))
    close_fd(sock_fd)

    with open(out_file, "wb") as f:
        f.write(received)

    with open(count_file, "w") as f:
        f.write(str(zero_copy_transfer_count()))

    with open(done_file, "w") as f:
        f.write("done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
