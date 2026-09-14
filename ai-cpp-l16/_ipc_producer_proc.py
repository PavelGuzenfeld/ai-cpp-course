"""
Helper process for the L16 IPC integration test.

mode=zerocopy: creates a shared-memory frame, writes the payload into it,
listens on a Unix socket, and hands the frame's fd to the first connecting
consumer via SCM_RIGHTS -- no bytes of the payload itself cross the socket.

mode=copy: listens on the same kind of socket and streams the payload as
raw bytes -- the baseline the zero-copy path is measured against.

Usage: _ipc_producer_proc.py <mode> <socket_path> <shm_name> <payload_file> <ready_file> <done_file>
"""

import sys
import time

from ipc_native import (
    ShmFrame,
    accept_unix_socket,
    close_fd,
    listen_unix_socket,
    send_bytes,
    send_fd,
)

TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.02


def main() -> int:
    mode, socket_path, shm_name, payload_file, ready_file, done_file = sys.argv[1:7]

    with open(payload_file, "rb") as f:
        payload = f.read()

    frame = None
    if mode == "zerocopy":
        frame = ShmFrame(shm_name, len(payload))
        frame.write(payload)

    listen_fd = listen_unix_socket(socket_path)
    with open(ready_file, "w") as f:
        f.write("ready")

    client_fd = accept_unix_socket(listen_fd)
    if mode == "zerocopy":
        assert frame is not None
        send_fd(client_fd, frame.fd())
    else:
        send_bytes(client_fd, payload)
    close_fd(client_fd)
    close_fd(listen_fd)

    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with open(done_file, "r"):
                return 0
        except FileNotFoundError:
            time.sleep(POLL_INTERVAL_S)

    sys.stderr.write("producer: timed out waiting for consumer\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
