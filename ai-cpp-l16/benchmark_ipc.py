"""
Benchmark: copy vs. zero-copy IPC at two frame sizes. CSV to stdout.

The control size isolates pure sync/fd-transfer overhead from the cost of
actually moving bytes -- the same technique the source project used to tell
"we removed the copy" from "we removed the work".

Run:
    python3 benchmark_ipc.py
"""
import csv
import sys
import threading
import time

from ipc_native import (
    ShmFrame,
    accept_unix_socket,
    close_fd,
    connect_unix_socket,
    listen_unix_socket,
    recv_bytes,
    recv_fd,
    send_bytes,
    send_fd,
)

FRAME_SIZES = {
    "control_64x64_rgba": 64 * 64 * 4,
    "frame_1080p_rgb": 1920 * 1080 * 3,
}
ITERATIONS = 200


def _make_pair(path):
    listen_fd = listen_unix_socket(path)
    client_fd = connect_unix_socket(path)
    server_fd = accept_unix_socket(listen_fd)
    close_fd(listen_fd)
    return server_fd, client_fd


def bench_copy(size, path):
    # A frame larger than the socket's send buffer would deadlock a
    # synchronous send-then-recv in one process: send() blocks for buffer
    # space nothing is draining yet. Send from a background thread so the
    # two sides run concurrently, matching the real producer/consumer shape.
    server_fd, client_fd = _make_pair(path)
    payload = b"\x00" * size

    def sender():
        for _ in range(ITERATIONS):
            send_bytes(server_fd, payload)

    t0 = time.perf_counter()
    sender_thread = threading.Thread(target=sender)
    sender_thread.start()
    for _ in range(ITERATIONS):
        recv_bytes(client_fd, size)
    sender_thread.join()
    elapsed = time.perf_counter() - t0

    close_fd(server_fd)
    close_fd(client_fd)
    return elapsed


def bench_zero_copy(size, path, shm_name):
    server_fd, client_fd = _make_pair(path)
    frame = ShmFrame(shm_name, size)
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        send_fd(server_fd, frame.fd())
        received_fd = recv_fd(client_fd)
        close_fd(received_fd)
    elapsed = time.perf_counter() - t0
    close_fd(server_fd)
    close_fd(client_fd)
    return elapsed


def main() -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["path", "frame", "size_bytes", "iterations", "elapsed_s", "us_per_transfer"])
    for label, size in FRAME_SIZES.items():
        copy_s = bench_copy(size, f"/tmp/bench_copy_{label}.sock")
        writer.writerow(["copy", label, size, ITERATIONS,
                          f"{copy_s:.6f}", f"{copy_s / ITERATIONS * 1e6:.2f}"])

        zc_s = bench_zero_copy(size, f"/tmp/bench_zc_{label}.sock", f"bench_zc_{label}")
        writer.writerow(["zero-copy", label, size, ITERATIONS,
                          f"{zc_s:.6f}", f"{zc_s / ITERATIONS * 1e6:.2f}"])


if __name__ == "__main__":
    main()
