"""
Unit tests for Lesson 16: the shared-memory and socket primitives in
isolation, all in-process (see test_integration_ipc.py for the real
two-process round trip).
"""

import os
import sys
import uuid

sys.path.insert(0, ".")

from ipc_native import (  # noqa: E402
    ShmFrame,
    ShmFrameView,
    accept_unix_socket,
    close_fd,
    connect_unix_socket,
    listen_unix_socket,
    recv_bytes,
    recv_fd,
    send_bytes,
    send_fd,
    zero_copy_transfer_count,
)


def _unique_name() -> str:
    return f"ai_cpp_l16_test_{os.getpid()}_{uuid.uuid4().hex}"


class TestShmFrame:
    def test_write_then_read_same_frame(self):
        frame = ShmFrame(_unique_name(), 16)
        payload = bytes(range(16))
        frame.write(payload)
        assert bytes(frame.read()) == payload

    def test_write_wrong_size_raises(self):
        frame = ShmFrame(_unique_name(), 16)
        try:
            frame.write(b"\x00" * 8)
            assert False, "expected an exception"
        except RuntimeError:
            pass


class TestShmFrameView:
    def test_view_over_the_producers_fd_sees_the_same_data(self):
        frame = ShmFrame(_unique_name(), 32)
        payload = bytes(range(32))
        frame.write(payload)

        # A dup(), not the original fd -- ShmFrameView closes what it's
        # given, and the test still needs `frame` alive afterward.
        view = ShmFrameView(os.dup(frame.fd()), 32)
        assert bytes(view.read()) == payload


class TestUnixSocketFdPassing:
    def test_send_fd_then_recv_fd_transfers_the_same_underlying_file(self):
        path = f"/tmp/{_unique_name()}.sock"
        listen_fd = listen_unix_socket(path)
        client_fd = connect_unix_socket(path)
        server_fd = accept_unix_socket(listen_fd)

        frame = ShmFrame(_unique_name(), 8)
        frame.write(b"\x01\x02\x03\x04\x05\x06\x07\x08")

        before = zero_copy_transfer_count()
        send_fd(server_fd, frame.fd())
        received_fd = recv_fd(client_fd)
        assert zero_copy_transfer_count() == before + 1

        view = ShmFrameView(received_fd, 8)
        assert bytes(view.read()) == b"\x01\x02\x03\x04\x05\x06\x07\x08"

        close_fd(server_fd)
        close_fd(client_fd)
        close_fd(listen_fd)
        os.unlink(path)

    def test_send_bytes_then_recv_bytes_round_trips(self):
        path = f"/tmp/{_unique_name()}.sock"
        listen_fd = listen_unix_socket(path)
        client_fd = connect_unix_socket(path)
        server_fd = accept_unix_socket(listen_fd)

        payload = bytes(range(200)) * 3  # 600 bytes, larger than one packet
        send_bytes(server_fd, payload)
        received = recv_bytes(client_fd, len(payload))
        assert bytes(received) == payload

        close_fd(server_fd)
        close_fd(client_fd)
        close_fd(listen_fd)
        os.unlink(path)
