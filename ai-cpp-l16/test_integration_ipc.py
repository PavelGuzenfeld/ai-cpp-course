"""
Integration test for Lesson 16: a real producer/consumer pair across two
processes, over both the zero-copy (SCM_RIGHTS) and copy paths.
"""

import os
import subprocess
import sys
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))


def _run_roundtrip(mode: str, payload: bytes) -> tuple[bytes, int]:
    """Runs the producer/consumer pair over `payload`, returns
    (bytes the consumer received, zero_copy_transfer_count it observed)."""
    tag = uuid.uuid4().hex
    socket_path = f"/tmp/ai_cpp_l16_{tag}.sock"
    shm_name = f"ai_cpp_l16_{tag}"

    with tempfile.TemporaryDirectory() as tmp:
        payload_file = os.path.join(tmp, "payload")
        ready_file = os.path.join(tmp, "ready")
        done_file = os.path.join(tmp, "done")
        out_file = os.path.join(tmp, "out")
        count_file = os.path.join(tmp, "count")

        with open(payload_file, "wb") as f:
            f.write(payload)

        try:
            producer = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(HERE, "_ipc_producer_proc.py"),
                    mode, socket_path, shm_name, payload_file, ready_file, done_file,
                ],
                cwd=HERE,
            )

            consumer = subprocess.run(
                [
                    sys.executable,
                    os.path.join(HERE, "_ipc_consumer_proc.py"),
                    mode, socket_path, str(len(payload)), ready_file, done_file, out_file, count_file,
                ],
                cwd=HERE,
                timeout=20,
            )
            assert consumer.returncode == 0, "consumer process failed"

            producer_rc = producer.wait(timeout=20)
            assert producer_rc == 0, "producer process failed"

            with open(out_file, "rb") as f:
                received = f.read()
            with open(count_file) as f:
                count = int(f.read())

            return received, count
        finally:
            for p in (socket_path,):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


class TestPixelPerfectRoundtrip:
    def test_zero_copy_path_is_byte_for_byte_identical(self):
        payload = bytes((i * 37 + 11) % 256 for i in range(64 * 1024))
        received, _ = _run_roundtrip("zerocopy", payload)
        assert received == payload

    def test_copy_path_is_byte_for_byte_identical(self):
        payload = bytes((i * 37 + 11) % 256 for i in range(64 * 1024))
        received, _ = _run_roundtrip("copy", payload)
        assert received == payload


class TestZeroCopyVerification:
    """The falsification step: how would you know if you had silently
    fallen back to the copy path? Check that the fast path was actually
    used, not just that the bytes came out right."""

    def test_zerocopy_mode_actually_receives_an_fd(self):
        _, count = _run_roundtrip("zerocopy", b"\x00" * 4096)
        assert count >= 1

    def test_copy_mode_never_receives_an_fd(self):
        _, count = _run_roundtrip("copy", b"\x00" * 4096)
        assert count == 0
