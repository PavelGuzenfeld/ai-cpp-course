"""
Integration test for Lesson 3: shared memory across a real process boundary.

Spawns a writer process and a reader process (Exercise 1 from the README) and
verifies the reader observes the exact struct the writer wrote through
/dev/shm, not through any in-process Python state.
"""

import os
import subprocess
import sys
import tempfile
import uuid


def _unique_name() -> str:
    return f"ai_cpp_l3_itest_{os.getpid()}_{uuid.uuid4().hex}"


class TestShmCrossProcessRoundtrip:
    def test_writer_and_reader_in_separate_processes(self):
        name = _unique_name()
        here = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as tmp:
            ready_file = os.path.join(tmp, "ready")
            done_file = os.path.join(tmp, "done")
            out_file = os.path.join(tmp, "out")

            writer = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(here, "_shm_writer_proc.py"),
                    name, "1.25", "-3.5", "42.0", "7",
                    ready_file, done_file,
                ],
                cwd=here,
            )

            reader = subprocess.run(
                [
                    sys.executable,
                    os.path.join(here, "_shm_reader_proc.py"),
                    name, ready_file, done_file, out_file,
                ],
                cwd=here,
                timeout=15,
            )
            assert reader.returncode == 0, "reader process failed"

            writer_rc = writer.wait(timeout=15)
            assert writer_rc == 0, "writer process failed"

            with open(out_file) as f:
                x, y, z, seq = f.read().split(",")

            assert float(x) == 1.25
            assert float(y) == -3.5
            assert float(z) == 42.0
            assert int(seq) == 7
