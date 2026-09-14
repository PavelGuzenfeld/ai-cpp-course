"""
Unit tests for the Lesson 3 shared-memory round-trip module.

Tests:
  - ShmWriter/ShmReader round-trip within one process
  - Independent segments (different names) do not collide
  - Re-opening a live segment does not clobber previously written data
"""

import os
import sys
import uuid

sys.path.insert(0, ".")

from shm_roundtrip_native import ShmReader, ShmWriter  # noqa: E402


def _unique_name() -> str:
    return f"ai_cpp_l3_test_{os.getpid()}_{uuid.uuid4().hex}"


class TestShmRoundtrip:
    def test_write_then_read(self):
        name = _unique_name()
        writer = ShmWriter(name)
        writer.write(1.5, -2.5, 3.0, 42)

        reader = ShmReader(name)
        assert reader.read() == (1.5, -2.5, 3.0, 42)

    def test_overwrite_is_visible(self):
        name = _unique_name()
        writer = ShmWriter(name)
        reader = ShmReader(name)

        writer.write(0.0, 0.0, 0.0, 1)
        assert reader.read() == (0.0, 0.0, 0.0, 1)

        writer.write(9.9, 8.8, 7.7, 2)
        assert reader.read() == (9.9, 8.8, 7.7, 2)

    def test_independent_segments_do_not_collide(self):
        name_a = _unique_name()
        name_b = _unique_name()

        writer_a = ShmWriter(name_a)
        writer_b = ShmWriter(name_b)
        writer_a.write(1.0, 1.0, 1.0, 1)
        writer_b.write(2.0, 2.0, 2.0, 2)

        assert ShmReader(name_a).read() == (1.0, 1.0, 1.0, 1)
        assert ShmReader(name_b).read() == (2.0, 2.0, 2.0, 2)

    def test_reopening_existing_segment_preserves_data(self):
        """A second ShmWriter/ShmReader on the same live name must not
        truncate away the data the first writer already wrote — the
        segment's ftruncate is a no-op when the size is unchanged."""
        name = _unique_name()
        first_writer = ShmWriter(name)
        first_writer.write(4.0, 5.0, 6.0, 7)

        second_reader = ShmReader(name)
        assert second_reader.read() == (4.0, 5.0, 6.0, 7)
