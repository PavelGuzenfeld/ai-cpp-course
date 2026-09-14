"""
Unit tests for Lesson 15: SPSC ring and GIL release.

Tests:
  - The correct ring delivers every item, in order, across real OS threads
  - The publish-before-write ring demonstrably loses the ordering guarantee
  - Releasing the GIL lets a background Python thread make real progress
"""

import sys
import threading

sys.path.insert(0, ".")

from concurrency_native import busy_wait_ms, run_broken_ring, run_correct_ring  # noqa: E402


class TestSpscRing:
    def test_correct_ring_delivers_every_item_in_order(self):
        n = 5000
        received = run_correct_ring(n)
        assert received == list(range(n))

    def test_broken_ring_delivers_a_stale_duplicate_value(self):
        # publish-before-write lets the consumer read a slot the producer
        # hasn't overwritten yet, i.e. a value already delivered earlier --
        # not merely "some position differs from range(n)".
        n = 2000
        received = run_broken_ring(n)
        assert len(received) - len(set(received)) > 0


def _count_background_progress_during(release_gil):
    counter = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter[0] += 1

    t = threading.Thread(target=spin)
    t.start()
    busy_wait_ms(200, release_gil=release_gil)
    stop.set()
    t.join()  # t.join() itself drops the GIL, so this also measures the
    # unavoidable handoff tail after busy_wait_ms returns -- that tail is
    # the same in both cases, so it cancels out of the ratio below.
    return counter[0]


class TestGilRelease:
    def test_releasing_the_gil_lets_the_background_thread_run_far_more(self):
        held = _count_background_progress_during(release_gil=False)
        released = _count_background_progress_during(release_gil=True)

        # Both measurements include the same post-call GIL-handoff tail
        # (stop.set() and t.join() release it regardless); only the
        # released case accumulates progress *during* the 200ms call
        # itself, so it must be a large multiple of the held case. 5x
        # clears that tail's contribution with margin on any reasonable box.
        assert released > 5 * held
