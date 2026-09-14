"""
Integration tests for Lesson 15: sustained producer/consumer load, and the
GIL-release ring combined with real Python-side concurrent work.
"""

import sys
import threading
import time

sys.path.insert(0, ".")

from concurrency_native import busy_wait_ms, run_broken_ring, run_correct_ring  # noqa: E402


class TestSustainedLoad:
    def test_correct_ring_survives_many_rounds_at_increasing_pressure(self):
        for n in (1, 2, 100, 1000, 50_000):
            received = run_correct_ring(n)
            assert received == list(range(n)), f"failed at n={n}"

    def test_broken_ring_fails_reliably_not_just_once(self):
        """The deliberate delay in BrokenSpscRing::push widens the race
        window so this is a reliable failure, not a rare flake -- verify it
        fails across several independent runs."""
        n = 500
        failures = 0
        for _ in range(5):
            if run_broken_ring(n) != list(range(n)):
                failures += 1
        assert failures == 5


def _python_spin(iterations):
    total = 0
    for _ in range(iterations):
        total += 1
    return total


class TestGilReleaseDuringRealWork:
    def test_python_thread_and_released_c_thread_run_concurrently(self):
        """A background Python thread doing fixed, iteration-bounded work
        (not wall-clock-bounded -- a deadline-bounded loop would just exit
        on time whether or not it was starved, proving nothing) and a
        GIL-released C++ sleep should together take close to the slower one
        alone, not their sum -- proof the two actually overlap."""
        iterations = 5_000_000
        c_sleep_ms = 150

        t0 = time.monotonic()
        _python_spin(iterations)
        python_alone_s = time.monotonic() - t0

        t0 = time.monotonic()
        busy_wait_ms(c_sleep_ms, release_gil=True)
        c_alone_s = time.monotonic() - t0

        t = threading.Thread(target=_python_spin, args=(iterations,))
        t0 = time.monotonic()
        t.start()
        busy_wait_ms(c_sleep_ms, release_gil=True)
        t.join()
        concurrent_s = time.monotonic() - t0

        serialized_estimate_s = python_alone_s + c_alone_s

        # Concurrent execution should land much closer to the slower half
        # alone than to the sum of both -- generous margin to avoid flaking
        # on a loaded CI runner.
        assert concurrent_s < serialized_estimate_s * 0.75
