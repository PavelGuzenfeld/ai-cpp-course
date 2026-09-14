"""
Integration test for Lesson 17: the falsifier run against real timers, not
synthetic data. Only asserts the asymptotic fact that isn't close enough to
flake on any reasonable box: at a large n, O(n log n) decisively beats
O(n^2).
"""
import sys

sys.path.insert(0, ".")

from falsifier import find_crossover, sweep  # noqa: E402
from falsifier_native import time_insertion_sort, time_std_sort  # noqa: E402


class TestRealMeasurementFindsTheAsymptoticWinner:
    def test_std_sort_decisively_wins_at_large_n(self):
        n = 8192
        std_ns = time_std_sort(n, trials=20)
        ins_ns = time_insertion_sort(n, trials=5)  # O(n^2) at n=8192 is slow; fewer trials

        # O(n log n) vs O(n^2) at n=8192: not a close call on any hardware.
        # 5x is a generous margin, not a tuned threshold.
        assert std_ns * 5 < ins_ns

    def test_full_sweep_produces_a_decision(self):
        sizes = [8, 8192]
        results = sweep(sizes=sizes, trials=20)
        crossover = find_crossover(results)

        assert [n for n, _, _ in results] == sizes
        assert all(std_ns > 0 and ins_ns > 0 for _, std_ns, ins_ns in results)
        # The probe must land on one of the tested sizes or explicitly
        # refuse to pick one -- not silently return something else.
        assert crossover is None or crossover in sizes
