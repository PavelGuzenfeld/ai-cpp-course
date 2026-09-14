"""
Tests for Lesson 17's falsifier.

The crossover-detection and verdict-rendering LOGIC is tested against
controlled, synthetic timing data (deterministic, no hardware noise). The
real timers are only asked for a loose, asymptotic fact -- see
test_integration_falsifier.py for the harder claim, tested against real
measurements.
"""
import sys

sys.path.insert(0, ".")

from falsifier import find_crossover, render_verdict  # noqa: E402


class TestCrossoverDetectionLogic:
    def test_no_crossover_when_std_sort_always_wins(self):
        results = [(4, 10.0, 20.0), (64, 10.0, 100.0), (1024, 10.0, 10000.0)]
        assert find_crossover(results) == 4

    def test_crossover_at_the_size_where_std_sort_first_wins(self):
        results = [(4, 50.0, 10.0), (64, 50.0, 40.0), (1024, 50.0, 5000.0)]
        assert find_crossover(results) == 1024

    def test_refuted_when_std_sort_never_wins(self):
        results = [(4, 50.0, 10.0), (64, 400.0, 100.0), (1024, 9000.0, 5000.0)]
        assert find_crossover(results) is None


class TestVerdictRendering:
    def test_falsified_verdict_names_the_crossover_and_the_narrower_claim(self):
        results = [(4, 50.0, 10.0), (1024, 50.0, 5000.0)]
        verdict = render_verdict(results, crossover=1024)
        assert "FALSIFIED" in verdict
        assert "n=1024" in verdict

    def test_confirmed_verdict_when_crossover_is_the_first_size(self):
        results = [(4, 10.0, 20.0)]
        verdict = render_verdict(results, crossover=4)
        assert "CONFIRMED" in verdict

    def test_refuted_verdict_when_no_crossover(self):
        results = [(4, 50.0, 10.0), (1024, 9000.0, 5000.0)]
        verdict = render_verdict(results, crossover=None)
        assert "REFUTED" in verdict

    def test_verdict_includes_every_measured_size(self):
        results = [(4, 1.0, 2.0), (64, 3.0, 4.0), (1024, 5.0, 6.0)]
        verdict = render_verdict(results, crossover=1024)
        for n, _, _ in results:
            assert f"| {n} |" in verdict
