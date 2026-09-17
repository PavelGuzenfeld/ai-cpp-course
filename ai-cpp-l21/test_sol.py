"""Unit tests for L21 Part 2: speed-of-light derivation.

Pure arithmetic, so these assert exact values rather than ranges. Inputs are
chosen to be exactly representable in binary floating point wherever a
threshold comparison is under test, so `>=` versus `>` is observable.
"""

import sys

sys.path.insert(0, "/workspace/ai-cpp-l21")

import pytest  # noqa: E402

from sol import (  # noqa: E402
    AT_THE_FLOOR,
    HEADROOM,
    Machine,
    Node,
    Regime,
    budget_s,
    derive_node_sol,
    efficiency,
    fits_window,
    pipeline_sol_s,
    verdict,
)


def machine(**kw):
    base = dict(
        ops_per_s={"cpu": 1_000_000.0},
        bytes_per_s={"cpu": 1_000_000.0},
        tax_s_per_crossing=0.0,
        dispatch_s=0.0,
        completion_s=0.0,
    )
    base.update(kw)
    return Machine(**base)


class TestSolIsTheMaxOfTheFloorsNotTheirSum:
    def test_a_node_with_equal_compute_and_memory_floors_takes_one_of_them(self):
        # 1000 ops at 1e6 ops/s = 1 ms; 1000 bytes at 1e6 B/s = 1 ms.
        # Summing would give 2 ms and claim the stage cannot overlap its own
        # memory traffic with its own compute.
        s = derive_node_sol(Node("n", "cpu", ops=1000.0, byte_count=1000.0), machine())
        assert s.sol_s == 0.001

    def test_the_smaller_floor_does_not_contribute(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0, byte_count=1.0), machine())
        assert s.sol_s == 0.001


class TestTheRegimeNamesWhichFloorWon:
    def test_more_bytes_than_ops_is_memory_bound(self):
        s = derive_node_sol(Node("n", "cpu", ops=1.0, byte_count=1000.0), machine())
        assert s.regime is Regime.MEMORY

    def test_more_ops_than_bytes_is_compute_bound(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0, byte_count=1.0), machine())
        assert s.regime is Regime.COMPUTE

    def test_crossings_can_dominate_both(self):
        # 4 crossings at 1 ms each swamps 1 us of compute and 1 us of memory.
        s = derive_node_sol(
            Node("n", "cpu", ops=1.0, byte_count=1.0, crossings=4),
            machine(tax_s_per_crossing=0.001),
        )
        assert s.regime is Regime.TAX
        assert s.sol_s == 0.004


class TestDispatchAndCompletionAreSerialSoTheyAdd:
    def test_fixed_latencies_add_to_the_dominant_floor(self):
        s = derive_node_sol(
            Node("n", "cpu", ops=1000.0),
            machine(dispatch_s=0.25, completion_s=0.5),
        )
        assert s.sol_s == 0.001 + 0.25 + 0.5


class TestAnUnmeasuredUnitIsAnError:
    def test_missing_op_rate_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError):
            derive_node_sol(Node("n", "gpu", ops=1.0), machine())

    def test_missing_bandwidth_raises_rather_than_defaulting(self):
        m = machine(ops_per_s={"cpu": 1.0, "gpu": 1.0})
        with pytest.raises(KeyError):
            derive_node_sol(Node("n", "gpu", ops=1.0), m)


class TestPipelineSolSumsAlongThePath:
    def test_three_identical_nodes_cost_three_times_one(self):
        nodes = [Node(f"n{i}", "cpu", ops=1000.0) for i in range(3)]
        assert pipeline_sol_s(nodes, machine()) == 0.003


class TestFitsWindowReservesMarginBeforeComparing:
    def test_a_floor_exactly_at_the_reserved_limit_fits(self):
        # SOL 0.75 s against a 1 s window with 25% reserved: exactly 0.75.
        nodes = [Node("n", "cpu", ops=750_000.0)]
        assert fits_window(nodes, machine(), window_s=1.0, margin=0.25)

    def test_a_floor_just_past_the_reserved_limit_does_not_fit(self):
        nodes = [Node("n", "cpu", ops=750_001.0)]
        assert not fits_window(nodes, machine(), window_s=1.0, margin=0.25)

    def test_margin_outside_zero_to_one_is_rejected(self):
        with pytest.raises(ValueError):
            fits_window([], machine(), window_s=1.0, margin=1.0)


class TestBudgetIsAFractionOfSolNotOfTheDeadline:
    def test_a_seventy_percent_budget_is_sol_divided_by_that_fraction(self):
        s = derive_node_sol(Node("n", "cpu", ops=700.0), machine())
        assert budget_s(s, fraction=0.7) == pytest.approx(0.001, rel=1e-12)

    def test_a_fraction_of_zero_is_rejected(self):
        s = derive_node_sol(Node("n", "cpu", ops=1.0), machine())
        with pytest.raises(ValueError):
            budget_s(s, fraction=0.0)


class TestEfficiencyIsSolOverMeasured:
    def test_a_stage_running_at_its_floor_is_one(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0), machine())
        assert efficiency(s, 0.001) == 1.0

    def test_a_stage_taking_four_times_its_floor_is_a_quarter(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0), machine())
        assert efficiency(s, 0.004) == 0.25

    def test_zero_measured_time_is_rejected_rather_than_dividing(self):
        s = derive_node_sol(Node("n", "cpu", ops=1.0), machine())
        with pytest.raises(ValueError):
            efficiency(s, 0.0)


class TestVerdictThresholdsAreInclusiveWhereClaimed:
    def _sol_of(self, seconds):
        # ops chosen so the compute floor is exactly `seconds`.
        return derive_node_sol(Node("n", "cpu", ops=seconds * 1_000_000.0), machine())

    def test_exactly_at_the_floor_threshold_counts_as_at_the_floor(self):
        # measured = 1.0 makes efficiency exactly AT_THE_FLOOR, so `>=` vs `>`
        # is observable here rather than hidden by rounding.
        assert "at the floor" in verdict(self._sol_of(AT_THE_FLOOR), 1.0)

    def test_just_below_the_floor_threshold_is_not_at_the_floor(self):
        assert "at the floor" not in verdict(self._sol_of(0.5), 1.0)

    def test_exactly_at_the_headroom_threshold_reports_headroom(self):
        assert "headroom" in verdict(self._sol_of(HEADROOM), 1.0)

    def test_between_the_thresholds_is_neither_verdict(self):
        v = verdict(self._sol_of(0.5), 1.0)
        assert "at the floor" not in v and "real headroom" not in v


class TestBeatingYourOwnFloorIsAModelBugNotAWin:
    def test_measuring_faster_than_sol_reports_the_model_as_wrong(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0), machine())
        # Measured 0.5 ms against a 1 ms floor: physically impossible, so the
        # floor's inputs are wrong. Must not read as "at the floor".
        v = verdict(s, 0.0005)
        assert "IMPOSSIBLE" in v
        assert "at the floor" not in v

    def test_measuring_exactly_at_sol_is_not_flagged_as_impossible(self):
        s = derive_node_sol(Node("n", "cpu", ops=1000.0), machine())
        assert "IMPOSSIBLE" not in verdict(s, 0.001)


class TestATaxBoundStageGetsADifferentInstruction:
    def test_tax_dominated_headroom_says_cross_less_often(self):
        s = derive_node_sol(
            Node("n", "cpu", ops=1.0, byte_count=1.0, crossings=1),
            machine(tax_s_per_crossing=0.001),
        )
        assert s.regime is Regime.TAX
        assert "cross less often" in verdict(s, 0.1)

    def test_memory_dominated_headroom_does_not_say_cross_less_often(self):
        s = derive_node_sol(Node("n", "cpu", byte_count=1000.0), machine())
        assert "cross less often" not in verdict(s, 0.1)
