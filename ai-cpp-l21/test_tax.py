"""Unit tests for L21 step 1.8: the tax table.

These assert the *orderings* the lesson teaches, not absolute nanosecond
values -- absolute values are per-machine and are the student's to measure.
Each margin below is deliberately loose because the quantity is a timing on a
shared runner; the ordering is what the lesson claims, and an ordering that
inverts is a real finding rather than jitter.
"""

import sys

sys.path.insert(0, "/workspace/install/lib/python3.10/site-packages")

import pytest  # noqa: E402

tax_bench = pytest.importorskip(
    "tax_bench", reason="compiled module not built; colcon build --packages-select nanobind-l21"
)

sys.path.insert(0, "/workspace/ai-cpp-l21")
from measure_tax import ffi_crossing_ns, python_loop_floor_ns  # noqa: E402


class TestEveryTaxIsPositiveAndFinite:
    """A crossing that measures as free means the loop was optimised away."""

    @pytest.mark.parametrize(
        "name",
        [
            "syscall_floor_ns",
            "clock_gettime_vdso_ns",
            "clock_gettime_syscall_ns",
            "malloc_small_ns",
            "mutex_uncontended_ns",
        ],
    )
    def test_crossing_costs_more_than_zero(self, name):
        value = getattr(tax_bench, name)(20000)
        assert value > 0.0

    def test_pipe_round_trip_did_not_error(self):
        # roundtrip_ns returns -1.0 on a failed read/write rather than raising.
        assert tax_bench.pipe_roundtrip_ns(2000) > 0.0

    def test_socketpair_round_trip_did_not_error(self):
        assert tax_bench.socketpair_roundtrip_ns(2000) > 0.0


class TestTheVdsoIsWhyTimersAreNotSyscalls:
    def test_vdso_clock_read_is_cheaper_than_the_forced_syscall(self):
        # Measured 11x on x86-64 and 7x on an Orin NX. Asserting only 2x:
        # the claim is that the vDSO removes the kernel transition, and a
        # machine where it buys less than 2x would be genuinely surprising.
        vdso = tax_bench.clock_gettime_vdso_ns(100000)
        forced = tax_bench.clock_gettime_syscall_ns(100000)
        assert vdso * 2 < forced


class TestFaultingPagesInIsTheHiddenCostOfAllocation:
    def test_allocating_and_touching_a_megabyte_costs_more_than_a_small_malloc(self):
        small = tax_bench.malloc_small_ns(100000)
        faulted = tax_bench.malloc_page_faulted_ns(500, 1 << 20)
        assert faulted > small * 10


class TestCrossingLessOftenIsTheWholePoint:
    def test_one_call_per_item_costs_far_more_per_item_than_one_call_per_batch(self):
        # The lesson's central claim for step 1.8. Measured 55x on x86-64 and
        # 83x on an Orin NX; asserting 10x leaves room for a slow shared
        # runner without letting a genuine regression through.
        per_call, per_batched = ffi_crossing_ns(items=50_000)
        assert per_call > per_batched * 10

    def test_the_crossing_costs_more_than_the_python_loop_that_drives_it(self):
        # If this inverts, the "crossing cost" being reported is really the
        # interpreter's own per-iteration cost and the row is meaningless.
        per_call, per_batched = ffi_crossing_ns(items=50_000)
        crossing = per_call - per_batched
        assert crossing > python_loop_floor_ns(items=50_000) * 0.25
