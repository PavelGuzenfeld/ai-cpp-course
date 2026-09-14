"""
Unit tests for Lesson 20: numerical robustness in stateful pipelines.

Tests:
  - NaiveFilter: one NaN permanently wedges the filter
  - CoastingFilter: the same NaN is discarded, state stays finite
  - naive_likelihood_sum underflows to exactly 0.0 on a heavy-tailed input
  - log_sum_exp stays finite on the same input
"""

import math
import sys

import pytest

sys.path.insert(0, ".")

from robustness_native import (  # noqa: E402
    CoastingFilter,
    NaiveFilter,
    log_sum_exp,
    naive_likelihood_sum,
    naive_normalize,
)


class TestNaNPoisoning:
    def test_naive_filter_wedges_on_one_nan(self):
        f = NaiveFilter(alpha=0.5)
        for x in [1.0, 2.0, 3.0]:
            f.update(x)
        assert math.isfinite(f.value)

        f.update(float("nan"))
        assert math.isnan(f.value)

        # Every subsequent update reads the poisoned state: nan + finite == nan
        for x in [4.0, 5.0, 6.0]:
            f.update(x)
            assert math.isnan(f.value)

    def test_coasting_filter_discards_the_nan(self):
        f = CoastingFilter(alpha=0.5)
        for x in [1.0, 2.0, 3.0]:
            f.update(x)
        value_before_poison = f.value

        f.update(float("nan"))
        assert f.value == value_before_poison  # coasted: state unchanged

        f.update(4.0)
        assert math.isfinite(f.value)

    def test_coasting_filter_also_discards_infinity(self):
        f = CoastingFilter(alpha=0.5)
        f.update(1.0)
        value_before = f.value

        f.update(float("inf"))
        assert f.value == value_before


class TestLogSumExpUnderflow:
    def test_naive_sum_underflows_to_exactly_zero(self):
        # exp(x) underflows to 0.0 in double precision below roughly -745.
        heavy_tailed_log_likelihoods = [-800.0, -820.0, -900.0]
        assert naive_likelihood_sum(heavy_tailed_log_likelihoods) == 0.0

    def test_naive_normalisation_produces_nan(self):
        # In C++, 0.0 / 0.0 is nan by IEEE 754, silently -- no exception.
        # (Python's `/` raises ZeroDivisionError instead; this exercises
        # the actual C++ division the lesson is about.)
        heavy_tailed_log_likelihoods = [-800.0, -820.0, -900.0]
        normalised = naive_normalize(heavy_tailed_log_likelihoods, 0)
        assert math.isnan(normalised)

    def test_log_sum_exp_stays_finite_on_the_same_input(self):
        heavy_tailed_log_likelihoods = [-800.0, -820.0, -900.0]
        result = log_sum_exp(heavy_tailed_log_likelihoods)
        assert math.isfinite(result)
        assert result > -800.001  # dominated by the largest term, barely nudged

    def test_log_sum_exp_matches_naive_when_no_underflow(self):
        log_likelihoods = [-1.0, -2.0, -3.0]
        naive = math.log(naive_likelihood_sum(log_likelihoods))
        assert log_sum_exp(log_likelihoods) == pytest.approx(naive)
