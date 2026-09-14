#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace nb = nanobind;

// ===========================================================================
// A recursive filter with persistent state. `NaiveFilter::update` writes
// whatever it is given into `mean_` -- including a NaN, which then poisons
// every subsequent call, because `x + nan == nan` for any finite x.
// ===========================================================================
class NaiveFilter
{
public:
    explicit NaiveFilter(double alpha) : alpha_(alpha) {}

    void update(double x) noexcept
    {
        mean_ = alpha_ * x + (1.0 - alpha_) * mean_;
    }

    [[nodiscard]] double value() const noexcept { return mean_; }

private:
    double alpha_;
    double mean_ = 0.0;
};

// The fix: when the input is not finite, coast -- hold the previous state
// instead of writing the bad value into it. Detecting the NaN is not
// enough; the question is what you write.
class CoastingFilter
{
public:
    explicit CoastingFilter(double alpha) : alpha_(alpha) {}

    void update(double x) noexcept
    {
        if (!std::isfinite(x))
        {
            return; // coast: previous state is preserved
        }
        mean_ = alpha_ * x + (1.0 - alpha_) * mean_;
    }

    [[nodiscard]] double value() const noexcept { return mean_; }

private:
    double alpha_;
    double mean_ = 0.0;
};

// ===========================================================================
// Mode-probability normalisation, naive vs. log-space.
//
// exp() of a sufficiently negative log-likelihood underflows to exactly
// 0.0 in double precision (below roughly -745). If every mode's
// log-likelihood is that negative -- a real outcome for a heavy-tailed
// outlier -- the naive sum is 0.0 and the caller's normalisation divides
// by zero. Log-space with max-subtraction never forms that intermediate.
// ===========================================================================
[[nodiscard]] double naive_likelihood_sum(std::vector<double> const &log_likelihoods)
{
    double sum = 0.0;
    for (double ll : log_likelihoods)
    {
        sum += std::exp(ll);
    }
    return sum;
}

// The naive normalisation itself: exp(x) / sum(exp(*)). When every term
// underflows, this is 0.0 / 0.0, which IEEE 754 defines as nan -- silently,
// with no exception. (Python's `/` operator raises ZeroDivisionError
// instead; this function exists so the test observes the C++ behaviour the
// lesson is actually about.)
[[nodiscard]] double naive_normalize(std::vector<double> const &log_likelihoods, std::size_t index)
{
    double const total = naive_likelihood_sum(log_likelihoods);
    return std::exp(log_likelihoods.at(index)) / total;
}

[[nodiscard]] double log_sum_exp(std::vector<double> const &log_likelihoods)
{
    double const max_ll = *std::max_element(log_likelihoods.begin(), log_likelihoods.end());
    double sum = 0.0;
    for (double ll : log_likelihoods)
    {
        sum += std::exp(ll - max_ll);
    }
    return max_ll + std::log(sum);
}

NB_MODULE(robustness_native, m)
{
    nb::class_<NaiveFilter>(m, "NaiveFilter")
        .def(nb::init<double>(), nb::arg("alpha"))
        .def("update", &NaiveFilter::update, nb::arg("x"))
        .def_prop_ro("value", &NaiveFilter::value);

    nb::class_<CoastingFilter>(m, "CoastingFilter")
        .def(nb::init<double>(), nb::arg("alpha"))
        .def("update", &CoastingFilter::update, nb::arg("x"))
        .def_prop_ro("value", &CoastingFilter::value);

    m.def("naive_likelihood_sum", &naive_likelihood_sum, nb::arg("log_likelihoods"));
    m.def("naive_normalize", &naive_normalize, nb::arg("log_likelihoods"), nb::arg("index"));
    m.def("log_sum_exp", &log_sum_exp, nb::arg("log_likelihoods"));
}
