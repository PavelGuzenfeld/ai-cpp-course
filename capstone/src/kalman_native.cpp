/**
 * kalman_native: a fixed 4-state / 2-measurement Kalman filter with every
 * matrix pre-allocated once in the constructor -- the baseline rebuilds
 * F, Q, H and R on every predict()/update() call instead.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nb = nanobind;

namespace
{
// Minimal row-major fixed-size matrix helpers -- only the shapes this
// filter needs (4x4, 4x2, 2x4, 2x2), no general linear algebra library.
void matmul(const double *a, int ar, int ac, const double *b, int bc, double *out)
{
    for (int i = 0; i < ar; ++i)
        for (int j = 0; j < bc; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < ac; ++k)
                sum += a[i * ac + k] * b[k * bc + j];
            out[i * bc + j] = sum;
        }
}

void transpose(const double *a, int ar, int ac, double *out)
{
    for (int i = 0; i < ar; ++i)
        for (int j = 0; j < ac; ++j)
            out[j * ar + i] = a[i * ac + j];
}

void invert2x2(const double *s, double *out)
{
    double det = s[0] * s[3] - s[1] * s[2];
    if (std::abs(det) < 1e-300)
        throw std::runtime_error("singular innovation covariance");
    double inv_det = 1.0 / det;
    out[0] = s[3] * inv_det;
    out[1] = -s[1] * inv_det;
    out[2] = -s[2] * inv_det;
    out[3] = s[0] * inv_det;
}
} // namespace

class FastKalmanFilter
{
public:
    FastKalmanFilter(double dt, double process_noise, double measurement_noise)
    {
        state_.fill(0.0);
        cov_.fill(0.0);
        cov_[0 * 4 + 0] = cov_[1 * 4 + 1] = cov_[2 * 4 + 2] = cov_[3 * 4 + 3] = 1.0;

        f_.fill(0.0);
        f_[0 * 4 + 0] = f_[1 * 4 + 1] = f_[2 * 4 + 2] = f_[3 * 4 + 3] = 1.0;
        f_[0 * 4 + 2] = dt;
        f_[1 * 4 + 3] = dt;

        q_.fill(0.0);
        q_[0 * 4 + 0] = q_[1 * 4 + 1] = q_[2 * 4 + 2] = q_[3 * 4 + 3] = process_noise;

        h_.fill(0.0);
        h_[0 * 4 + 0] = 1.0;
        h_[1 * 4 + 1] = 1.0;

        r_.fill(0.0);
        r_[0 * 2 + 0] = r_[1 * 2 + 1] = measurement_noise;
    }

    nb::ndarray<nb::numpy, double> state()
    {
        size_t shape[1] = {4};
        return nb::ndarray<nb::numpy, double>(state_.data(), 1, shape, nb::handle());
    }

    nb::ndarray<nb::numpy, double> covariance()
    {
        size_t shape[2] = {4, 4};
        return nb::ndarray<nb::numpy, double>(cov_.data(), 2, shape, nb::handle());
    }

    nb::ndarray<nb::numpy, double> predict()
    {
        std::array<double, 4> new_state{};
        matmul(f_.data(), 4, 4, state_.data(), 1, new_state.data());
        state_ = new_state;

        std::array<double, 16> f_cov{};
        matmul(f_.data(), 4, 4, cov_.data(), 4, f_cov.data());
        std::array<double, 16> f_t{};
        transpose(f_.data(), 4, 4, f_t.data());
        std::array<double, 16> new_cov{};
        matmul(f_cov.data(), 4, 4, f_t.data(), 4, new_cov.data());
        for (int i = 0; i < 16; ++i)
            cov_[i] = new_cov[i] + q_[i];

        return state();
    }

    nb::ndarray<nb::numpy, double> update(nb::ndarray<const double, nb::ndim<1>> measurement)
    {
        if (measurement.shape(0) != 2)
            throw std::invalid_argument("measurement must have 2 elements");

        std::array<double, 2> h_state{};
        matmul(h_.data(), 2, 4, state_.data(), 1, h_state.data());
        std::array<double, 2> y{measurement(0) - h_state[0], measurement(1) - h_state[1]};

        std::array<double, 8> h_cov{};
        matmul(h_.data(), 2, 4, cov_.data(), 4, h_cov.data());
        std::array<double, 8> h_t{};
        transpose(h_.data(), 2, 4, h_t.data());
        std::array<double, 4> s{};
        matmul(h_cov.data(), 2, 4, h_t.data(), 2, s.data());
        s[0] += r_[0];
        s[3] += r_[3];

        std::array<double, 4> s_inv{};
        invert2x2(s.data(), s_inv.data());

        std::array<double, 8> cov_ht{};
        matmul(cov_.data(), 4, 4, h_t.data(), 2, cov_ht.data());
        std::array<double, 8> k{};
        matmul(cov_ht.data(), 4, 2, s_inv.data(), 2, k.data());

        std::array<double, 4> ky{};
        matmul(k.data(), 4, 2, y.data(), 1, ky.data());
        for (int i = 0; i < 4; ++i)
            state_[i] += ky[i];

        std::array<double, 16> k_h{};
        matmul(k.data(), 4, 2, h_.data(), 4, k_h.data());
        std::array<double, 16> i_minus_kh{};
        for (int i = 0; i < 16; ++i)
            i_minus_kh[i] = ((i % 5 == 0) ? 1.0 : 0.0) - k_h[i];
        std::array<double, 16> new_cov{};
        matmul(i_minus_kh.data(), 4, 4, cov_.data(), 4, new_cov.data());
        cov_ = new_cov;

        return state();
    }

private:
    std::array<double, 4> state_{};
    std::array<double, 16> cov_{};
    std::array<double, 16> f_{};
    std::array<double, 16> q_{};
    std::array<double, 8> h_{};
    std::array<double, 4> r_{};
};

NB_MODULE(kalman_native, m)
{
    m.doc() = "Pre-allocated 4-state Kalman filter (capstone component 1)";

    nb::class_<FastKalmanFilter>(m, "FastKalmanFilter")
        .def(nb::init<double, double, double>(),
             nb::arg("dt") = 1.0, nb::arg("process_noise") = 0.01,
             nb::arg("measurement_noise") = 0.1)
        .def("predict", &FastKalmanFilter::predict)
        .def("update", &FastKalmanFilter::update, nb::arg("measurement"))
        .def_prop_ro("state", &FastKalmanFilter::state, nb::rv_policy::reference_internal)
        .def_prop_ro("covariance", &FastKalmanFilter::covariance, nb::rv_policy::reference_internal);
}
