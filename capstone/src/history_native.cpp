/**
 * history_native: a fixed-capacity circular buffer of fixed-width rows.
 * push() never allocates (fixed backing array); latest(n) assembles one
 * fused (n, dim) array in a single C++ pass instead of the baseline's
 * per-element Python-level .copy() calls.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

class CircularBuffer
{
public:
    CircularBuffer(int capacity, int dim)
        : capacity_(capacity), dim_(dim), storage_(static_cast<std::size_t>(capacity) * dim, 0.0)
    {
        if (capacity <= 0 || dim <= 0)
            throw std::invalid_argument("capacity and dim must be > 0");
    }

    void push(nb::ndarray<const double, nb::ndim<1>> row)
    {
        if (static_cast<int>(row.shape(0)) != dim_)
            throw std::invalid_argument("row size does not match dim");
        int slot = write_index_ % capacity_;
        for (int i = 0; i < dim_; ++i)
            storage_[static_cast<std::size_t>(slot) * dim_ + i] = row(i);
        ++write_index_;
    }

    nb::ndarray<nb::numpy, double> latest(int n)
    {
        int size = std::min(write_index_, capacity_);
        if (n > size)
            n = size;

        auto *out = new double[static_cast<std::size_t>(n) * dim_];
        for (int i = 0; i < n; ++i)
        {
            // i=0 is the most recent entry, matching the baseline's order.
            int slot = (write_index_ - 1 - i + capacity_) % capacity_;
            for (int d = 0; d < dim_; ++d)
                out[static_cast<std::size_t>(i) * dim_ + d] = storage_[static_cast<std::size_t>(slot) * dim_ + d];
        }

        std::size_t shape[2] = {static_cast<std::size_t>(n), static_cast<std::size_t>(dim_)};
        nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<double *>(p); });
        return nb::ndarray<nb::numpy, double>(out, 2, shape, owner);
    }

    [[nodiscard]] int size() const { return std::min(write_index_, capacity_); }

private:
    int capacity_;
    int dim_;
    int write_index_{0};
    std::vector<double> storage_;
};

NB_MODULE(history_native, m)
{
    m.doc() = "Fixed-capacity circular buffer, no per-element allocation (capstone component 3)";

    nb::class_<CircularBuffer>(m, "CircularBuffer")
        .def(nb::init<int, int>(), nb::arg("capacity"), nb::arg("dim"))
        .def("push", &CircularBuffer::push, nb::arg("row"))
        .def("latest", &CircularBuffer::latest, nb::arg("n"))
        .def("__len__", &CircularBuffer::size);
}
