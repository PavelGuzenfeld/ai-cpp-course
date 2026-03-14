#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

/// A circular buffer that stores rows of double-precision data and returns
/// views (not copies) via nanobind::ndarray.
///
/// Motivation: tracker_engine's HistoryNP.latest() does `.copy()` on every
/// call, allocating and copying memory each time. This C++ version returns
/// an ndarray view directly into the ring buffer's contiguous storage,
/// eliminating all allocation and copy overhead.
///
/// The buffer stores `max_entries` rows, each of `row_size` doubles.
/// Data is stored contiguously so that when the buffer has not wrapped,
/// latest(n) can return a single view.
class HistoryView
{
public:
    /// Create a circular buffer for `max_entries` rows of `row_size` doubles each.
    HistoryView(size_t max_entries, size_t row_size)
        : max_entries_{max_entries}, row_size_{row_size}
    {
        if (max_entries == 0 || row_size == 0)
        {
            throw std::invalid_argument("max_entries and row_size must be > 0");
        }
        storage_.resize(max_entries * row_size, 0.0);
    }

    /// Push a row of data into the buffer.
    void push(nb::ndarray<nb::numpy, double, nb::ndim<1>> arr)
    {
        if (static_cast<size_t>(arr.shape(0)) != row_size_)
        {
            throw std::invalid_argument(
                "expected row of size " + std::to_string(row_size_) +
                ", got " + std::to_string(arr.shape(0)));
        }

        double* dst = storage_.data() + (head_ * row_size_);
        const double* src = arr.data();
        int64_t stride = arr.stride(0);  // element stride (not bytes)
        for (size_t i = 0; i < row_size_; ++i)
        {
            dst[i] = src[i * stride];
        }

        head_ = (head_ + 1) % max_entries_;
        if (count_ < max_entries_)
        {
            count_++;
        }
    }

    /// Return the latest `n` entries as a 2D numpy array (n x row_size).
    ///
    /// Returns a contiguous copy of the requested data. For zero-copy access
    /// to the underlying storage, use the Python-level HistoryView wrapper.
    nb::ndarray<nb::numpy, double> latest(size_t n)
    {
        if (n == 0)
        {
            throw std::invalid_argument("n must be > 0");
        }
        if (n > count_)
        {
            n = count_;
        }
        if (count_ == 0)
        {
            throw std::runtime_error("buffer is empty");
        }

        size_t total = n * row_size_;
        auto* out = new double[total];
        nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });

        // Copy n rows from the circular buffer into contiguous output
        for (size_t i = 0; i < n; ++i)
        {
            // Entry index: (head_ - n + i) mod max_entries_
            size_t entry_idx;
            if (head_ >= n)
                entry_idx = head_ - n + i;
            else
                entry_idx = (max_entries_ + head_ - n + i) % max_entries_;

            const double* src = storage_.data() + entry_idx * row_size_;
            double* dst = out + i * row_size_;
            std::memcpy(dst, src, row_size_ * sizeof(double));
        }

        size_t shape[] = {n, row_size_};
        return nb::ndarray<nb::numpy, double>(out, 2, shape, owner);
    }

    [[nodiscard]] size_t max_entries() const noexcept { return max_entries_; }
    [[nodiscard]] size_t row_size() const noexcept { return row_size_; }
    [[nodiscard]] size_t count() const noexcept { return count_; }

    /// Direct access to internal storage pointer (for testing view semantics).
    [[nodiscard]] uintptr_t data_ptr() const noexcept
    {
        return reinterpret_cast<uintptr_t>(storage_.data());
    }

private:
    size_t max_entries_;
    size_t row_size_;
    size_t head_{0};
    size_t count_{0};
    std::vector<double> storage_;
};

NB_MODULE(history_view_native, m)
{
    m.doc() = "Zero-copy circular buffer returning numpy array views into C++ memory";

    nb::class_<HistoryView>(m, "HistoryView")
        .def(nb::init<size_t, size_t>(),
             nb::arg("max_entries"), nb::arg("row_size"))
        .def("push", &HistoryView::push, nb::arg("row"),
             "Push a row of data into the circular buffer")
        .def("latest", &HistoryView::latest, nb::arg("n") = 1,
             "Return a view of the latest n entries (zero-copy when contiguous)")
        .def_prop_ro("max_entries", &HistoryView::max_entries)
        .def_prop_ro("row_size", &HistoryView::row_size)
        .def_prop_ro("count", &HistoryView::count)
        .def_prop_ro("data_ptr", &HistoryView::data_ptr);
}
