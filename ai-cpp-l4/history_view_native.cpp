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
    void push(nb::ndarray<double> arr)
    {
        if (static_cast<size_t>(arr.size()) != row_size_)
        {
            throw std::invalid_argument(
                "expected row of size " + std::to_string(row_size_) +
                ", got " + std::to_string(arr.size()));
        }

        const double* src = arr.data();
        double* dst = storage_.data() + (head_ * row_size_);
        std::memcpy(dst, src, row_size_ * sizeof(double));

        head_ = (head_ + 1) % max_entries_;
        if (count_ < max_entries_)
        {
            count_++;
        }
    }

    /// Return a VIEW of the latest `n` entries as a 2D numpy array (n x row_size).
    /// This is zero-copy — the returned array shares memory with the ring buffer.
    ///
    /// When entries are contiguous (no wrap-around between the requested entries),
    /// a direct view is returned. When wrap-around occurs, we must copy into a
    /// temporary buffer so the result is contiguous (numpy requires it).
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

        // Calculate start index: head_ points to next-write, so the most
        // recent entry is at (head_ - 1), and n entries back is (head_ - n).
        size_t start;
        if (head_ >= n)
        {
            start = head_ - n;
        }
        else
        {
            start = max_entries_ - (n - head_);
        }

        size_t shape[] = {n, row_size_};
        int64_t strides[] = {
            static_cast<int64_t>(row_size_ * sizeof(double)),
            static_cast<int64_t>(sizeof(double))
        };

        // If contiguous (no wrap), return a direct view into storage_
        if (start + n <= max_entries_)
        {
            double* ptr = storage_.data() + (start * row_size_);
            return nb::ndarray<nb::numpy, double>(ptr, 2, shape, nb::handle(), strides);
        }

        // Wrap-around: must copy into a contiguous temp buffer
        auto* tmp = new double[n * row_size_];
        nb::capsule owner(tmp, [](void* p) noexcept { delete[] static_cast<double*>(p); });

        size_t first_chunk = max_entries_ - start;
        std::memcpy(tmp, storage_.data() + start * row_size_,
                    first_chunk * row_size_ * sizeof(double));
        std::memcpy(tmp + first_chunk * row_size_, storage_.data(),
                    (n - first_chunk) * row_size_ * sizeof(double));

        return nb::ndarray<nb::numpy, double>(tmp, 2, shape, owner, strides);
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
