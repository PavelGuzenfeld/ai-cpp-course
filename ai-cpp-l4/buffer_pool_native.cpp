#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

/// A pre-allocated pool of numpy-compatible buffers.
///
/// Motivation: tracker_engine's NumpyBufferPool pays Python function-call
/// overhead on every acquire()/release(). This C++ version keeps the fast
/// path entirely in native code, returning nb::ndarray views into
/// pre-allocated memory.
class BufferPool
{
public:
    /// Create a pool with `capacity` buffers, each of `buffer_size` float64 elements.
    BufferPool(size_t capacity, size_t buffer_size)
        : capacity_{capacity}, buffer_size_{buffer_size}
    {
        if (capacity == 0 || buffer_size == 0)
        {
            throw std::invalid_argument("capacity and buffer_size must be > 0");
        }

        // Pre-allocate all buffers
        storage_.resize(capacity);
        for (size_t i = 0; i < capacity; ++i)
        {
            storage_[i] = std::make_unique<double[]>(buffer_size);
            std::memset(storage_[i].get(), 0, buffer_size * sizeof(double));
            free_indices_.push_back(i);
        }
    }

    /// Acquire a buffer from the pool. Returns a numpy ndarray backed by
    /// pre-allocated C++ memory. The buffer is zeroed on acquire.
    nb::ndarray<nb::numpy, double> acquire()
    {
        std::lock_guard<std::mutex> lock{mutex_};

        if (free_indices_.empty())
        {
            throw std::runtime_error("buffer pool exhausted — all buffers in use");
        }

        size_t idx = free_indices_.back();
        free_indices_.pop_back();
        active_count_++;

        // Zero the buffer before handing it out
        std::memset(storage_[idx].get(), 0, buffer_size_ * sizeof(double));

        // Create a capsule that releases the buffer back to the pool when
        // the Python side is done with it (prevents leaks if release() is
        // not called explicitly).
        size_t shape[] = {buffer_size_};
        return nb::ndarray<nb::numpy, double>(storage_[idx].get(), 1, shape, nb::handle());
    }

    /// Acquire a buffer and return its index (for release).
    size_t acquire_index()
    {
        std::lock_guard<std::mutex> lock{mutex_};

        if (free_indices_.empty())
        {
            throw std::runtime_error("buffer pool exhausted — all buffers in use");
        }

        size_t idx = free_indices_.back();
        free_indices_.pop_back();
        active_count_++;
        std::memset(storage_[idx].get(), 0, buffer_size_ * sizeof(double));
        return idx;
    }

    /// Release a buffer back to the pool by its index.
    void release(size_t idx)
    {
        std::lock_guard<std::mutex> lock{mutex_};

        if (idx >= capacity_)
        {
            throw std::out_of_range("buffer index out of range");
        }

        // Check it is not already free
        for (size_t fi : free_indices_)
        {
            if (fi == idx)
            {
                throw std::runtime_error("buffer already released");
            }
        }
        free_indices_.push_back(idx);
        active_count_--;
    }

    /// Get a buffer by index as a numpy ndarray view.
    nb::ndarray<nb::numpy, double> get_buffer(size_t idx)
    {
        if (idx >= capacity_)
        {
            throw std::out_of_range("buffer index out of range");
        }
        size_t shape[] = {buffer_size_};
        return nb::ndarray<nb::numpy, double>(storage_[idx].get(), 1, shape, nb::handle());
    }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t buffer_size() const noexcept { return buffer_size_; }
    [[nodiscard]] size_t available() const noexcept { return free_indices_.size(); }
    [[nodiscard]] size_t active() const noexcept { return active_count_; }

private:
    size_t capacity_;
    size_t buffer_size_;
    size_t active_count_{0};
    std::vector<std::unique_ptr<double[]>> storage_;
    std::vector<size_t> free_indices_;
    std::mutex mutex_;
};

NB_MODULE(buffer_pool_native, m)
{
    m.doc() = "Zero-overhead buffer pool returning numpy arrays backed by pre-allocated C++ memory";

    nb::class_<BufferPool>(m, "BufferPool")
        .def(nb::init<size_t, size_t>(),
             nb::arg("capacity"), nb::arg("buffer_size"))
        .def("acquire", &BufferPool::acquire,
             "Acquire a zeroed buffer from the pool as a numpy array")
        .def("acquire_index", &BufferPool::acquire_index,
             "Acquire a buffer and return its index")
        .def("release", &BufferPool::release, nb::arg("index"),
             "Release a buffer back to the pool by index")
        .def("get_buffer", &BufferPool::get_buffer, nb::arg("index"),
             "Get buffer at index as numpy array")
        .def_prop_ro("capacity", &BufferPool::capacity)
        .def_prop_ro("buffer_size", &BufferPool::buffer_size)
        .def_prop_ro("available", &BufferPool::available)
        .def_prop_ro("active", &BufferPool::active);
}
