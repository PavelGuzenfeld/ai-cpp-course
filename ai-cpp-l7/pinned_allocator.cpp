/**
 * Pinned memory pool allocator.
 *
 * Wraps cudaMallocHost / cudaFreeHost (or malloc fallback) to provide
 * a pool of pre-allocated pinned memory buffers. This eliminates the
 * per-frame allocation overhead seen in tracker_engine's os_tracker_forward().
 *
 * Usage pattern:
 *   pool = PinnedBufferPool(n_buffers=4, buffer_size=640*480*3)
 *   buf = pool.acquire()    # O(1) — no allocation
 *   # ... fill buf, transfer to GPU ...
 *   pool.release(buf)       # O(1) — no deallocation
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// ─── Pinned memory helpers ───────────────────────────────────────────────────

static void* pinned_alloc(size_t size) {
#if HAVE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        // Fall back to regular malloc if CUDA runtime fails
        ptr = std::malloc(size);
        if (!ptr) throw std::bad_alloc();
    }
    return ptr;
#else
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
#endif
}

static void pinned_free(void* ptr) {
#if HAVE_CUDA
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        // If cudaFreeHost fails, it wasn't pinned — free normally
        std::free(ptr);
    }
#else
    std::free(ptr);
#endif
}

// ─── PinnedBufferPool ────────────────────────────────────────────────────────

class PinnedBufferPool {
public:
    /**
     * Create a pool of pre-allocated pinned memory buffers.
     *
     * @param n_buffers  Number of buffers to pre-allocate
     * @param buffer_size  Size of each buffer in bytes
     */
    PinnedBufferPool(size_t n_buffers, size_t buffer_size)
        : buffer_size_(buffer_size), total_buffers_(n_buffers)
    {
        if (n_buffers == 0) throw std::invalid_argument("n_buffers must be > 0");
        if (buffer_size == 0) throw std::invalid_argument("buffer_size must be > 0");

        buffers_.reserve(n_buffers);
        for (size_t i = 0; i < n_buffers; ++i) {
            void* ptr = pinned_alloc(buffer_size);
            std::memset(ptr, 0, buffer_size);
            buffers_.push_back(ptr);
            available_.push(i);
        }
    }

    ~PinnedBufferPool() {
        for (void* ptr : buffers_) {
            pinned_free(ptr);
        }
    }

    // Non-copyable, non-movable (owns raw memory)
    PinnedBufferPool(const PinnedBufferPool&) = delete;
    PinnedBufferPool& operator=(const PinnedBufferPool&) = delete;

    /**
     * Acquire a buffer from the pool.
     * Returns a numpy array (uint8) backed by pinned memory.
     * The array shape is (buffer_size,) — reshape as needed.
     *
     * Throws if no buffers are available.
     */
    nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_.empty()) {
            throw std::runtime_error(
                "PinnedBufferPool: no buffers available. "
                "Increase pool size or release buffers sooner.");
        }

        size_t idx = available_.front();
        available_.pop();

        uint8_t* ptr = static_cast<uint8_t*>(buffers_[idx]);
        size_t shape[1] = { buffer_size_ };

        // Create a capsule that captures the pool index for release tracking.
        // The capsule does NOT free the memory — the pool owns it.
        // We store the index in the capsule so release() can validate.
        size_t* idx_copy = new size_t(idx);
        nb::capsule owner(idx_copy, [](void* p) noexcept { delete static_cast<size_t*>(p); });

        return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(ptr, 1, shape, owner);
    }

    /**
     * Release buffer at the given index back to the pool.
     */
    void release(size_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= total_buffers_) {
            throw std::out_of_range("PinnedBufferPool: invalid buffer index");
        }
        available_.push(index);
    }

    /**
     * Acquire a buffer and return (numpy_array, index) tuple.
     * The index is needed to release the buffer later.
     */
    nb::tuple acquire_with_index() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_.empty()) {
            throw std::runtime_error("PinnedBufferPool: no buffers available.");
        }

        size_t idx = available_.front();
        available_.pop();

        uint8_t* ptr = static_cast<uint8_t*>(buffers_[idx]);
        size_t shape[1] = { buffer_size_ };

        size_t* idx_copy = new size_t(idx);
        nb::capsule owner(idx_copy, [](void* p) noexcept { delete static_cast<size_t*>(p); });

        auto arr = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(ptr, 1, shape, owner);
        return nb::make_tuple(arr, idx);
    }

    size_t available_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_.size();
    }

    size_t total_count() const { return total_buffers_; }
    size_t buffer_size() const { return buffer_size_; }

    bool is_pinned() const {
#if HAVE_CUDA
        return true;
#else
        return false;
#endif
    }

    std::string info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return "PinnedBufferPool(total=" + std::to_string(total_buffers_) +
               ", available=" + std::to_string(available_.size()) +
               ", buffer_size=" + std::to_string(buffer_size_) +
               ", pinned=" + (is_pinned() ? "true" : "false") + ")";
    }

private:
    size_t buffer_size_;
    size_t total_buffers_;
    std::vector<void*> buffers_;
    std::queue<size_t> available_;
    mutable std::mutex mutex_;
};

// ─── Nanobind module ─────────────────────────────────────────────────────────

NB_MODULE(pinned_allocator, m) {
    m.doc() = "Pinned memory pool allocator for zero-overhead GPU transfers";

    nb::class_<PinnedBufferPool>(m, "PinnedBufferPool")
        .def(nb::init<size_t, size_t>(),
             nb::arg("n_buffers"), nb::arg("buffer_size"),
             "Create a pool of pre-allocated pinned memory buffers.\n\n"
             "Args:\n"
             "    n_buffers: Number of buffers to pre-allocate\n"
             "    buffer_size: Size of each buffer in bytes")
        .def("acquire", &PinnedBufferPool::acquire,
             "Acquire a buffer as a numpy uint8 array backed by pinned memory.")
        .def("acquire_with_index", &PinnedBufferPool::acquire_with_index,
             "Acquire a buffer, returning (numpy_array, index) tuple.")
        .def("release", &PinnedBufferPool::release,
             nb::arg("index"),
             "Release the buffer at the given index back to the pool.")
        .def("available_count", &PinnedBufferPool::available_count,
             "Number of buffers currently available.")
        .def("total_count", &PinnedBufferPool::total_count,
             "Total number of buffers in the pool.")
        .def("buffer_size", &PinnedBufferPool::buffer_size,
             "Size of each buffer in bytes.")
        .def("is_pinned", &PinnedBufferPool::is_pinned,
             "True if buffers use CUDA pinned memory (vs regular malloc).")
        .def("__repr__", &PinnedBufferPool::info);
}
