#include <atomic>
#include <thread> // For std::this_thread::yield

// Cache line size is typically 64 bytes
constexpr std::size_t CACHE_LINE_SIZE = 64;

template <typename T>
class AtomicSemaphore
{
private:
    alignas(CACHE_LINE_SIZE) T data_;                 // Prevent false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<int> state_; // Separate cache line

public:
    explicit AtomicSemaphore(T initial_data = {})
        : data_(std::move(initial_data)), state_(0) {}

    // Reader function
    [[nodiscard]] T const *read()
    {
        int backoff = 1;

        while (true)
        {
            int expected = state_.load(std::memory_order_acquire);

            // If no writer is active, increment the reader count
            if (expected >= 0 && state_.compare_exchange_weak(expected, expected + 1, std::memory_order_acquire))
            {
                break;
            }

            // Use exponential backoff with `yield()`
            for (int i = 0; i < backoff; ++i)
            {
                std::this_thread::yield();
            }

            // Cap the backoff to avoid excessive yielding
            backoff = std::min(backoff * 2, 1024);
        }

        // Access the shared data
        auto const *ptr = &data_;

        // Decrement the reader count
        state_.fetch_sub(1, std::memory_order_release);
        return ptr;
    }

    // Writer function
    void write(T const &new_data)
    {
        int backoff = 1;

        while (true)
        {
            int expected = 0;

            // Try to acquire the writer lock
            if (state_.compare_exchange_weak(expected, -1, std::memory_order_acquire))
            {
                break;
            }

            // Use exponential backoff with `yield()`
            for (int i = 0; i < backoff; ++i)
            {
                std::this_thread::yield();
            }

            // Cap the backoff to avoid excessive yielding
            backoff = std::min(backoff * 2, 1024);
        }

        // Perform the write operation
        data_ = new_data;

        // Release the writer lock
        state_.store(0, std::memory_order_release);
    }
};
