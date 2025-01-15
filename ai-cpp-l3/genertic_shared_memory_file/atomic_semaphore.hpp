#include <atomic>
#include <thread>

constexpr std::size_t CACHE_LINE_SIZE = 64;

template <typename T>
class AtomicSemaphore
{
private:
    alignas(CACHE_LINE_SIZE) T data_;
    alignas(CACHE_LINE_SIZE) std::atomic<int> state_;           // +1 for readers, -1 for writer
    alignas(CACHE_LINE_SIZE) std::atomic<bool> writer_waiting_; // Indicates a writer is waiting

public:
    explicit AtomicSemaphore(T initial_data = {})
        : data_(std::move(initial_data)), state_(0), writer_waiting_(false) {}

    // Reader function
    [[nodiscard]] T const *read()
    {
        while (true)
        {
            int expected = state_.load(std::memory_order_acquire);

            if (expected >= 0 && !writer_waiting_.load(std::memory_order_acquire))
            {
                if (state_.compare_exchange_weak(expected, expected + 1, std::memory_order_acquire))
                {
                    break;
                }
            }
            std::this_thread::yield(); // Back off
        }

        auto const *ptr = &data_;
        state_.fetch_sub(1, std::memory_order_release);
        return ptr;
    }

    // Writer function
    void write(T const &new_data)
    {
        writer_waiting_.store(true, std::memory_order_release);

        while (true)
        {
            int expected = 0;
            if (state_.compare_exchange_weak(expected, -1, std::memory_order_acquire))
            {
                break;
            }
            std::this_thread::yield(); // Back off
        }

        data_ = new_data;
        writer_waiting_.store(false, std::memory_order_release);
        state_.store(0, std::memory_order_release);
    }
};
