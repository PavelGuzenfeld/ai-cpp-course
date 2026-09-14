#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string_view>
#include <thread>

namespace safe_shm
{
    class SwapRunner
    {
    public:
        using task_fn = std::function<void()>;
        using log_fn = std::function<void(std::string_view)>;

        SwapRunner(task_fn task, log_fn log)
            : task_(std::move(task)),
              log_(std::move(log)),
              thread_([this](std::stop_token st) { run(st); })
        {
        }

        SwapRunner(SwapRunner const &) = delete;
        SwapRunner &operator=(SwapRunner const &) = delete;
        SwapRunner(SwapRunner &&) = delete;
        SwapRunner &operator=(SwapRunner &&) = delete;

        void trigger() noexcept
        {
            idle_.store(0, std::memory_order_relaxed);
            triggered_.store(1, std::memory_order_release);
            triggered_.notify_one();
        }

        void wait() noexcept
        {
            idle_.wait(0, std::memory_order_acquire);
        }

    private:
        void run(std::stop_token st) noexcept
        {
            // When stop is requested, wake the trigger wait
            std::stop_callback cb(st, [this]
                                  {
                triggered_.store(1, std::memory_order_release);
                triggered_.notify_one(); });

            while (!st.stop_requested())
            {
                triggered_.wait(0, std::memory_order_acquire);
                if (st.stop_requested())
                    break;
                triggered_.store(0, std::memory_order_relaxed);

                try
                {
                    task_();
                }
                catch (...)
                {
                    log_("SwapRunner: task threw exception\n");
                }

                idle_.store(1, std::memory_order_release);
                idle_.notify_all();
            }

            // Ensure any pending wait() unblocks on shutdown
            idle_.store(1, std::memory_order_release);
            idle_.notify_all();
        }

        task_fn task_;
        log_fn log_;
        std::atomic<uint32_t> triggered_{0};
        std::atomic<uint32_t> idle_{1};
        std::jthread thread_; // must be last — starts thread in constructor
    };
} // namespace safe_shm
