#include "async_runner.hpp"
#include <atomic>
#include <condition_variable>
#include <fmt/core.h>
#include <mutex>
#include <string_view>
#include <thread>

namespace tasks
{
    struct AsyncRunner::Impl
    {
        std::atomic<bool> stop_flag_ = false;
        std::atomic<unsigned int> task_counter_ = 0;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::jthread swap_thread_;
    };

    AsyncRunner::AsyncRunner(callback_t consumer_fn, log_printer_t log_printer)
        : consumer_(std::move(consumer_fn)), log_printer_(std::move(log_printer)), pimpl_(std::make_unique<Impl>())
    {
        pimpl_->swap_thread_ = std::jthread([this]
                                            { swapLoop(); });
    }

    AsyncRunner::~AsyncRunner()
    {
        pimpl_->stop_flag_ = true;
        pimpl_->cv_.notify_all(); // wake up thread to let it exit
    }

    void AsyncRunner::trigger_once() noexcept
    {
        pimpl_->task_counter_++;  // enqueue a "task" (we just count calls here)
        pimpl_->cv_.notify_one(); // wake up the thread
    }

    void AsyncRunner::swapLoop() noexcept
    try
    {
        while (!pimpl_->stop_flag_)
        {
            std::unique_lock lock(pimpl_->mutex_);
            pimpl_->cv_.wait(lock, [this]
                             { return pimpl_->task_counter_ > 0 || pimpl_->stop_flag_; });

            if (pimpl_->stop_flag_)
            {
                break;
            }

            try
            {
                consumer_(); // call the consumer function
            }
            catch (std::exception const &e)
            {
                auto const msg = fmt::format("Exception caught in swapLoop callback: {}", e.what());
                log_printer_(e.what());
            }
            catch (...)
            {
                log_printer_("Unknown exception caught in swapLoop callback.");
            }
            
            pimpl_->task_counter_--;
        }
    }
    catch (std::exception const &e)
    {
        auto const msg = fmt::format("Exception caught in swapLoop: {}", e.what());
        log_printer_(e.what());
    }
    catch (...)
    {
        log_printer_("Unknown exception caught in swapLoop.");
    }

} // namespace tasks