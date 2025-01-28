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
        std::atomic<bool> stop_flag_ = true;
        std::atomic<unsigned int> task_counter_ = 0;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::jthread swap_thread_;
    };

    AsyncRunner::AsyncRunner(callback_t consumer_fn, log_printer_t log_printer)
        : consumer_(std::move(consumer_fn)), log_printer_(std::move(log_printer)), pimpl_(std::make_unique<Impl>())
    {
    }

    AsyncRunner::AsyncRunner(AsyncRunner &&other) noexcept
        : consumer_(std::move(other.consumer_)),
          log_printer_(std::move(other.log_printer_)),
          pimpl_(std::make_unique<Impl>())
    {
        if (other.pimpl_)
        {
            other.async_stop();
            pimpl_->task_counter_.store(other.pimpl_->task_counter_);
            pimpl_->stop_flag_.store(other.pimpl_->stop_flag_);
            // ensure the moved-from object's unique_ptr is reset properly
            other.pimpl_ = nullptr;
            // restart the thread in the new instance
            if (pimpl_ && pimpl_->stop_flag_ == true)
            {
                async_start();
            }
        }
    }
    
    AsyncRunner &AsyncRunner::operator=(AsyncRunner &&other) noexcept
    {
        if (this != &other)
        {
            async_stop();
            other.async_stop(); // stop the current thread if active
            consumer_ = std::move(other.consumer_);
            log_printer_ = std::move(other.log_printer_);
            pimpl_ = std::move(other.pimpl_);
            other.pimpl_ = nullptr;
            // restart the thread in the new instance
            if (pimpl_)
            {
                async_start();
            }
        }

        return *this;
    }

    AsyncRunner::~AsyncRunner() noexcept
    {
        wait_for_all_tasks();
        async_stop();
    }

    void AsyncRunner::async_start() noexcept
    {
        if (!pimpl_)
        {
            log_printer_("async_start called on a moved-from object.");
            return;
        }
        if (pimpl_->stop_flag_ == false)
        {
            return;
        }

        pimpl_->stop_flag_ = false;
        pimpl_->swap_thread_ = std::jthread([this]
                                            { swapLoop(); });
    }

    void AsyncRunner::async_stop() noexcept
    {
        if (!pimpl_)
        {
            return;
        }
        if (pimpl_->stop_flag_ == true)
        {
            return;
        }
        pimpl_->stop_flag_ = true;
        pimpl_->cv_.notify_all(); // wake up thread to let it exit
                                  // join the thread only after processing all pending tasks
        if (pimpl_->swap_thread_.joinable())
        {
            pimpl_->swap_thread_.join();
        }
    }

    void AsyncRunner::trigger_once() noexcept
    {
        if (!pimpl_)
        {
            log_printer_("trigger_once called on a moved-from object.");
            return;
        }
        if (pimpl_->stop_flag_ == true)
        {
            return;
        }
        pimpl_->task_counter_++;
        pimpl_->cv_.notify_one();
    }

    void AsyncRunner::swapLoop() noexcept
    try
    {
        if (!pimpl_)
        {
            return; // gracefully exit if the object is in a moved-from state
        }
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
            catch (const std::exception &e)
            {
                log_printer_(e.what());
            }
            catch (...)
            {
                log_printer_("Unknown exception caught in swapLoop.");
            }

            pimpl_->task_counter_--;

            // notify task completion
            pimpl_->cv_.notify_all();
        }
    }
    catch (const std::exception &e)
    {
        log_printer_(e.what());
    }
    catch (...)
    {
        log_printer_("Unknown exception caught in swapLoop.");
    }

    void AsyncRunner::wait_for_all_tasks() noexcept
    {
        if (!pimpl_)
        {
            return;
        }
        std::unique_lock lock(pimpl_->mutex_);
        pimpl_->cv_.wait(lock, [this]
                         { return pimpl_->task_counter_ == 0; });
    }

} // namespace tasks