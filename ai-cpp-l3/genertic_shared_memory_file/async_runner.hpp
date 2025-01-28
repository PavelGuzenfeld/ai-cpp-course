#pragma once
#include <functional>
#include <memory>

namespace tasks
{
    class AsyncRunner
    {
    public:
        using callback_t = std::function<void()>;
        using log_printer_t = std::function<void(std::string_view)>;

        AsyncRunner(callback_t consumer_fn, log_printer_t log_printer);
        ~AsyncRunner() noexcept;

        void async_start() noexcept;
        void async_stop() noexcept;

        void trigger_once() noexcept;
        void wait_for_all_tasks() noexcept;

        AsyncRunner(const AsyncRunner &) = delete;
        AsyncRunner &operator=(const AsyncRunner &) = delete;
        AsyncRunner(AsyncRunner &&other) noexcept;
        AsyncRunner &operator=(AsyncRunner &&other) noexcept;

    private:
        void swapLoop() noexcept;
        
        callback_t consumer_;
        log_printer_t log_printer_;

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace tasks