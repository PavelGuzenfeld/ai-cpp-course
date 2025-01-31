#pragma once
#include <functional>
#include <memory>

namespace tasks
{
    class SingleTaskRunner
    {
    public:
        using callback_t = std::function<void()>;
        using log_printer_t = std::function<void(std::string_view)>;

        SingleTaskRunner(callback_t consumer_fn, log_printer_t log_printer);
        ~SingleTaskRunner() noexcept;

        void async_start() noexcept;
        void async_stop() noexcept;

        void trigger_once() noexcept;
        void wait_for_all_tasks() noexcept;

        SingleTaskRunner(const SingleTaskRunner &) = delete;
        SingleTaskRunner &operator=(const SingleTaskRunner &) = delete;
        SingleTaskRunner(SingleTaskRunner &&other) noexcept;
        SingleTaskRunner &operator=(SingleTaskRunner &&other) noexcept;

    private:
        void swapLoop() noexcept;
        
        callback_t consumer_;
        log_printer_t log_printer_;

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace tasks