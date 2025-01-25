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
        ~AsyncRunner();

        void trigger_once() noexcept;

        // copy/move disabled, as before
        AsyncRunner(const AsyncRunner &) = delete;
        AsyncRunner &operator=(const AsyncRunner &) = delete;
        AsyncRunner(AsyncRunner &&other) = delete;
        AsyncRunner &operator=(AsyncRunner &&other) = delete;

    private:
        void swapLoop() noexcept;

        callback_t consumer_;
        log_printer_t log_printer_;

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace tasks