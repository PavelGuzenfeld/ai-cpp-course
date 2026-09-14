#pragma once
#include "safe-shm/double_buffer_swapper.hpp"
#include "safe-shm/flat_type.hpp"
#include "safe-shm/logger.hpp"
#include "safe-shm/shm_lock.hpp"
#include "safe-shm/snapshot.hpp"
#include "safe-shm/swap_runner.hpp"
#include "shm/shm.hpp"
#include <atomic>
#include <cassert>
#include <memory>

namespace safe_shm
{
    template <FlatType T>
    class DblBufLoader
    {
    public:
        explicit DblBufLoader(std::string const &shm_name,
                              void (*log)(std::string_view) = default_logger)
            : shm_(shm::path(shm_name), sizeof(T)),
              lock_(shm_name + "_lock"),
              pre_allocated_(std::make_unique<T>()),
              swapper_ptr_(nullptr),
              published_ptr_(nullptr),
              swapper_(&swapper_ptr_, pre_allocated_.get()),
              runner_(std::make_unique<SwapRunner>(
                  [this]
                  {
                      lock_.lock();
                      swapper_.swap();
                      published_ptr_.store(swapper_ptr_, std::memory_order_release);
                      lock_.unlock();
                  },
                  [log](std::string_view msg)
                  { log(msg); }))
        {
            swapper_.set_active(get_shm());
            published_ptr_.store(swapper_ptr_, std::memory_order_release);
        }

        ~DblBufLoader()
        {
            runner_.reset();
            lock_.destroy();
        }

        DblBufLoader(DblBufLoader const &) = delete;
        DblBufLoader &operator=(DblBufLoader const &) = delete;
        DblBufLoader(DblBufLoader &&) = delete;
        DblBufLoader &operator=(DblBufLoader &&) = delete;

        Snapshot<T> load()
        {
            auto *shm = get_shm();
            assert(shm && "shared memory data is null");
            swapper_.stage(shm);
            runner_->trigger();
            return Snapshot<T>{&published_ptr_};
        }

        void wait()
        {
            runner_->wait();
        }

    private:
        T *get_shm() const noexcept
        {
            return static_cast<T *>(shm_.get());
        }

        shm::Shm shm_;
        ShmLock lock_;
        std::unique_ptr<T> pre_allocated_;
        T *swapper_ptr_;
        std::atomic<T *> published_ptr_;
        DoubleBufferSwapper<T> swapper_;
        std::unique_ptr<SwapRunner> runner_;
    };
} // namespace safe_shm
