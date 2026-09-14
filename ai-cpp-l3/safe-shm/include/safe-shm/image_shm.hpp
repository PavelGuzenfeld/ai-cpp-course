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
    class DoubleBufferShm
    {
    public:
        explicit DoubleBufferShm(std::string const &shm_name,
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

        ~DoubleBufferShm()
        {
            runner_.reset();
            lock_.destroy();
        }

        DoubleBufferShm(DoubleBufferShm const &) = delete;
        DoubleBufferShm &operator=(DoubleBufferShm const &) = delete;
        DoubleBufferShm(DoubleBufferShm &&) = delete;
        DoubleBufferShm &operator=(DoubleBufferShm &&) = delete;

        void store(T const &data)
        {
            lock_.lock();
            *get_shm() = data;
            lock_.unlock();
        }

        Snapshot<T> load()
        {
            swapper_.stage(get_shm());
            runner_->trigger();
            return Snapshot<T>{&published_ptr_};
        }

        void wait()
        {
            runner_->wait();
        }

        void const *shm_addr() const noexcept { return shm_.get(); }
        void const *pre_allocated_addr() const noexcept { return pre_allocated_.get(); }

    private:
        T *get_shm() const noexcept
        {
            auto *p = static_cast<T *>(shm_.get());
            assert(p && "shared memory data is null");
            return p;
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
