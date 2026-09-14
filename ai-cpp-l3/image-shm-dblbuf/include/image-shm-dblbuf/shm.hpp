#pragma once
#include "double-buffer-swapper/swapper.hpp"
#include "flat-type/flat.hpp"
#include "shm/semaphore.hpp"
#include "shm/shm.hpp"
#include "single-task-runner/runner.hpp"
#include <atomic>
#include <cassert>
#include <fmt/core.h>

namespace image_shm
{
    inline void default_logger(std::string_view msg) noexcept
    {
        fmt::print("{}", msg);
    }

    template <FlatType T>
    class Snapshot
    {
    public:
        explicit Snapshot(std::atomic<T *> *ptr) noexcept : ptr_(ptr) {}

        T const &operator*() const noexcept
        {
            auto *p = ptr_->load(std::memory_order_acquire);
            assert(p && "snapshot data is null");
            return *p;
        }

        T const *operator->() const noexcept
        {
            auto *p = ptr_->load(std::memory_order_acquire);
            assert(p && "snapshot data is null");
            return p;
        }

        T *get() const noexcept
        {
            return ptr_->load(std::memory_order_acquire);
        }

    private:
        std::atomic<T *> *ptr_;
    };

    template <FlatType T>
    class DoubleBufferShm
    {
    public:
        explicit DoubleBufferShm(std::string const &shm_name,
                                 void (*log)(std::string_view) = default_logger)
            : shm_(shm::path(shm_name), sizeof(T)),
              sem_(shm_name + "_sem", 1),
              pre_allocated_(std::make_unique<T>()),
              swapper_ptr_(nullptr),
              published_ptr_(nullptr)
        {
            swapper_ = std::make_unique<DoubleBufferSwapper<T>>(&swapper_ptr_, pre_allocated_.get());
            runner_ = std::make_unique<run::SingleTaskRunner>(
                [this]
                {
                    sem_.wait();
                    swapper_->swap();
                    published_ptr_.store(swapper_ptr_, std::memory_order_release);
                    sem_.post();
                },
                [log](std::string_view msg)
                { log(msg); });
            runner_->async_start();
            swapper_->set_active(get_shm());
            published_ptr_.store(swapper_ptr_, std::memory_order_release);
        }

        ~DoubleBufferShm()
        {
            runner_->async_stop();
            sem_.destroy();
        }

        DoubleBufferShm(DoubleBufferShm const &) = delete;
        DoubleBufferShm &operator=(DoubleBufferShm const &) = delete;
        DoubleBufferShm(DoubleBufferShm &&) = delete;
        DoubleBufferShm &operator=(DoubleBufferShm &&) = delete;

        void store(T const &data)
        {
            sem_.wait();
            *get_shm() = data;
            sem_.post();
        }

        Snapshot<T> load()
        {
            swapper_->stage(get_shm());
            runner_->trigger_once();
            return Snapshot<T>{&published_ptr_};
        }

        void wait()
        {
            runner_->wait_for_all_tasks();
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
        shm::Semaphore sem_;
        std::unique_ptr<T> pre_allocated_;
        std::unique_ptr<DoubleBufferSwapper<T>> swapper_;
        std::unique_ptr<run::SingleTaskRunner> runner_;
        T *swapper_ptr_;
        std::atomic<T *> published_ptr_;
    };
} // namespace image_shm
