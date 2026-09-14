#pragma once
#include "shm/shm.hpp"
#include "shm/semaphore.hpp"
#include <functional>

namespace flat_shm
{
    template <typename T>
    struct FlatShmProducerConsumer
    {
        constexpr FlatShmProducerConsumer(std::string const &shm_name)
            : impl_(shm_name, sizeof(T)),
              sem_read_(shm_name + "_read", 0),
              sem_write_(shm_name + "_write", 1)
        {
        }

        inline void produce(T const &data)
        {
            sem_write_.wait();
            get() = data;
            sem_read_.post();
        }

        inline T const &consume_unsafe()
        {
            return get();
        }

        void consume(std::function<void(T const &)> consumer) noexcept
        {
            sem_read_.wait();
            consumer(get());
            sem_write_.post();
        }

    private:
        shm::Shm impl_;
        shm::Semaphore sem_read_;
        shm::Semaphore sem_write_;

        inline T &get() noexcept
        {
            return *static_cast<T *>(impl_.get());
        }

    };
} // namespace flat_shm