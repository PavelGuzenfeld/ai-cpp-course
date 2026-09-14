#pragma once
#include "safe-shm/flat_type.hpp"
#include "shm/semaphore.hpp"
#include "shm/shm.hpp"
#include <functional>

namespace safe_shm
{
    template <FlatType T>
    class ProducerConsumer
    {
    public:
        explicit ProducerConsumer(std::string const &shm_name)
            : impl_(shm::path(shm_name), sizeof(T)),
              sem_read_(shm_name + "_read", 0),
              sem_write_(shm_name + "_write", 1)
        {
        }

        void produce(T const &data)
        {
            sem_write_.wait();
            get() = data;
            sem_read_.post();
        }

        void consume(std::function<void(T const &)> consumer)
        {
            sem_read_.wait();
            consumer(get());
            sem_write_.post();
        }

    private:
        T &get() noexcept
        {
            return *static_cast<T *>(impl_.get());
        }

        shm::Shm impl_;
        shm::Semaphore sem_read_;
        shm::Semaphore sem_write_;
    };
} // namespace safe_shm
