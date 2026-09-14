#pragma once
#include "safe-shm/flat_type.hpp"
#include "safe-shm/shm_lock.hpp"
#include "shm/shm.hpp"

namespace safe_shm
{
    template <FlatType T>
    class Storage
    {
    public:
        explicit Storage(std::string const &shm_name)
            : shm_(shm::path(shm_name), sizeof(T)),
              lock_(shm_name + "_lock")
        {
        }

        void store(T const &data)
        {
            lock_.lock();
            *static_cast<T *>(shm_.get()) = data;
            lock_.unlock();
        }

    private:
        shm::Shm shm_;
        ShmLock lock_;
    };
} // namespace safe_shm
