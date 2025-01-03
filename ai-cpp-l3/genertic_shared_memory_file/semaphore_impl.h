#pragma once
#include <expected>     // std::expected, std::unexpected
#include <fcntl.h>      // O_CREAT, O_RDWR
#include <fmt/format.h> // fmt::format
#include <semaphore.h>  // sem_open, sem_wait, sem_post, sem_close
#include <string>       // std::string

namespace flat_shm_impl
{
    struct Semaphore
    {
        std::string sem_name_;
        sem_t *sem_ = nullptr;
    };

    inline std::expected<Semaphore, std::string> create(std::string const &name, int initial_value = 0)
    {
        sem_t *sem = sem_open(name.c_str(), O_CREAT | O_EXCL, 0644, initial_value);
        if (sem == SEM_FAILED)
        {
            if (errno == EEXIST)
            {
                sem = sem_open(name.c_str(), O_RDWR, 0644, initial_value);
                if (sem == SEM_FAILED)
                {
                    return std::unexpected(fmt::format("sem_open failed: {} for semaphore: {}", strerror(errno), name));
                }
            }
            else
            {
                return std::unexpected(fmt::format("sem_open failed: {} for semaphore: {}", strerror(errno), name));
            }
        }
        return Semaphore{name, sem};
    }

    inline bool is_valid(Semaphore const &sem)
    {
        return sem.sem_ != nullptr;
    }

    inline bool wait(Semaphore &sem)
    {
        return sem_wait(sem.sem_) == 0;
    }

    inline void post(Semaphore &sem)
    {
        sem_post(sem.sem_);
    }

    void destroy(Semaphore &sem)
    {
        if (sem.sem_)
        {
            post(sem);
            sem_close(sem.sem_);
            sem_unlink(sem.sem_name_.c_str());
            sem.sem_ = nullptr;
        }
    }
    class Guard
    {
    public:
        // Acquire the semaphore upon construction
        explicit Guard(Semaphore &semaphore)
            : sem_(semaphore), locked_(false)
        {
            if (is_valid(sem_))
            {
                if (wait(sem_))
                {
                    locked_ = true;
                }
                else
                {
                    // You could throw an exception or handle error differently
                    // For minimal demonstration, just set locked_ to false.
                    locked_ = false;
                }
            }
        }

        Guard(const Guard &) = delete;
        Guard &operator=(const Guard &) = delete;
        Guard(Guard &&) = delete;
        Guard &operator=(Guard &&) = delete;

        ~Guard()
        {
            unlockIfNeeded();
        }

        bool isLocked() const { return locked_; }

    private:
        Semaphore &sem_;
        bool locked_;
        unsigned int count_;

        void unlockIfNeeded()
        {
            if (locked_)
            {
                post(sem_);
                locked_ = false;
            }
        }
    };
} // namespace flat_shm_impl