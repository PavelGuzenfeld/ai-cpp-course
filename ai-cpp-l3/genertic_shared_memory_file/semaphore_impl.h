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
        static std::expected<Semaphore, std::string> create(std::string const &name, int initial_value = 0)
        {
            auto sem = sem_open(name.c_str(), O_CREAT | O_EXCL, 0644, initial_value);
            if (sem == SEM_FAILED)
            {
                return std::unexpected(fmt::format("sem_open failed: {} for semaphore: {}", strerror(errno), name));
            }
            return Semaphore{sem, name};
        }

        [[nodiscard]] inline explicit operator bool() const noexcept
        {
            return sem_ != nullptr;
        }

        inline bool wait() const noexcept
        {
            return sem_wait(sem_);
        }

        inline void post() const noexcept
        {
            sem_post(sem_);
        }

        ~Semaphore() noexcept
        {
            if (sem_)
            {
                sem_close(sem_);
                sem_unlink(sem_name_.c_str());
                sem_ = nullptr;
            }
        }

    private:
        Semaphore(sem_t *sem, std::string const &name)
            : sem_(sem), sem_name_(name)
        {
        }
        sem_t *sem_ = nullptr;
        std::string sem_name_;
    };

    class Guard
    {
    public:
        // Acquire the semaphore upon construction
        explicit Guard(Semaphore  &semaphore)
            : sem_(semaphore), locked_(false)
        {
            if (sem_)
            {
                if (sem_.wait() == 0)
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
        Semaphore & sem_;
        bool locked_;

        void unlockIfNeeded()
        {
            if (locked_)
            {
                sem_.post();
                locked_ = false;
            }
        }
    };
} // namespace flat_shm_impl