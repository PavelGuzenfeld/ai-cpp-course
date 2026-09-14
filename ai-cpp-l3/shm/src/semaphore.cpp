#include "shm/semaphore.hpp"          // shm::Semaphore, shm::Guard
#include "shm/error.hpp"                  // handle_errorno
#include "exception-rt/exception.hpp" // std::runtime_error
#include <cassert>                    // assert
#include <expected>                   // std::expected, std::unexpected
#include <fcntl.h>                    // O_CREAT, O_RDWR
#include <fmt/format.h>               // fmt::format
#include <semaphore.h>                // sem_open, sem_wait, sem_post, sem_close

namespace shm
{
    Semaphore::Semaphore(const std::string &name, int initial_value)
        : sem_name_(std::move(name))
    {
        assert(!sem_name_.empty() && "semaphore name is empty");
        assert(initial_value >= 0 && "initial value is negative");
        sem_t *sem = sem_open(sem_name_.c_str(), O_CREAT | O_EXCL, 0644, initial_value);
        if (sem == SEM_FAILED)
        {
            if (errno == EEXIST)
            {
                sem = sem_open(sem_name_.c_str(), O_RDWR, 0644, initial_value);
                if (sem == SEM_FAILED)
                {
                    throw std::runtime_error(fmt::format("sem_open failed: {} for semaphore: {}", strerror(errno), sem_name_));
                }
            }
            else
            {
                throw std::runtime_error(fmt::format("sem_open failed: {} for semaphore: {}", strerror(errno), sem_name_));
            }
        }

        int sem_value{};
        if (sem_getvalue(sem, &sem_value) == -1)
        {
            sem_close(sem);
            throw std::runtime_error(fmt::format("sem_getvalue failed: {} for semaphore: {}", strerror(errno), sem_name_));
        }

        if (sem_value < 0 || sem_value > initial_value)
        {
            fmt::print(stderr, "warning: resetting semaphore '{}' due to unexpected value {}\n", sem_name_, sem_value);
            sem_close(sem);
            sem_unlink(sem_name_.c_str());
            sem = sem_open(sem_name_.c_str(), O_CREAT | O_EXCL, 0644, initial_value);
            if (sem == SEM_FAILED)
            {
                throw std::runtime_error(fmt::format("sem_open failed during reset: {} for semaphore: {}", strerror(errno), sem_name_));
            }
        }

        sem_ = sem;
    }

    Semaphore::~Semaphore()
    {
        destroy();
    }

    Semaphore::Semaphore(Semaphore &&other) noexcept
        : sem_name_(std::move(other.sem_name_)), sem_(other.sem_)
    {
        other.sem_ = nullptr;
        other.sem_name_.clear();
    }

    Semaphore &Semaphore::operator=(Semaphore &&other) noexcept
    {
        if (this != &other)
        {
            destroy();
            sem_name_ = std::move(other.sem_name_);
            sem_ = other.sem_;
            other.sem_ = nullptr;
            other.sem_name_.clear();
        }
        return *this;
    }

    [[nodiscard]] bool Semaphore::is_valid() const
    {
        return sem_ != nullptr;
    }

    void Semaphore::wait()
    {
        assert(is_valid() && "semaphore is not valid");
        auto result = sem_wait(static_cast<sem_t *>(sem_));
        handle_errorno(result, "sem_wait");
    }

    void Semaphore::post()
    {
        assert(is_valid() && "semaphore is not valid");
        auto result = sem_post(static_cast<sem_t *>(sem_));
        handle_errorno(result, "sem_post");
    }

    void Semaphore::destroy()
    {
        if (!is_valid())
        {
            return;
        }
        post();
        sem_unlink(sem_name_.c_str());
        sem_close(static_cast<sem_t *>(sem_));
        sem_ = nullptr;
    }

    Guard::Guard(Semaphore &semaphore)
        : sem_(semaphore), locked_(false)
    {
        sem_.wait();
        locked_ = true;
    }

    Guard::~Guard()
    {
        unlockIfNeeded();
    }

    [[nodiscard]] bool Guard::isLocked() const
    {
        return locked_;
    }

    void Guard::unlockIfNeeded()
    {
        if (locked_)
        {
            sem_.post();
            locked_ = false;
        }
    }
} // namespace shm
