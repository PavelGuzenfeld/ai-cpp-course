#pragma once
#include <string> // std::string

namespace shm
{
    class Semaphore
    {
    public:
        Semaphore(const std::string &name, int initial_value = 0);
        ~Semaphore();

        Semaphore(const Semaphore &) = delete;
        Semaphore &operator=(const Semaphore &) = delete;
        Semaphore(Semaphore &&other) noexcept;
        Semaphore &operator=(Semaphore &&other) noexcept;

        [[nodiscard]] bool is_valid() const;
        
        void wait();
        void post();

        void destroy();

    private:
        std::string sem_name_;
        void *sem_ = nullptr;
    };

    class Guard
    {
    public:
        explicit Guard(Semaphore &semaphore);
        ~Guard();

        Guard(const Guard &) = delete;
        Guard &operator=(const Guard &) = delete;
        Guard(Guard &&) = delete;
        Guard &operator=(Guard &&) = delete;

        [[nodiscard]] bool isLocked() const;

    private:
        Semaphore &sem_;
        bool locked_;
        void unlockIfNeeded();
    };
} // namespace shm