#pragma once
#include <cstddef> // std::size_t
#include <string>  // std::string

namespace shm
{
    constexpr auto SHARED_MEM_PATH = "/dev/shm/";

    [[nodiscard]] std::string path(std::string const &file_name) noexcept;

    class Shm
    {
    public:
        Shm(std::string file_path, std::size_t size);
        Shm(const Shm &) = delete;
        Shm &operator=(const Shm &) = delete;
        Shm(Shm &&other) noexcept;
        Shm &operator=(Shm &&other) noexcept;
        ~Shm() noexcept;

        [[nodiscard]] void *get() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] std::string file_path() const noexcept;

        void destroy() noexcept;

    private:
        std::string file_path_;
        std::size_t size_ = 0;
        int fd_ = -1;
        mutable void *data_ = nullptr;
    };
}
