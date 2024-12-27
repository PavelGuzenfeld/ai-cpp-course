#pragma once
#include <expected>     // std::expected, std::unexpected
#include <fcntl.h>      // O_CREAT, O_RDWR
#include <fmt/format.h> // fmt::format
#include <string>       // std::string
#include <sys/mman.h>   // mmap, PROT_WRITE, MAP_SHARED
#include <unistd.h>     // ftruncate, close, open

namespace flat_shm_impl
{
    constexpr auto const SHARED_MEM_PATH = "/dev/shm/";
    constexpr auto const READ_WRITE_ALL = 0666;
    struct shm
    {
        std::string file_path_;
        int fd_ = -1;
        void *data_ = nullptr;
        std::size_t size_ = 0;

        inline void const *read_unsafe() const noexcept
        {
            return data_;
        }

        inline void *write_ref_unsafe() noexcept
        {
            return data_;
        }
    };

    inline std::expected<shm, std::string> make(std::string const &shm_name, std::size_t size) noexcept
    {
        auto const file_path = fmt::format("{}{}", SHARED_MEM_PATH, shm_name);

        // 1) open the shared memory file
        int fd = open(file_path.c_str(), O_CREAT | O_RDWR, READ_WRITE_ALL);
        if (fd < 0)
        {
            auto const error_msg = fmt::format("Shared memory open failed: {} for file: {}",
                                               strerror(errno), file_path);
            return std::unexpected(error_msg);
        }

        // 2) set the size of the shared memory file
        if (ftruncate(fd, size) < 0)
        {
            close(fd);
            auto const error_msg = fmt::format("Shared memory ftruncate failed: {} for file: {}",
                                               strerror(errno), file_path);
            return std::unexpected(error_msg);
        }

        // 3) map the shared memory
        std::byte *shm_ptr = static_cast<std::byte *>(
            mmap(nullptr, size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (shm_ptr == MAP_FAILED)
        {
            close(fd);
            auto const error_msg = fmt::format("mmap failed: {} for file: {}",
                                               strerror(errno), file_path);
            return std::unexpected(error_msg);
        }

        // 5) Construct and return the 'shm' object
        return shm{file_path, fd, shm_ptr, size};
    }

    inline void destroy(shm &impl) noexcept
    {
        if (impl.data_)
        {
            munmap(impl.data_, impl.size_);
            impl.data_ = nullptr;
        }
        if (impl.fd_ >= 0)
        {
            close(impl.fd_);
            impl.fd_ = -1;
        }
    }

} // namespace flat_shm_impl
