#include "flat_type.hpp"
#include <expected>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/mman.h>
#include <unistd.h>

namespace flat_shm
{
    constexpr auto const SHARED_MEM_PATH = "/dev/shm/";
    constexpr auto const READ_WRITE_ALL = 0666;

    template <FlatType FLAT_TYPE>
    struct SharedMemory
    {
        static constexpr std::expected<SharedMemory<FLAT_TYPE>, std::string> create(std::string const &shm_name) noexcept
        {
            auto const size = sizeof(FLAT_TYPE);
            auto const impl = create_impl(shm_name, size);
            if (impl.has_value())
            {
                return SharedMemory<FLAT_TYPE>(impl->file_path_, impl->fd_, impl->shm_ptr_, size);
            }
            return std::unexpected(impl.error());
        }

        SharedMemory(SharedMemory const &) = delete;
        SharedMemory &operator=(SharedMemory const &) = delete;

        SharedMemory(SharedMemory &&other) noexcept
            : impl_(std::move(other.impl_))
        {
            other.impl_ = SharedMemoryImpl{};
        }

        SharedMemory &operator=(SharedMemory &&other) noexcept
        {
            if (this != &other)
            {
                close_shm(impl_);
                impl_ = std::move(other.impl_);
                other.impl_ = SharedMemoryImpl{};
            }
            return *this;
        }

        ~SharedMemory() noexcept
        {
            close_shm(impl_);
        }

        inline void write(FLAT_TYPE const &data) const noexcept
        {
            std::memcpy(impl_.shm_ptr_, &data, sizeof(FLAT_TYPE));
        }

        inline FLAT_TYPE const &read() const noexcept
        {
            return *static_cast<FLAT_TYPE *>(impl_.shm_ptr_);
        }

        inline auto size() const noexcept
        {
            return impl_.size_;
        }

        inline auto path() const noexcept
        {
            return impl_.file_path_;
        }

    private:
        struct SharedMemoryImpl
        {
            std::string file_path_;
            int fd_ = -1;
            void *shm_ptr_ = nullptr;
            std::size_t size_ = 0;
        } impl_;

        static std::expected<SharedMemoryImpl, std::string> create_impl(std::string const &shm_name, std::size_t size) noexcept
        {
            auto const file_path = fmt::format("{}{}", SHARED_MEM_PATH, shm_name);

            // open the shared memory file
            int fd = open(file_path.c_str(), O_CREAT | O_RDWR, READ_WRITE_ALL);
            if (fd < 0)
            {
                auto const error_msg = fmt::format("Share memory open open file failed due to: {} for file: {}", strerror(errno), file_path);
                return std::unexpected(error_msg);
            }

            // set the size of the shared memory file
            if (ftruncate(fd, size) < 0)
            {
                close(fd);
                auto const error_msg = fmt::format("Shared memory ftruncate failed due to: {} for file: {}", strerror(errno), file_path);
                return std::unexpected(error_msg);
            }

            // map the shared memory
            void *shm_ptr = mmap(0, size, PROT_WRITE, MAP_SHARED, fd, 0);
            if (shm_ptr == MAP_FAILED)
            {
                close(fd);
                auto const error_msg = fmt::format("mmap failed failed due to: {} for file: {}", strerror(errno), file_path);
                return std::unexpected(error_msg);
            }

            return SharedMemoryImpl{file_path, fd, shm_ptr, size};
        }

        constexpr SharedMemory(std::string file_path, int fd, void *shm_ptr, std::size_t size) noexcept
            : impl_{file_path, fd, shm_ptr, size}

        {
        }

        static void close_shm(SharedMemoryImpl &impl) noexcept
        {
            if (impl.fd_ >= 0)
            {
                munmap(impl.shm_ptr_, impl.size_);
                close(impl.fd_);
                impl.fd_ = -1;
            }
        }
    };
} // namespace flat_shem