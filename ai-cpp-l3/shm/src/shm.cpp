#include "shm/shm.hpp"                // Shm
#include "exception-rt/exception.hpp" // std::runtime_error
#include <cstring>                    // strerror (because you never included it properly)
#include <fcntl.h>                    // O_CREAT, O_RDWR
#include <fmt/format.h>               // fmt::format
#include <sys/mman.h>                 // mmap, PROT_WRITE, MAP_SHARED
#include <unistd.h>                   // ftruncate, close, open, unlink

namespace shm
{
    constexpr auto READ_WRITE_ALL = 0666;

    std::string path(std::string const& file_path) noexcept
    {
        return fmt::format("{}{}", SHARED_MEM_PATH, file_path);
    }

    Shm::Shm(std::string file_path, std::size_t size)
        : file_path_(std::move(file_path)),
          size_(size)
    {
        int fd = open(file_path_.c_str(), O_CREAT | O_RDWR, READ_WRITE_ALL);
        if (fd < 0)
        {
            throw std::runtime_error(fmt::format("Shared memory open failed: {} for file: {}",
                                                 strerror(errno), file_path_));
        }

        if (ftruncate(fd, size) < 0)
        {
            close(fd);
            throw std::runtime_error(fmt::format("Shared memory ftruncate failed: {} for file: {}",
                                                 strerror(errno), file_path_));
        }

        void *shm_ptr = mmap(nullptr, size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
        if (shm_ptr == MAP_FAILED)
        {
            close(fd);
            throw std::runtime_error(fmt::format("mmap failed: {} for file: {}",
                                                 strerror(errno), file_path_));
        }

        fd_ = fd;
        data_ = shm_ptr;
    }

    Shm::Shm(Shm &&other) noexcept
        : file_path_(std::move(other.file_path_)),
          size_(other.size_),
          fd_(other.fd_),
          data_(other.data_)
    {
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Shm &Shm::operator=(Shm &&other) noexcept
    {
        if (this != &other)
        {
            destroy();
            file_path_ = std::move(other.file_path_);
            size_ = other.size_;
            fd_ = other.fd_;
            data_ = other.data_;
            other.fd_ = -1;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    Shm::~Shm() noexcept
    {
        destroy();
    }

    [[nodiscard]] void *Shm::get() const noexcept
    {
        return data_;
    }

    [[nodiscard]] std::size_t Shm::size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] std::string Shm::file_path() const noexcept
    {
        return file_path_;
    }

    void Shm::destroy() noexcept
    {
        if (!file_path_.empty())
        {
            unlink(file_path_.c_str());
            file_path_.clear();
        }
        if (data_)
        {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }
} // namespace shm
