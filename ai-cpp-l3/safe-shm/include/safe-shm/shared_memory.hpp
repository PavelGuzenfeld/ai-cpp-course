#pragma once
#include "safe-shm/flat_type.hpp"
#include "shm/shm.hpp"

namespace safe_shm
{
    template <FlatType FLAT>
    class SharedMemory
    {
    public:
        explicit SharedMemory(std::string const &file_path)
            : impl_(shm::path(file_path), sizeof(FLAT))
        {
        }

        FLAT &get() noexcept
        {
            return *static_cast<FLAT *>(impl_.get());
        }

        auto size() const noexcept
        {
            return impl_.size();
        }

        auto path() const noexcept
        {
            return impl_.file_path();
        }

    private:
        shm::Shm impl_;
    };
} // namespace safe_shm
