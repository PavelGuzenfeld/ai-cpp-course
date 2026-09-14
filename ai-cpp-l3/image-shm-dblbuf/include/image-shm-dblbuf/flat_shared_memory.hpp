#pragma once
#include "flat-type/flat.hpp"
#include "shm/shm.hpp"

namespace flat_shm
{
    template <FlatType FLAT>
    struct SharedMemory
    {
        SharedMemory(std::string const &file_path)
            : impl_(shm::path(file_path), sizeof(FLAT))
        {
        }

        inline FLAT &get() noexcept
        {
            return *static_cast<FLAT *>(impl_.get());
        }

        inline auto size() const noexcept
        {
            return impl_.size();
        }

        inline auto path() const noexcept
        {
            return impl_.file_path();
        }

    private:
        shm::Shm impl_;
    };
} // namespace flat_shm