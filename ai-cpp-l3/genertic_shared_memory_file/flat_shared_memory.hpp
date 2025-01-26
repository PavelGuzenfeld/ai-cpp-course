#pragma once
#include "flat_shm_impl.h"
#include "flat_type.hpp"

namespace flat_shm
{
    template <FlatType FLAT_TYPE>
    struct SharedMemory
    {
        static constexpr std::expected<SharedMemory<FLAT_TYPE>, std::string> create(std::string const &shm_name) noexcept
        {
            auto const size = sizeof(FLAT_TYPE);
            auto impl = flat_shm_impl::create(shm_name, size);
            if (impl.has_value())
            {
                return SharedMemory<FLAT_TYPE>{std::move(impl.value())};
            }
            return std::unexpected(impl.error());
        }

        SharedMemory(SharedMemory const &) = delete;
        SharedMemory &operator=(SharedMemory const &) = delete;

        SharedMemory(SharedMemory &&other) noexcept
            : impl_(std::move(other.impl_))
        {
        }

        SharedMemory &operator=(SharedMemory &&other) noexcept
        {
            if (this != &other)
            {
                flat_shm_impl::destroy(impl_);
                impl_ = std::move(other.impl_);
            }
            return *this;
        }

        ~SharedMemory() noexcept
        {
            flat_shm_impl::destroy(impl_);
        }

        inline FLAT_TYPE &write_ref() noexcept
        {
            return *static_cast<FLAT_TYPE *>(flat_shm_impl::write_ref_unsafe(impl_));
        }

        inline FLAT_TYPE const &read() const noexcept
        {
            return *static_cast<FLAT_TYPE const *>(flat_shm_impl::read_unsafe(impl_));
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
        flat_shm_impl::shm impl_;


        SharedMemory(flat_shm_impl::shm &&impl) noexcept
            : impl_(std::move(impl))
        {
        }
    };
} // namespace flat_shem