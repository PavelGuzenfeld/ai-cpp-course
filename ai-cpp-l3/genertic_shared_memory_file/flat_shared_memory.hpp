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
            auto const impl = flat_shm_impl::make(shm_name, size);
            if (impl.has_value())
            {
                return SharedMemory<FLAT_TYPE>{impl.value()};
            }
            return std::unexpected(impl.error());
        }

        SharedMemory(SharedMemory const &) = delete;
        SharedMemory &operator=(SharedMemory const &) = delete;

        SharedMemory(SharedMemory &&other) noexcept
            : impl_(std::move(other.impl_))
        {
            other.impl_ = flat_shm_impl::shm{};
        }

        SharedMemory &operator=(SharedMemory &&other) noexcept
        {
            if (this != &other)
            {
                flat_shm_impl::destroy(impl_);
                impl_ = std::move(other.impl_);
                other.impl_ = flat_shm_impl::shm{};
            }
            return *this;
        }

        ~SharedMemory() noexcept
        {
            flat_shm_impl::destroy(impl_);
        }

        inline FLAT_TYPE & write_ref() noexcept
        {
            return *static_cast<FLAT_TYPE *>(impl_.write_ref());
        }

        inline FLAT_TYPE const &read() const noexcept
        {
            return *static_cast<FLAT_TYPE const *>(impl_.read());
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

        constexpr SharedMemory(flat_shm_impl::shm impl) noexcept
            : impl_{impl}
        {
        }
    };
} // namespace flat_shem