#pragma once
#include <fmt/core.h>
#include <string_view>

namespace safe_shm
{
    inline void default_logger(std::string_view msg) noexcept
    {
        fmt::print("{}", msg);
    }
} // namespace safe_shm
