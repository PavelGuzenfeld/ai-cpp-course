#include "shm/error.hpp" // shm::handle_errorno
#include <cerrno>        // errno
#include <fmt/format.h>  // fmt::format
#include <stdexcept>     // std::runtime_error

namespace shm
{
    void handle_errorno(int errorno, std::string const &trace_fn_name)
    {
        if (errorno == 0)
        {
            return;
        }
        throw std::runtime_error(fmt::format("{} failed: {}", trace_fn_name, strerror(errorno)));
    }
} // namespace shm