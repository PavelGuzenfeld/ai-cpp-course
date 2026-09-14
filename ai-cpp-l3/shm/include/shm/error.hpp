#pragma once
#include <string> // std::string

namespace shm
{
    void handle_errorno(int errorno, std::string const &trace_fn_name);
} // namespace shm