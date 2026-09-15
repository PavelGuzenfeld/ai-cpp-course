#include "../record_parser.hpp"

// The standalone build (see CMakeLists.txt) links this same source without
// -fsanitize=fuzzer and calls LLVMFuzzerTestOneInput directly over a fixed
// corpus — a deterministic smoke sweep, not a fuzz campaign. THIS binary,
// built with clang and -fsanitize=fuzzer,address, is the actual fuzzer: it
// mutates its own inputs under coverage guidance to find crashes and hangs.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size)
{
    (void)record_parser::parse(data, size);
    return 0;
}
