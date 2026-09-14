/// Cross-language interop writer: writes known values to shared memory
/// for the Python verifier (cross_language_verify.py) to read back.
/// This is NOT a doctest — it's a standalone main() used by run_cross_language_test.sh.

#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/seqlock.hpp"
#include "safe-shm/stamped.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fmt/core.h>
#include <thread>

int main()
{
    namespace fs = std::filesystem;
    using StampedF64 = safe_shm::Stamped<double>;

    // Clean up any stale signal file
    fs::remove("/tmp/xlang_test_done");

    // Test 1: Write π via SeqlockWriter<double>
    safe_shm::SeqlockWriter<double> seqlock_writer("xlang_seqlock");
    seqlock_writer.store(M_PI);

    // Test 2: Write 20 stamped values via CyclicBufferWriter<Stamped<double>, 64>
    safe_shm::CyclicBufferWriter<StampedF64, 64> cb_writer("xlang_cyclic");
    for (int i = 0; i < 20; ++i)
    {
        StampedF64 entry{
            .timestamp_ns = static_cast<uint64_t>(i) * 1000,
            .sequence = static_cast<uint64_t>(i),
            .data = static_cast<double>(i) * 1.5};
        cb_writer.insert(entry);
    }

    fmt::print("C++ writer ready\n");

    // Wait for the Python verifier to signal completion
    while (!fs::exists("/tmp/xlang_test_done"))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Test 4 (continued): Read back values written by Python
    safe_shm::SeqlockReader<double> py_seqlock("xlang_py_write");
    auto py_val = py_seqlock.load();
    if (std::abs(py_val - 2.71828) > 1e-4)
    {
        fmt::print("FAIL: expected 2.71828 from Python seqlock, got {}\n", py_val);
        return 1;
    }

    safe_shm::CyclicBufferReader<StampedF64, 64> py_cb("xlang_py_cyclic");
    if (py_cb.available() != 10)
    {
        fmt::print("FAIL: expected 10 entries from Python cyclic buffer, got {}\n",
                   py_cb.available());
        return 1;
    }
    auto py_latest = py_cb.get_latest();
    if (py_latest.sequence != 9 || std::abs(py_latest.data - 18.0) > 1e-10)
    {
        fmt::print("FAIL: Python cyclic latest: seq={} data={} (expected seq=9 data=18.0)\n",
                   py_latest.sequence, py_latest.data);
        return 1;
    }

    fmt::print("C++ read-back of Python data passed\n");
    return 0;
}
