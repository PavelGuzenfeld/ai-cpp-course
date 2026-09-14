/// SHM Lifecycle example: ownership, heartbeat, and cleanup.
///
/// Demonstrates:
/// - OwnedShm<T> RAII wrapper with ShmHeader
/// - Heartbeat-based liveness detection
/// - shm_exists / shm_list / shm_remove utilities
///
/// Build:
///   g++ -std=c++23 -O2 -o lifecycle_example lifecycle_example.cpp -I../include -lshm -lexception-rt -lfmt

#include "safe-shm/shm_lifecycle.hpp"
#include <chrono>
#include <fmt/core.h>
#include <thread>

struct Config
{
    double gain;
    double offset;
    uint32_t mode;
};

int main()
{
    constexpr auto name = "example_lifecycle_config";

    // List existing SHM segments
    fmt::print("Existing SHM segments:\n");
    for (auto const &seg : safe_shm::shm_list())
        fmt::print("  /dev/shm/{}\n", seg);

    // Create an owned segment
    {
        safe_shm::OwnedShm<Config> owned(name);
        fmt::print("\nCreated: /dev/shm/{}\n", name);
        fmt::print("  exists: {}\n", safe_shm::shm_exists(name));
        fmt::print("  magic: 0x{:08X}\n", owned.header().magic);
        fmt::print("  writer_pid: {}\n", owned.header().writer_pid);
        fmt::print("  alive: {}\n", safe_shm::is_writer_alive(owned.header()));

        // Write config data
        owned.data() = {1.5, 0.0, 3};
        owned.update_heartbeat();

        fmt::print("  config: gain={} offset={} mode={}\n",
                   owned.data().gain, owned.data().offset, owned.data().mode);

        // Check heartbeat freshness (1 second tolerance)
        fmt::print("  fresh (1s): {}\n",
                   safe_shm::is_writer_alive(owned.header(), 1'000'000'000));

        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        owned.update_heartbeat();
        fmt::print("  heartbeat updated\n");
    }
    // OwnedShm destructor runs — shm::Shm unlinks the file

    fmt::print("\nAfter destruction:\n");
    fmt::print("  exists: {}\n", safe_shm::shm_exists(name));

    // Manual cleanup of stale segments
    fmt::print("\nLooking for stale 'example_' segments:\n");
    for (auto const &seg : safe_shm::shm_list("example_"))
    {
        fmt::print("  removing: {}\n", seg);
        safe_shm::shm_remove(seg);
    }

    return 0;
}
