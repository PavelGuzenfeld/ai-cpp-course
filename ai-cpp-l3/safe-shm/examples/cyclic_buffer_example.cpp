/// CyclicBuffer example: lock-free ring buffer with time-series lookups.
///
/// Demonstrates:
/// - CyclicBufferWriter/Reader for cross-process ring buffer
/// - Stamped<T> metadata envelope
/// - TimeSeries temporal queries (find_closest, interpolation)
///
/// Build:
///   g++ -std=c++23 -O2 -o cyclic_example cyclic_buffer_example.cpp -I../include -lshm -lexception-rt -lfmt
///
/// Usage:
///   ./cyclic_example writer   # terminal 1
///   ./cyclic_example reader   # terminal 2

#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/time_series.hpp"
#include <chrono>
#include <cstring>
#include <fmt/core.h>
#include <thread>

struct IMUReading
{
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
};

static_assert(safe_shm::FlatType<IMUReading>);

using StampedIMU = safe_shm::Stamped<IMUReading>;
constexpr std::size_t BUFFER_SIZE = 64; // must be power of two
constexpr auto SHM_NAME = "example_cyclic_imu";

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fmt::print("Usage: {} <writer|reader>\n", argv[0]);
        return 1;
    }

    if (std::strcmp(argv[1], "writer") == 0)
    {
        safe_shm::CyclicBufferWriter<StampedIMU, BUFFER_SIZE> writer(SHM_NAME);
        fmt::print("IMU writer started (buffer capacity: {})\n", BUFFER_SIZE);

        for (uint64_t seq = 0; seq < 200; ++seq)
        {
            IMUReading imu{
                .accel_x = 0.01f * static_cast<float>(seq),
                .accel_y = 0.0f,
                .accel_z = -9.81f,
                .gyro_x = 0.0f,
                .gyro_y = 0.0f,
                .gyro_z = 0.001f * static_cast<float>(seq)};

            auto stamped = safe_shm::stamp(imu, seq);
            writer.insert(stamped);

            if (seq % 50 == 0)
                fmt::print("  wrote seq={} ts={} accel_x={:.3f}\n",
                           seq, stamped.timestamp_ns, imu.accel_x);

            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 200 Hz
        }
        fmt::print("Writer done. Total writes: {}\n", writer.total_writes());
    }
    else if (std::strcmp(argv[1], "reader") == 0)
    {
        // TimeSeries wraps CyclicBufferReader with temporal queries
        safe_shm::TimeSeries<StampedIMU, BUFFER_SIZE> ts(SHM_NAME);
        fmt::print("IMU reader started\n");

        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // let writer fill buffer

        fmt::print("Available: {}/{}\n", ts.available(), ts.capacity());

        // Read latest
        auto latest = ts.get_latest();
        fmt::print("Latest: seq={} accel_x={:.3f}\n",
                   latest.sequence, latest.data.accel_x);

        // Staleness check
        auto now = safe_shm::monotonic_now_ns();
        auto fresh = ts.get_latest_if_fresh(now - 1'000'000'000); // 1 second
        if (fresh)
            fmt::print("Data is fresh (within 1s)\n");
        else
            fmt::print("Data is stale (older than 1s)\n");

        // Find closest to a specific timestamp
        auto target_ts = latest.timestamp_ns - 100'000'000; // 100ms ago
        auto closest = ts.find_closest(target_ts);
        if (closest)
            fmt::print("Closest to {}ns ago: seq={} delta={}ns\n",
                       100'000'000, closest->sequence,
                       closest->timestamp_ns > target_ts
                           ? closest->timestamp_ns - target_ts
                           : target_ts - closest->timestamp_ns);

        // Interpolation pair
        auto interp = ts.find_interpolation_pair(target_ts);
        if (interp)
            fmt::print("Interpolation: before.seq={} after.seq={} alpha={:.3f}\n",
                       interp->before.sequence, interp->after.sequence, interp->alpha);
    }

    return 0;
}
