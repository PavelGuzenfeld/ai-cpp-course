/// Seqlock example: lock-free single-value shared memory.
/// Run writer in one terminal, reader in another.
///
/// Build:
///   g++ -std=c++23 -O2 -o seqlock_writer seqlock_example.cpp -I../include -lshm -lexception-rt -lfmt
///
/// Usage:
///   ./seqlock_example writer   # terminal 1
///   ./seqlock_example reader   # terminal 2

#include "safe-shm/seqlock.hpp"
#include <chrono>
#include <cstring>
#include <fmt/core.h>
#include <thread>

struct SensorData
{
    double temperature;
    double pressure;
    double humidity;
    uint64_t timestamp_ms;
};

static_assert(safe_shm::FlatType<SensorData>);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fmt::print("Usage: {} <writer|reader>\n", argv[0]);
        return 1;
    }

    constexpr auto shm_name = "example_seqlock_sensor";

    if (std::strcmp(argv[1], "writer") == 0)
    {
        safe_shm::SeqlockWriter<SensorData> writer(shm_name);
        fmt::print("Writer started. Publishing sensor data...\n");

        for (int i = 0; i < 100; ++i)
        {
            SensorData data{
                .temperature = 20.0 + static_cast<double>(i) * 0.1,
                .pressure = 1013.25,
                .humidity = 45.0 + static_cast<double>(i % 10),
                .timestamp_ms = static_cast<uint64_t>(i) * 100};
            writer.store(data);
            fmt::print("  wrote: temp={:.1f} pressure={:.2f} humidity={:.1f}\n",
                       data.temperature, data.pressure, data.humidity);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    else if (std::strcmp(argv[1], "reader") == 0)
    {
        safe_shm::SeqlockReader<SensorData> reader(shm_name);
        fmt::print("Reader started. Reading sensor data...\n");

        uint32_t last_seq = 0;
        for (int i = 0; i < 50; ++i)
        {
            // Non-blocking read
            auto data = reader.load();
            fmt::print("  read: temp={:.1f} pressure={:.2f} humidity={:.1f} ts={}\n",
                       data.temperature, data.pressure, data.humidity, data.timestamp_ms);

            // Blocking read — waits for new data
            auto result = reader.load_blocking(last_seq, uint64_t{500'000'000}); // 500ms timeout
            if (result)
                fmt::print("  blocking: temp={:.1f} (new data)\n", result->temperature);
            else
                fmt::print("  blocking: timeout (no new data)\n");
        }
    }

    return 0;
}
