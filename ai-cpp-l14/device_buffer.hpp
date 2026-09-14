#pragma once
// One implementation, compiled twice (see CMakeLists.txt): once against the
// mock header (-DDEVICE_MOCK_API, what CI and this lesson's tests actually
// run), once against the real vendor header (-DDEVICE_REAL_API, compiled
// only -- there is no real device to link against on x86 CI; this proves
// the code type-checks against the real layout without ever running it).
#if defined(DEVICE_MOCK_API)
#include "mock_device_header.h"
using ActiveFrame = MockDeviceFrame;
#elif defined(DEVICE_REAL_API)
#include "real_device_header.h"
using ActiveFrame = DeviceFrame;
#else
#error "define DEVICE_MOCK_API or DEVICE_REAL_API"
#endif

class DeviceBuffer
{
public:
    explicit DeviceBuffer(ActiveFrame const &frame) : frame_(frame) {}

    [[nodiscard]] std::uint32_t width() const noexcept { return frame_.width; }
    [[nodiscard]] std::uint32_t height() const noexcept { return frame_.height; }
    [[nodiscard]] std::uint64_t timestamp_ns() const noexcept { return frame_.timestamp_ns; }

    // Exists so a test can observe the data[] fill pattern is deterministic
    // and varies per frame, without exposing the raw buffer to Python.
    [[nodiscard]] unsigned data_checksum() const noexcept
    {
        unsigned sum = 0;
        for (auto byte : frame_.data)
        {
            sum += byte;
        }
        return sum;
    }

private:
    ActiveFrame frame_;
};
