#pragma once
// One implementation, compiled twice (see CMakeLists.txt): once against the
// mock header (-DDEVICE_MOCK_API) and once against the real vendor header
// (-DDEVICE_REAL_API, backed by V4L2 in real_device_v4l2.cpp).
//
// The two compilations produce classes with DIFFERENT layouts, so they must
// not share a mangled name -- both modules load into one Python process, and
// two definitions of ::DeviceBuffer with different members is an ODR
// violation (see L19). nanobind catches it at import with "type
// 'DeviceBuffer' was already registered"; the per-API namespace is the fix.
#if defined(DEVICE_MOCK_API)
#include "mock_device_header.h"
#define DEVICE_API_NS device_mock_api
#elif defined(DEVICE_REAL_API)
#include "real_device_header.h"
#define DEVICE_API_NS device_real_api
#else
#error "define DEVICE_MOCK_API or DEVICE_REAL_API"
#endif

namespace DEVICE_API_NS
{
#if defined(DEVICE_MOCK_API)
    using ActiveFrame = MockDeviceFrame;
#else
    using ActiveFrame = DeviceFrame;
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
} // namespace DEVICE_API_NS

using DEVICE_API_NS::ActiveFrame;
using DEVICE_API_NS::DeviceBuffer;
