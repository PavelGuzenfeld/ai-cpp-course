#pragma once
// The vendor-style API this lesson mocks: a .h with a fixed struct layout you
// do not control, and three C entry points. real_device_v4l2.cpp implements
// it against a /dev/video* node, so "the real path" is a device that can
// actually exist rather than a hypothetical.
//
// The layout is still the constraint the mock must match exactly -- see the
// static_asserts in mock_device_header.h.
#include <cstdint>

struct DeviceFrame
{
    std::uint32_t width;
    std::uint32_t height;
    std::uint64_t timestamp_ns;
    std::uint8_t data[64];
};

extern "C"
{
    struct DeviceHandle;
    DeviceHandle *device_open(char const *path);
    int device_read_frame(DeviceHandle *handle, DeviceFrame *out);
    void device_close(DeviceHandle *handle);
}
