#pragma once
// Stands in for a vendor-shipped hardware SDK header -- the kind you get as
// a .h file with no source, whose struct layout you do not control and
// cannot change. This lesson's whole point is testing code written against
// this header without the hardware it describes.
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
