#pragma once
// Exercise 4 material: swap this in for mock_device_header.h and rebuild.
// `height` and `width` are swapped -- same sizeof, same field types, wrong
// offsets. This is the PRODUCTION_PLAN.md 1.6 bug: a mock that is "close
// enough" passes its own tests and still misreads every real frame's
// dimensions. Include this file in place of mock_device_header.h to watch
// the static_assert catch it at compile time instead of at runtime.
#include <cstddef>
#include <cstdint>

#include "real_device_header.h"

struct MockDeviceFrame
{
    std::uint32_t height; // BUG: swapped with width below
    std::uint32_t width;
    std::uint64_t timestamp_ns;
    std::uint8_t data[64];
};

static_assert(sizeof(MockDeviceFrame) == sizeof(DeviceFrame),
              "MockDeviceFrame size diverges from the real DeviceFrame layout");
static_assert(offsetof(MockDeviceFrame, width) == offsetof(DeviceFrame, width),
              "width offset diverges from the real layout");
static_assert(offsetof(MockDeviceFrame, height) == offsetof(DeviceFrame, height),
              "height offset diverges from the real layout");
