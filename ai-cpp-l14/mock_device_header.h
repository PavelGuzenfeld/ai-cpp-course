#pragma once
// The mock's struct layout must match the real header exactly, or bugs in
// real code can't be caught by mock tests (PRODUCTION_PLAN.md 1.6). It
// includes the real header purely to static_assert against it -- never to
// depend on it at runtime.
#include <cstddef>
#include <cstdint>

#include "real_device_header.h"

struct MockDeviceFrame
{
    std::uint32_t width;
    std::uint32_t height;
    std::uint64_t timestamp_ns;
    std::uint8_t data[64];
};

static_assert(sizeof(MockDeviceFrame) == sizeof(DeviceFrame),
              "MockDeviceFrame size diverges from the real DeviceFrame layout");
static_assert(offsetof(MockDeviceFrame, width) == offsetof(DeviceFrame, width),
              "width offset diverges from the real layout");
static_assert(offsetof(MockDeviceFrame, height) == offsetof(DeviceFrame, height),
              "height offset diverges from the real layout");
static_assert(offsetof(MockDeviceFrame, timestamp_ns) == offsetof(DeviceFrame, timestamp_ns),
              "timestamp_ns offset diverges from the real layout");
static_assert(offsetof(MockDeviceFrame, data) == offsetof(DeviceFrame, data),
              "data offset diverges from the real layout");
