// Compiled as an OBJECT library only (see CMakeLists.txt) -- there is no
// real vendor .so to link against on x86 CI. This file's only job is to
// prove device_buffer.hpp type-checks against the real header's own
// DeviceFrame, not just against the mock's.
#include <cstdint>

#define DEVICE_REAL_API
#include "device_buffer.hpp"

// Exists only to force DeviceBuffer's template/method instantiation against
// the real header's DeviceFrame -- never called.
std::uint32_t device_buffer_layout_check_instantiate(DeviceFrame const &frame)
{
    DeviceBuffer const buffer(frame);
    return buffer.width();
}
