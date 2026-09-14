#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <stdexcept>
#include <string>

#define DEVICE_MOCK_API
#include "device_buffer.hpp"

namespace nb = nanobind;

// A purely-software stand-in for the vendor device: deterministic synthetic
// frames, no hardware, no I/O. This is what CI actually builds and tests.
class MockDevice
{
public:
    explicit MockDevice(std::string path) : path_(std::move(path)) {}

    [[nodiscard]] DeviceBuffer read_frame()
    {
        MockDeviceFrame frame{};
        frame.width = 64;
        frame.height = 48;
        frame.timestamp_ns = next_timestamp_ns_;
        next_timestamp_ns_ += 33'333'333; // ~30 fps
        for (std::size_t i = 0; i < sizeof(frame.data); ++i)
        {
            frame.data[i] = static_cast<std::uint8_t>((frame_count_ + i) & 0xFF);
        }
        ++frame_count_;
        return DeviceBuffer(frame);
    }

    [[nodiscard]] int frame_count() const noexcept { return frame_count_; }

private:
    std::string path_;
    std::uint64_t next_timestamp_ns_ = 0;
    int frame_count_ = 0;
};

NB_MODULE(device_mock_native, m)
{
    nb::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def_prop_ro("width", &DeviceBuffer::width)
        .def_prop_ro("height", &DeviceBuffer::height)
        .def_prop_ro("timestamp_ns", &DeviceBuffer::timestamp_ns)
        .def_prop_ro("data_checksum", &DeviceBuffer::data_checksum);

    nb::class_<MockDevice>(m, "MockDevice")
        .def(nb::init<std::string>(), nb::arg("path"))
        .def("read_frame", &MockDevice::read_frame)
        .def_prop_ro("frame_count", &MockDevice::frame_count);
}
