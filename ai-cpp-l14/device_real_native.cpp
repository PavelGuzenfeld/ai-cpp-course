#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#define DEVICE_REAL_API
#include "device_buffer.hpp"

namespace nb = nanobind;

// The same shape as MockDevice, backed by a real V4L2 node. Built and
// installed unconditionally; it is *opening* the device that fails when no
// camera is present, which is what lets the tests skip rather than stub.
class RealDevice
{
public:
    explicit RealDevice(std::string path) : path_(std::move(path))
    {
        handle_ = device_open(path_.c_str());
        if (handle_ == nullptr)
        {
            throw std::runtime_error("device_open failed for " + path_);
        }
    }

    RealDevice(RealDevice const &) = delete;
    RealDevice &operator=(RealDevice const &) = delete;

    ~RealDevice() { device_close(handle_); }

    [[nodiscard]] DeviceBuffer read_frame()
    {
        DeviceFrame frame{};
        if (device_read_frame(handle_, &frame) != 0)
        {
            throw std::runtime_error("device_read_frame failed");
        }
        ++frame_count_;
        return DeviceBuffer(frame);
    }

    [[nodiscard]] int frame_count() const noexcept { return frame_count_; }

private:
    std::string path_;
    DeviceHandle *handle_ = nullptr;
    int frame_count_ = 0;
};

NB_MODULE(device_real_native, m)
{
    nb::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def_prop_ro("width", &DeviceBuffer::width)
        .def_prop_ro("height", &DeviceBuffer::height)
        .def_prop_ro("timestamp_ns", &DeviceBuffer::timestamp_ns)
        .def_prop_ro("data_checksum", &DeviceBuffer::data_checksum);

    nb::class_<RealDevice>(m, "RealDevice")
        .def(nb::init<std::string>(), nb::arg("path"))
        .def("read_frame", &RealDevice::read_frame)
        .def_prop_ro("frame_count", &RealDevice::frame_count);
}
