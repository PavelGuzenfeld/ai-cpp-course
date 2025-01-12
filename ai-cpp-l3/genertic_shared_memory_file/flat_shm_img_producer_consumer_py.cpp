#include "atomic_semaphore.hpp"
#include "flat_shm_impl.h"
#include "image.hpp"
#include "semaphore_impl.h"

#include <cstring> // For std::memcpy
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// This struct is now used as a shared_ptr in Python
struct ProducerConsumer
{
    flat_shm_impl::shm shm_;
    flat_shm_impl::Semaphore sem_read_;
    flat_shm_impl::Semaphore sem_write_;
    std::shared_ptr<img::Image4K_RGB> image_ = std::make_shared<img::Image4K_RGB>();
};

// The function returns std::shared_ptr<ProducerConsumer>
[[nodiscard]] constexpr ProducerConsumer create(const std::string &shm_name)
{
    auto impl = flat_shm_impl::create(shm_name, sizeof(img::Image4K_RGB));
    if (!impl)
        throw std::runtime_error(impl.error());

    auto sem_read = flat_shm_impl::create(shm_name + "_read", 0);
    if (!sem_read)
        throw std::runtime_error(sem_read.error());

    auto sem_write = flat_shm_impl::create(shm_name + "_write", 1);
    if (!sem_write)
        throw std::runtime_error(sem_write.error());

    return {
        std::move(impl.value()),
        std::move(sem_read.value()),
        std::move(sem_write.value())};
}

void destroy(ProducerConsumer &producer_consumer)
{
    flat_shm_impl::destroy(producer_consumer.shm_);
    flat_shm_impl::destroy(producer_consumer.sem_read_);
    flat_shm_impl::destroy(producer_consumer.sem_write_);
}

//--------------------------------------------------------------------------------------------
using AtomicImage = AtomicSemaphore<img::Image4K_RGB>;

struct AtomicProducerConsumer
{
    flat_shm_impl::shm shm_;
    std::shared_ptr<img::Image4K_RGB> image_ = std::make_shared<img::Image4K_RGB>();
};

[[nodiscard]] constexpr AtomicProducerConsumer create_atomic(const std::string &shm_name)
{
    auto impl = flat_shm_impl::create(shm_name, sizeof(AtomicImage));
    if (!impl)
        throw std::runtime_error(impl.error());

    return {
        std::move(impl.value())};
}

void destroy_atomic(AtomicProducerConsumer &producer_consumer)
{
    flat_shm_impl::destroy(producer_consumer.shm_);
}

// Create the Pybind11 module
PYBIND11_MODULE(Share_memory_image_producer_consumer, m)
{
    // Expose ImageType enum
    py::enum_<img::ImageType>(m, "ImageType")
        .value("RGB", img::ImageType::RGB)
        .value("RGBA", img::ImageType::RGBA)
        .value("NV12", img::ImageType::NV12)
        .export_values();

    // Expose Image4K_RGB
    py::class_<img::Image4K_RGB, std::shared_ptr<img::Image4K_RGB>>(m, "Image4K_RGB")
        .def(py::init<>())                                              // default constructor
        .def_readwrite("timestamp", &img::Image4K_RGB::timestamp)       // expose timestamp
        .def_readwrite("frame_number", &img::Image4K_RGB::frame_number) // expose frame_number
        .def("get_data", [](const img::Image4K_RGB &self)
             {
                 py::ssize_t shape[] = {
                     static_cast<py::ssize_t>(img::Image4K_RGB::height),
                     static_cast<py::ssize_t>(img::Image4K_RGB::width),
                     static_cast<py::ssize_t>(img::channels(img::Image4K_RGB::type))};
                 py::ssize_t strides[] = {
                     static_cast<py::ssize_t>(img::Image4K_RGB::width * img::channels(img::Image4K_RGB::type)),
                     static_cast<py::ssize_t>(img::channels(img::Image4K_RGB::type)),
                     1};

                 return py::array_t<uint8_t>(
                     {shape, shape + 3},     // shape
                     {strides, strides + 3}, // strides
                     self.data.data());      // pointer to the data
             })
        .def("set_data", [](img::Image4K_RGB &self, py::array_t<uint8_t> array)
             {
        auto buf = array.request();
        if (buf.ndim != 3 ||
            buf.shape[0] != img::Image4K_RGB::height ||
            buf.shape[1] != img::Image4K_RGB::width ||
            buf.shape[2] != static_cast<std::size_t>(img::channels(img::Image4K_RGB::type))) {
            throw std::runtime_error("Invalid array shape; expected (2160, 3840, 3)");
        }
        std::memcpy(self.data.data(), buf.ptr, buf.size * sizeof(uint8_t)); });

    py::class_<ProducerConsumer>(m, "ProducerConsumer")
        .def(py::init(&create), py::return_value_policy::move)
        .def("close", &destroy)
        .def("store", [](ProducerConsumer &self, img::Image4K_RGB const &image)
             {
    if (!self.image_) {
        throw std::runtime_error("Image pointer is null.");
    }
    // flat_shm_impl::wait(self.sem_write_);
    std::memcpy(self.shm_.data_, &image, sizeof(img::Image4K_RGB));
    flat_shm_impl::post(self.sem_read_); })
        .def("load", [](ProducerConsumer &self) -> std::shared_ptr<img::Image4K_RGB>
             {
                 if (!self.image_)
                 {
                     throw std::runtime_error("Image pointer is null.");
                 }
                 flat_shm_impl::wait(self.sem_read_);
                 std::memcpy(self.image_.get(), self.shm_.data_, sizeof(img::Image4K_RGB));
                 // flat_shm_impl::post(self.sem_write_);
                 return self.image_; }, py::return_value_policy::reference_internal);

    py::class_<AtomicProducerConsumer>(m, "AtomicProducerConsumer")
        .def(py::init(&create_atomic), py::return_value_policy::move)
        .def("close", &destroy_atomic)
        .def("store", [](AtomicProducerConsumer &self, img::Image4K_RGB const &image)
             {
                if (!self.image_) {
                    throw std::runtime_error("Image pointer is null.");
                }
                // self.shm_.write(image);
                auto atomic_image = static_cast<AtomicImage *>(self.shm_.data_);
                atomic_image->write(image); })
        .def("load", [](AtomicProducerConsumer &self) -> std::shared_ptr<img::Image4K_RGB>
             {
                    if (!self.image_)
                    {
                        throw std::runtime_error("Image pointer is null.");
                    }
                    auto * atomic_image = static_cast<AtomicImage *>(self.shm_.data_);
                    auto const* result = atomic_image->read();

                        std::memcpy(self.image_.get(), result, sizeof(img::Image4K_RGB));
                        return self.image_; }, py::return_value_policy::reference_internal);
}
