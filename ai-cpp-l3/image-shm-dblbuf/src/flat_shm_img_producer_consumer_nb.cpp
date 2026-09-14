#include "image-shm-dblbuf/image.hpp"
#include "image-shm-dblbuf/shm.hpp"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include <cstring>

namespace nb = nanobind;
using namespace nb::literals;

using Image = img::Image4K_RGB;
using ImageShm = image_shm::DoubleBufferShm<Image>;
using ImageSnapshot = image_shm::Snapshot<Image>;

struct ProducerConsumer
{
    shm::Shm shm_;
    shm::Semaphore sem_read_;
    shm::Semaphore sem_write_;
    std::shared_ptr<Image> image_ = std::make_shared<Image>();

    ProducerConsumer(std::string const &shm_name)
        : shm_(shm::path(shm_name), sizeof(Image)),
          sem_read_(shm_name + "_read_sem", 0),
          sem_write_(shm_name + "_write_sem", 1)
    {
    }
};

NB_MODULE(image_shm_dblbuff, m)
{
    nb::class_<Image>(m, "Image4K_RGB")
        .def(nb::init<>())
        .def_rw("timestamp", &Image::timestamp)
        .def_rw("frame_number", &Image::frame_number)
        .def("shape", [](Image const &)
             { return std::array<size_t, 3>{
                   Image::height,
                   Image::width,
                   static_cast<size_t>(img::channels(Image::type))}; })
        .def("get_data", [](Image const &self)
             { return nb::ndarray<uint8_t const, nb::numpy, nb::shape<2160, 3840, 3>>(self.data.data()); },
             nb::rv_policy::reference_internal)
        .def("set_data", [](Image &self,
                            nb::ndarray<uint8_t const,
                                        nb::shape<Image::height, Image::width,
                                                  static_cast<std::size_t>(img::channels(Image::type))>>
                                array)
             { std::memcpy(self.data.data(), array.data(), array.size() * sizeof(uint8_t)); });

    nb::class_<ImageSnapshot>(m, "Snapshot")
        .def("timestamp", [](ImageSnapshot const &self)
             { return self->timestamp; }, nb::rv_policy::copy)
        .def("frame_number", [](ImageSnapshot const &self)
             { return self->frame_number; }, nb::rv_policy::copy)
        .def("get_data", [](ImageSnapshot const &self)
             { return nb::ndarray<uint8_t const, nb::numpy, nb::shape<2160, 3840, 3>>(self->data.data()); },
             nb::rv_policy::reference_internal)
        .def("__repr__", [](ImageSnapshot const &self) -> std::string
             { return fmt::format("Snapshot(ptr={:p}, timestamp={}, frame_number={})",
                                  static_cast<void const *>(self.get()),
                                  self->timestamp,
                                  self->frame_number); });

    nb::class_<ProducerConsumer>(m, "ProducerConsumer")
        .def(nb::init<std::string>(), nb::rv_policy::reference_internal)
        .def("store", [](ProducerConsumer &self, Image const &image)
             {
                 self.sem_write_.wait();
                 *static_cast<Image *>(self.shm_.get()) = image;
                 self.sem_read_.post(); })
        .def("load", [](ProducerConsumer &self) -> std::shared_ptr<Image>
             {
                 self.sem_read_.wait();
                 std::memcpy(self.image_.get(), self.shm_.get(), sizeof(Image));
                 self.sem_write_.post();
                 return self.image_; },
             nb::rv_policy::reference_internal);

    nb::class_<ImageShm>(m, "DoubleBufferShm")
        .def(nb::init<std::string>(), nb::rv_policy::reference_internal)
        .def("store", [](ImageShm &self, Image const &image)
             { self.store(image); })
        .def("load", [](ImageShm &self) -> ImageSnapshot
             { return self.load(); }, nb::rv_policy::reference_internal)
        .def("wait", &ImageShm::wait)
        .def("__repr__", [](ImageShm const &self) -> std::string
             { return fmt::format("DoubleBufferShm(shm={:p}, pre_allocated={:p})",
                                  self.shm_addr(),
                                  self.pre_allocated_addr()); });
}
