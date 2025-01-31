#include "async_runner.hpp"
#include "atomic_producer_consumer.hpp"
#include "atomic_semaphore.hpp"
#include "double_buffer_swapper.hpp"
#include "flat_shm_impl.h"
#include "image.hpp"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "semaphore_impl.h"

namespace nb = nanobind;
using namespace nb::literals;

struct ProducerConsumer
{
    flat_shm_impl::shm shm_;
    flat_shm_impl::Semaphore sem_read_;
    flat_shm_impl::Semaphore sem_write_;
    std::shared_ptr<img::Image4K_RGB> image_ = std::make_shared<img::Image4K_RGB>();
};

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

    return {std::move(impl.value()), std::move(sem_read.value()), std::move(sem_write.value())};
}

void destroy(ProducerConsumer &producer_consumer)
{
    flat_shm_impl::destroy(producer_consumer.shm_);
    flat_shm_impl::destroy(producer_consumer.sem_read_);
    flat_shm_impl::destroy(producer_consumer.sem_write_);
}

//--------------------------------------------------------------------------------------------

NB_MODULE(Share_memory_image_producer_consumer_nb, m)
{

    nb::class_<img::Image4K_RGB>(m, "Image4K_RGB")
        .def(nb::init<>())
        .def_rw("timestamp", &img::Image4K_RGB::timestamp)
        .def_rw("frame_number", &img::Image4K_RGB::frame_number)
        .def("shape", [](const img::Image4K_RGB &)
             { return std::array<size_t, 3>{
                   img::Image4K_RGB::height,
                   img::Image4K_RGB::width,
                   static_cast<size_t>(img::channels(img::Image4K_RGB::type))}; })

        .def("get_data", [](const img::Image4K_RGB &self)
             { return nb::ndarray<uint8_t const, nb::numpy, nb::shape<2160, 3840, 3>>((self.data.data())); }, nb::rv_policy::reference_internal)

        .def("set_data", [](img::Image4K_RGB &self, nb::ndarray<uint8_t const, nb::shape<img::Image4K_RGB::height, img::Image4K_RGB::width, static_cast<std::size_t>(img::channels(img::Image4K_RGB::type))>> array)
             { std::memcpy(self.data.data(), array.data(), array.size() * sizeof(uint8_t)); });

    nb::class_<ProducerConsumer>(m, "ProducerConsumer")
        .def_static("create", [](nb::str const &shm_name)
                    { return create(shm_name.c_str()); })
        .def(nb::init<>())
        .def("close", &destroy)
        .def("store", [](ProducerConsumer &self, img::Image4K_RGB const &image)
             {
            if (!self.image_) throw std::runtime_error("Image pointer is null.");
            std::memcpy(self.shm_.data_, &image, sizeof(img::Image4K_RGB));
            flat_shm_impl::post(self.sem_read_); })
        .def("load", [](ProducerConsumer &self) -> std::shared_ptr<img::Image4K_RGB>
             {
            if (!self.image_) throw std::runtime_error("Image pointer is null.");
            flat_shm_impl::wait(self.sem_read_);
            std::memcpy(self.image_.get(), self.shm_.data_, sizeof(img::Image4K_RGB));
            return self.image_; }, nb::rv_policy::reference_internal);

    nb::class_<AtomicProducerConsumer>(m, "AtomicProducerConsumer")
        .def_static("create", [](nb::str const &shm_name)
                    { return create_atomic(shm_name.c_str()); })
        .def(nb::init<flat_shm_impl::shm &&>())
        .def("close", &destroy_atomic)
        .def("store", [](AtomicProducerConsumer &self, img::Image4K_RGB const &image)
             {
        if (!self.image_)
            throw std::runtime_error("Image pointer is null.");
        auto atomic_image = static_cast<AtomicImage *>(self.shm_.data_);
        atomic_image->write(image); })

        .def("load", [](AtomicProducerConsumer &self) -> img::Image4K_RGB const *
             {
                if (!self.image_)
                     throw std::runtime_error("Image pointer is null.");
                auto atomic_image = static_cast<AtomicImage *>(self.shm_.data_);
                self.img_ptr_ = const_cast<img::Image4K_RGB *>(atomic_image->read());
                self.swapper_->stage(self.img_ptr_);
                self.runner_->trigger_once();
                return self.img_ptr_; }, nb::rv_policy::reference_internal);
}
