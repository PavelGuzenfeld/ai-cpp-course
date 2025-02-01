#pragma once
#include "async_runner.hpp"
#include "atomic_semaphore.hpp"
#include "double_buffer_swapper.hpp"
#include "flat_shm_impl.h"
#include "image.hpp"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"

using AtomicImage = AtomicSemaphore<img::Image4K_RGB>;

struct AtomicProducerConsumer
{
    flat_shm_impl::shm shm_;
    std::shared_ptr<img::Image4K_RGB> image_ = std::make_shared<img::Image4K_RGB>();
    img::Image4K_RGB *img_ptr_ = nullptr;
    std::unique_ptr<DoubleBufferSwapper<img::Image4K_RGB>> swapper_ = nullptr;
    std::unique_ptr<run::SingleTaskRunner> runner_ = nullptr;

    AtomicProducerConsumer(flat_shm_impl::shm &&shm)
        : shm_(std::move(shm))
    {
        swapper_ = std::make_unique<DoubleBufferSwapper<img::Image4K_RGB>>(&img_ptr_, image_.get());
        runner_ = std::make_unique<run::SingleTaskRunner>([this]
                                                       { swapper_->swap(); }, [this](std::string_view msg)
                                                       { log(msg); });
    }

    inline void log(std::string_view msg) const noexcept
    {
        fmt::print("{}", msg);
    }
};

[[nodiscard]] constexpr AtomicProducerConsumer create_atomic(const std::string &shm_name)
{
    auto impl = flat_shm_impl::create(shm_name, sizeof(AtomicImage));
    if (!impl)
        throw std::runtime_error(impl.error());

    return std::move(impl.value());
};

void destroy_atomic(AtomicProducerConsumer &producer_consumer)
{
    flat_shm_impl::destroy(producer_consumer.shm_);
}