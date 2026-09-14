#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/image.hpp"
#include "safe-shm/image_shm.hpp"

using Image = safe_shm::img::Image4K_RGB;
using ImageShm = safe_shm::DoubleBufferShm<Image>;

TEST_CASE("store and load round-trip")
{
    ImageShm shm("doctest_dblbuf");

    REQUIRE(shm.shm_addr() != nullptr);

    auto img = std::make_unique<Image>();
    img->timestamp = 123456789;
    img->frame_number = 123;
    std::fill(img->data.begin(), img->data.end(), 0x42);

    shm.store(*img);

    auto snapshot = shm.load();
    shm.wait();

    REQUIRE(snapshot->timestamp == 123456789);
    REQUIRE(snapshot->frame_number == 123);
    REQUIRE(std::all_of(snapshot->data.begin(), snapshot->data.end(),
                        [](auto v) { return v == 0x42; }));
}

TEST_CASE("snapshot transitions from shm to pre-allocated copy")
{
    ImageShm shm("doctest_transition");

    auto img = std::make_unique<Image>();
    img->timestamp = 999;
    img->frame_number = 7;
    std::fill(img->data.begin(), img->data.end(), 0xAB);

    shm.store(*img);

    auto snapshot = shm.load();
    REQUIRE(snapshot.get() == shm.shm_addr());

    shm.wait();
    REQUIRE(snapshot.get() == shm.pre_allocated_addr());

    REQUIRE(snapshot->timestamp == 999);
    REQUIRE(snapshot->frame_number == 7);
    REQUIRE(std::all_of(snapshot->data.begin(), snapshot->data.end(),
                        [](auto v) { return v == 0xAB; }));
}

TEST_CASE("multiple load cycles")
{
    ImageShm shm("doctest_multi_load");

    auto img = std::make_unique<Image>();
    img->timestamp = 1;
    img->frame_number = 1;
    std::fill(img->data.begin(), img->data.end(), 0x01);

    shm.store(*img);
    auto s1 = shm.load();
    shm.wait();
    REQUIRE(s1->timestamp == 1);

    img->timestamp = 2;
    img->frame_number = 2;
    std::fill(img->data.begin(), img->data.end(), 0x02);

    shm.store(*img);
    auto s2 = shm.load();
    shm.wait();
    REQUIRE(s2->timestamp == 2);
    REQUIRE(s2->frame_number == 2);
}
