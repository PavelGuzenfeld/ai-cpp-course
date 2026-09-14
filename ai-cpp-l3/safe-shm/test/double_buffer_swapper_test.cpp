#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/double_buffer_swapper.hpp"
#include <array>

TEST_CASE("swap copies staged data to pre-allocated and updates active pointer")
{
    int pre_allocated = 0;
    int *active = nullptr;
    safe_shm::DoubleBufferSwapper<int> swapper(&active, &pre_allocated);

    int source = 42;
    swapper.set_active(&source);
    REQUIRE(active == &source);
    REQUIRE(*active == 42);

    int staged = 99;
    swapper.stage(&staged);
    swapper.swap();

    CHECK(pre_allocated == 99);
    CHECK(active == &pre_allocated);
    CHECK(*active == 99);
}

TEST_CASE("multiple swap cycles")
{
    int pre_allocated = 0;
    int *active = nullptr;
    safe_shm::DoubleBufferSwapper<int> swapper(&active, &pre_allocated);

    int source = 10;
    swapper.set_active(&source);

    for (int i = 1; i <= 5; ++i)
    {
        int staged = i * 100;
        swapper.stage(&staged);
        swapper.swap();
        REQUIRE(*active == i * 100);
        REQUIRE(active == &pre_allocated);

        // Reset active to source for next cycle
        source = i * 100;
        swapper.set_active(&source);
    }
}

TEST_CASE("swap with struct type")
{
    struct Payload
    {
        int id;
        double value;
        char tag[8];
    };

    Payload pre_allocated{};
    Payload *active = nullptr;
    safe_shm::DoubleBufferSwapper<Payload> swapper(&active, &pre_allocated);

    Payload source{1, 3.14, "hello"};
    swapper.set_active(&source);

    Payload staged{2, 2.71, "world"};
    swapper.stage(&staged);
    swapper.swap();

    CHECK(active == &pre_allocated);
    CHECK(active->id == 2);
    CHECK(active->value == doctest::Approx(2.71));
    CHECK(std::string(active->tag) == "world");
}

TEST_CASE("swap with large array type")
{
    using BigArray = std::array<uint8_t, 4096>;

    BigArray pre_allocated{};
    BigArray *active = nullptr;
    safe_shm::DoubleBufferSwapper<BigArray> swapper(&active, &pre_allocated);

    BigArray source;
    source.fill(0xAA);
    swapper.set_active(&source);

    BigArray staged;
    staged.fill(0xBB);
    swapper.stage(&staged);
    swapper.swap();

    CHECK(active == &pre_allocated);
    for (auto byte : *active)
    {
        CHECK(byte == 0xBB);
    }
}
