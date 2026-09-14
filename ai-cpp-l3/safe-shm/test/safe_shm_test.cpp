#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/dblbuf_loader.hpp"
#include "safe-shm/storage.hpp"

TEST_CASE("store and load round-trip")
{
    constexpr auto shm_name = "test_shm";
    safe_shm::Storage<int> storage(shm_name);

    auto const data = 42;
    storage.store(data);

    safe_shm::DblBufLoader<int> loader(shm_name);
    auto snapshot = loader.load();
    loader.wait();
    REQUIRE(*snapshot == data);
}

TEST_CASE("store and load struct")
{
    struct Payload
    {
        int x;
        double y;
    };

    constexpr auto shm_name = "test_struct_shm";
    safe_shm::Storage<Payload> storage(shm_name);
    storage.store({7, 3.14});

    safe_shm::DblBufLoader<Payload> loader(shm_name);
    auto snapshot = loader.load();
    loader.wait();
    REQUIRE(snapshot->x == 7);
    REQUIRE(snapshot->y == doctest::Approx(3.14));
}

TEST_CASE("multiple stores update snapshot")
{
    constexpr auto shm_name = "test_multi_shm";
    safe_shm::Storage<int> storage(shm_name);

    storage.store(1);
    safe_shm::DblBufLoader<int> loader(shm_name);
    auto snapshot = loader.load();
    loader.wait();
    REQUIRE(*snapshot == 1);

    storage.store(99);
    snapshot = loader.load();
    loader.wait();
    REQUIRE(*snapshot == 99);
}
