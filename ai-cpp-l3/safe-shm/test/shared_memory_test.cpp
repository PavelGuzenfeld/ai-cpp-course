#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/shared_memory.hpp"
#include "safe-shm/producer_consumer.hpp"

TEST_CASE("SharedMemory with int")
{
    safe_shm::SharedMemory<int> shm("doctest_int");
    shm.get() = 42;
    REQUIRE(shm.get() == 42);
}

TEST_CASE("SharedMemory with double")
{
    safe_shm::SharedMemory<double> shm("doctest_double");
    shm.get() = 42.42;
    REQUIRE(shm.get() == doctest::Approx(42.42));
}

TEST_CASE("SharedMemory with array")
{
    safe_shm::SharedMemory<int[10]> shm("doctest_array");
    int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::copy_n(data, 10, shm.get());
    for (int i = 0; i < 10; ++i)
    {
        REQUIRE(shm.get()[i] == i);
    }
    REQUIRE(shm.size() == sizeof(data));
    REQUIRE(shm.path() == "/dev/shm/doctest_array");
}

TEST_CASE("SharedMemory move constructor")
{
    safe_shm::SharedMemory<int> shm("doctest_move_ctor");
    shm.get() = 42;
    auto moved = std::move(shm);
    REQUIRE(moved.get() == 42);
}

TEST_CASE("SharedMemory with struct")
{
    struct FlatStruct
    {
        int a;
        double b;
        char buffer[50];
    };
    safe_shm::SharedMemory<FlatStruct> shm("doctest_struct");
    shm.get() = {42, 42.42, "Hello, shared memory!"};
    REQUIRE(shm.get().a == 42);
    REQUIRE(shm.get().b == doctest::Approx(42.42));
    REQUIRE(std::string(shm.get().buffer) == "Hello, shared memory!");
    REQUIRE(shm.size() == sizeof(FlatStruct));
    REQUIRE(shm.path() == "/dev/shm/doctest_struct");
}

TEST_CASE("SharedMemory with nested struct")
{
    struct Inner
    {
        int a;
        double b;
        char buffer[50];
    };
    struct Outer
    {
        Inner inner;
        int c;
    };

    safe_shm::SharedMemory<Outer> shm("doctest_nested");
    shm.get() = {{42, 42.42, "Hello!"}, 7};
    auto val = shm.get();
    REQUIRE(val.inner.a == 42);
    REQUIRE(val.inner.b == doctest::Approx(42.42));
    REQUIRE(std::string(val.inner.buffer) == "Hello!");
    REQUIRE(val.c == 7);
}
