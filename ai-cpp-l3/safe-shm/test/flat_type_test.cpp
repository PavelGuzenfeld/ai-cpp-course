#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/flat_type.hpp"
#include <array>
#include <string>
#include <vector>

struct TrivialStruct
{
    int x;
    double y;
    char buf[32];
};

struct NestedTrivial
{
    TrivialStruct inner;
    int z;
};

struct WithStdArray
{
    std::array<uint8_t, 1024> data;
    uint64_t timestamp;
};

struct NonTrivialMember
{
    std::string name;
    int value;
};

struct VirtualBase
{
    virtual ~VirtualBase() = default;
    int x;
};

TEST_CASE("FlatType accepts trivially copyable + standard layout types")
{
    SUBCASE("fundamental types")
    {
        CHECK(safe_shm::FlatType<int>);
        CHECK(safe_shm::FlatType<double>);
        CHECK(safe_shm::FlatType<float>);
        CHECK(safe_shm::FlatType<char>);
        CHECK(safe_shm::FlatType<uint8_t>);
        CHECK(safe_shm::FlatType<uint64_t>);
        CHECK(safe_shm::FlatType<bool>);
    }

    SUBCASE("C arrays")
    {
        CHECK(safe_shm::FlatType<int[10]>);
        CHECK(safe_shm::FlatType<char[256]>);
        CHECK(safe_shm::FlatType<double[3]>);
    }

    SUBCASE("trivial structs")
    {
        CHECK(safe_shm::FlatType<TrivialStruct>);
        CHECK(safe_shm::FlatType<NestedTrivial>);
        CHECK(safe_shm::FlatType<WithStdArray>);
    }
}

TEST_CASE("FlatType rejects non-flat types")
{
    SUBCASE("types with non-trivial members")
    {
        CHECK_FALSE(safe_shm::FlatType<std::string>);
        CHECK_FALSE(safe_shm::FlatType<std::vector<int>>);
        CHECK_FALSE(safe_shm::FlatType<NonTrivialMember>);
    }

    SUBCASE("types with virtual functions")
    {
        CHECK_FALSE(safe_shm::FlatType<VirtualBase>);
    }
}
