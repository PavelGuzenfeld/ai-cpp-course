#include <doctest/doctest.h>

#include "exception-rt/exception.hpp"

#include <cxxabi.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <unwind.h>
#include <vector>

// Redeclare ABI structs to validate header size at compile time.
// These match the Itanium C++ ABI used by GCC and Clang on Linux.
namespace abi_check
{
struct cxa_exception
{
    std::type_info *exceptionType;
    void (*exceptionDestructor)(void *);
    void (*unexpectedHandler)();
    std::terminate_handler terminateHandler;
    cxa_exception *nextException;
    int handlerCount;
    int handlerSwitchValue;
    const char *actionRecord;
    const char *languageSpecificData;
    void *catchTemp;
    void *adjustedPtr;
    _Unwind_Exception unwindHeader;
};

struct cxa_refcounted_exception
{
    int referenceCount;
    cxa_exception exc;
};
} // namespace abi_check

static_assert(EXCEPTION_RT_ABI_HEADER_SIZE >= sizeof(abi_check::cxa_refcounted_exception),
              "EXCEPTION_RT_ABI_HEADER_SIZE is smaller than sizeof(__cxa_refcounted_exception) — "
              "the C++ runtime will write past the reserved header space");

TEST_CASE("no slots in use at start")
{
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("throw and catch int")
{
    try
    {
        throw 42;
    }
    catch (int e)
    {
        CHECK(e == 42);
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("throw and catch double")
{
    try
    {
        throw 3.14;
    }
    catch (double e)
    {
        CHECK(e == doctest::Approx(3.14));
    }
}

TEST_CASE("throw and catch std::string")
{
    try
    {
        throw std::string("hello world");
    }
    catch (const std::string &e)
    {
        CHECK(e == "hello world");
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("throw and catch std::runtime_error")
{
    try
    {
        throw std::runtime_error("something went wrong");
    }
    catch (const std::runtime_error &e)
    {
        CHECK(std::string(e.what()) == "something went wrong");
    }
}

TEST_CASE("throw and catch std::logic_error")
{
    try
    {
        throw std::logic_error("bad logic");
    }
    catch (const std::logic_error &e)
    {
        CHECK(std::string(e.what()) == "bad logic");
    }
}

struct CustomException
{
    int code;
    double value;
    char message[64];
};

TEST_CASE("throw and catch custom struct")
{
    try
    {
        CustomException ex{};
        ex.code = 404;
        ex.value = 9.81;
        std::snprintf(ex.message, sizeof(ex.message), "not found");
        throw ex;
    }
    catch (const CustomException &e)
    {
        CHECK(e.code == 404);
        CHECK(e.value == doctest::Approx(9.81));
        CHECK(std::string(e.message) == "not found");
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("nested throw in catch")
{
    try
    {
        try
        {
            throw 1;
        }
        catch (int)
        {
            throw std::string("nested");
        }
    }
    catch (const std::string &e)
    {
        CHECK(e == "nested");
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("rethrow preserves value")
{
    try
    {
        try
        {
            throw 99;
        }
        catch (...)
        {
            throw; // rethrow
        }
    }
    catch (int e)
    {
        CHECK(e == 99);
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("exception_ptr keeps slot alive")
{
    std::exception_ptr eptr;
    CHECK(exception_rt::slots_in_use() == 0);

    try
    {
        throw 42;
    }
    catch (...)
    {
        eptr = std::current_exception();
    }

    CHECK(exception_rt::slots_in_use() == 1);

    eptr = nullptr;
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("rethrow_exception works")
{
    std::exception_ptr eptr;
    try
    {
        throw std::runtime_error("saved");
    }
    catch (...)
    {
        eptr = std::current_exception();
    }

    try
    {
        std::rethrow_exception(eptr);
    }
    catch (const std::runtime_error &e)
    {
        CHECK(std::string(e.what()) == "saved");
    }

    eptr = nullptr;
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("pointer comes from pool")
{
    try
    {
        throw 42;
    }
    catch (...)
    {
        auto eptr = std::current_exception();
        // The exception object lives in our pool
        // We can't easily get the raw pointer from exception_ptr,
        // but we verify via slots_in_use that our allocator was used
        CHECK(exception_rt::slots_in_use() == 1);
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("owns validates slot-aligned pointers only")
{
    int stack_var = 0;
    CHECK_FALSE(exception_rt::owns(&stack_var));
    CHECK_FALSE(exception_rt::owns(nullptr));

    // A thrown object pointer should be owned
    try
    {
        throw 42;
    }
    catch (int &e)
    {
        CHECK(exception_rt::owns(&e));
    }
}

TEST_CASE("many sequential throws reuse slots")
{
    for (int i = 0; i < 1000; ++i)
    {
        try
        {
            throw i;
        }
        catch (int e)
        {
            CHECK(e == i);
        }
    }
    CHECK(exception_rt::slots_in_use() == 0);
}

TEST_CASE("concurrent throws from multiple threads")
{
    constexpr int num_threads = 4;
    constexpr int iterations = 1000;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&success_count, t]
        {
            for (int i = 0; i < iterations; ++i)
            {
                try
                {
                    throw t * iterations + i;
                }
                catch (int e)
                {
                    if (e == t * iterations + i)
                    {
                        success_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto &th : threads) th.join();
    CHECK(success_count == num_threads * iterations);
    CHECK(exception_rt::slots_in_use() == 0);
}
