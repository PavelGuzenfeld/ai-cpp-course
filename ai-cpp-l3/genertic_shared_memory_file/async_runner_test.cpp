#include "async_runner.hpp"
#include <chrono>
#include <fmt/core.h>
#include <thread>
#include <cassert>

void test_inrement_nubmer()
{
    unsigned int number = 0;
    tasks::AsyncRunner runner([&number]()
                              { number++; },
                              [](std::string_view msg)
                              { fmt::print("Error: {}\n", msg); });
    runner.trigger_once();  // increment number
    runner.trigger_once();  // increment number
    runner.trigger_once();  // increment number

    std::this_thread::sleep_for(std::chrono::seconds(1));
    assert(number == 3);
    fmt::print("Number is {}\n", number);
}

void test_trigger_exception()
{
    unsigned int exception_counter = 0;
    
    tasks::AsyncRunner runner([]()
                              {
                                  throw std::runtime_error("Test exception");
                              },
                              [&exception_counter](std::string_view msg)
                              {
                                  fmt::print("Error: {}\n", msg);
                                  exception_counter++;
                              });
    runner.trigger_once();
    runner.trigger_once();
    runner.trigger_once();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    assert(exception_counter == 3);
    fmt::print("Exception counter is {}\n", exception_counter);
}


int main()
{
    test_inrement_nubmer();
    test_trigger_exception();
    fmt::print("All tests passed\n");
    return 0;
}