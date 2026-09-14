#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/swap_runner.hpp"
#include <atomic>
#include <chrono>
#include <thread>

TEST_CASE("trigger executes task")
{
    std::atomic<int> counter{0};
    safe_shm::SwapRunner runner(
        [&counter] { counter.fetch_add(1, std::memory_order_relaxed); },
        [](std::string_view) {});

    runner.trigger();
    runner.wait();
    CHECK(counter.load() == 1);
}

TEST_CASE("multiple triggers execute sequentially")
{
    std::atomic<int> counter{0};
    safe_shm::SwapRunner runner(
        [&counter] { counter.fetch_add(1, std::memory_order_relaxed); },
        [](std::string_view) {});

    for (int i = 0; i < 100; ++i)
    {
        runner.trigger();
        runner.wait();
    }
    CHECK(counter.load() == 100);
}

TEST_CASE("wait returns immediately when idle")
{
    std::atomic<int> counter{0};
    safe_shm::SwapRunner runner(
        [&counter] { counter.fetch_add(1, std::memory_order_relaxed); },
        [](std::string_view) {});

    auto start = std::chrono::steady_clock::now();
    runner.wait();
    auto elapsed = std::chrono::steady_clock::now() - start;

    CHECK(elapsed < std::chrono::milliseconds(50));
    CHECK(counter.load() == 0);
}

TEST_CASE("destructor stops runner cleanly")
{
    std::atomic<int> counter{0};
    {
        safe_shm::SwapRunner runner(
            [&counter] { counter.fetch_add(1, std::memory_order_relaxed); },
            [](std::string_view) {});

        runner.trigger();
        runner.wait();
    }
    CHECK(counter.load() == 1);
}

TEST_CASE("rapid trigger-wait cycles under contention")
{
    std::atomic<int> counter{0};
    safe_shm::SwapRunner runner(
        [&counter]
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            counter.fetch_add(1, std::memory_order_relaxed);
        },
        [](std::string_view) {});

    constexpr int N = 50;
    for (int i = 0; i < N; ++i)
    {
        runner.trigger();
        runner.wait();
    }
    CHECK(counter.load() == N);
}

TEST_CASE("task exception is caught and logged")
{
    std::atomic<bool> logged{false};
    safe_shm::SwapRunner runner(
        [] { throw std::runtime_error("test"); },
        [&logged](std::string_view) { logged.store(true, std::memory_order_relaxed); });

    runner.trigger();
    runner.wait();
    CHECK(logged.load());

    // Runner survives the exception and accepts more tasks
    // Can't replace the task, but the runner should still be alive
    // The original task will throw again
    runner.trigger();
    runner.wait();
    // logged was already true, runner didn't crash
    CHECK(logged.load());
}

TEST_CASE("concurrent triggers from multiple threads")
{
    std::atomic<int> counter{0};
    safe_shm::SwapRunner runner(
        [&counter] { counter.fetch_add(1, std::memory_order_relaxed); },
        [](std::string_view) {});

    constexpr int THREADS = 4;
    constexpr int PER_THREAD = 25;

    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (int t = 0; t < THREADS; ++t)
    {
        auto fn = [&runner]
        {
            for (int i = 0; i < PER_THREAD; ++i)
            {
                runner.trigger();
                runner.wait();
            }
        };
        threads.push_back(std::thread(fn));
    }
    for (auto &th : threads)
        th.join();

    // Due to coalescing (trigger while already triggered), counter may be
    // less than THREADS * PER_THREAD, but must be at least PER_THREAD
    CHECK(counter.load() >= PER_THREAD);
    CHECK(counter.load() <= THREADS * PER_THREAD);
}
