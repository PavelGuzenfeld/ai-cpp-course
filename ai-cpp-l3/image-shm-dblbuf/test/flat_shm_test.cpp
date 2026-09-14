#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "image-shm-dblbuf/flat_shared_memory.hpp"
#include "image-shm-dblbuf/flat_shm_producer_consumer.hpp"
#include <chrono>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <vector>
#include <fmt/core.h>

TEST_CASE("SharedMemory with int")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<int>("doctest_int");
    shared_memory.get() = 42;
    REQUIRE(shared_memory.get() == 42);
}

TEST_CASE("SharedMemory with double")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<double>("doctest_double");
    shared_memory.get() = 42.42;
    REQUIRE(shared_memory.get() == doctest::Approx(42.42));
}

TEST_CASE("SharedMemory with char")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<char>("doctest_char");
    shared_memory.get() = 'c';
    REQUIRE(shared_memory.get() == 'c');
}

TEST_CASE("SharedMemory with array")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<int[10]>("doctest_array");
    int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::copy_n(data, 10, shared_memory.get());
    for (int i = 0; i < 10; ++i)
    {
        REQUIRE(shared_memory.get()[i] == i);
    }
    REQUIRE(shared_memory.size() == sizeof(data));
    REQUIRE(shared_memory.path() == "/dev/shm/doctest_array");
}

TEST_CASE("SharedMemory move constructor")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<int>("doctest_move_ctor");
    shared_memory.get() = 42;
    auto moved = std::move(shared_memory);
    REQUIRE(moved.get() == 42);
}

TEST_CASE("SharedMemory move assignment")
{
    using namespace flat_shm;
    auto shared_memory = SharedMemory<int>("doctest_move_assign_src");
    shared_memory.get() = 42;
    auto dest = SharedMemory<int>("doctest_move_assign_dst");
    dest = std::move(shared_memory);
    REQUIRE(dest.get() == 42);
}

TEST_CASE("SharedMemory with struct")
{
    using namespace flat_shm;
    struct FlatStruct
    {
        int a;
        double b;
        char buffer[50];
    };
    auto shared_memory = SharedMemory<FlatStruct>("doctest_struct");
    shared_memory.get() = {42, 42.42, "Hello, shared memory!"};
    REQUIRE(shared_memory.get().a == 42);
    REQUIRE(shared_memory.get().b == doctest::Approx(42.42));
    REQUIRE(std::string(shared_memory.get().buffer) == "Hello, shared memory!");
    REQUIRE(shared_memory.size() == sizeof(FlatStruct));
    REQUIRE(shared_memory.path() == "/dev/shm/doctest_struct");
}

TEST_CASE("SharedMemory with nested struct")
{
    using namespace flat_shm;
    struct FlatStruct
    {
        int a;
        double b;
        char buffer[50];
    };
    struct NestedFlatStruct
    {
        FlatStruct inner;
        int c;
    };

    auto shared_memory = SharedMemory<NestedFlatStruct>("doctest_nested");
    shared_memory.get() = {{42, 42.42, "Hello, shared memory!"}, 42};
    auto read_struct = shared_memory.get();
    REQUIRE(read_struct.inner.a == 42);
    REQUIRE(read_struct.inner.b == doctest::Approx(42.42));
    REQUIRE(std::string(read_struct.inner.buffer) == "Hello, shared memory!");
    REQUIRE(read_struct.c == 42);
    REQUIRE(shared_memory.size() == sizeof(NestedFlatStruct));
    REQUIRE(shared_memory.path() == "/dev/shm/doctest_nested");
}

TEST_CASE("SharedMemory inter-process with semaphores")
{
    using namespace flat_shm;
    constexpr auto SHARED_MEM_IMAGE_4K_SIZE = 3 * 3840 * 2160;
    using ImageDataType = std::array<std::byte, SHARED_MEM_IMAGE_4K_SIZE>;

    struct TimedImage
    {
        std::chrono::high_resolution_clock::time_point time_stamp;
        ImageDataType pixels;
    };

    struct Stats
    {
        std::chrono::microseconds duration_accumulator = std::chrono::microseconds{0};
        size_t read_count = 0;
    };

    auto large_data = std::make_unique<TimedImage>();
    std::fill(large_data->pixels.begin(), large_data->pixels.end(), std::byte{0x42});

    auto shared_memory = SharedMemory<TimedImage>("doctest_image_shm");
    auto shared_stats = SharedMemory<Stats>("doctest_stats_shm");

    constexpr int N = 10;
    std::vector<pid_t> child_pids;

    sem_unlink("/doctest_write_sem");
    sem_unlink("/doctest_read_sem");
    sem_t *sem_write = sem_open("/doctest_write_sem", O_CREAT | O_EXCL, 0644, 1);
    sem_t *sem_read = sem_open("/doctest_read_sem", O_CREAT | O_EXCL, 0644, 0);

    REQUIRE(sem_write != SEM_FAILED);
    REQUIRE(sem_read != SEM_FAILED);

    auto const &read_stats = shared_stats.get();
    auto const &read_data = shared_memory.get();

    for (int i = 0; i < N; ++i)
    {
        pid_t pid = fork();
        REQUIRE(pid >= 0);

        if (pid == 0)
        {
            sem_wait(sem_read);
            for (std::size_t j = 0; j < SHARED_MEM_IMAGE_4K_SIZE; ++j)
            {
                if (read_data.pixels[j] != std::byte{0x42})
                {
                    _exit(EXIT_FAILURE);
                }
            }
            auto const now = std::chrono::high_resolution_clock::now();
            auto const read_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                now - read_data.time_stamp).count();

            shared_stats.get() = {
                .duration_accumulator = std::chrono::microseconds{
                    read_duration + read_stats.duration_accumulator.count()},
                .read_count = read_stats.read_count + 1};

            sem_post(sem_write);
            _exit(EXIT_SUCCESS);
        }
        else
        {
            sem_wait(sem_write);
            large_data->time_stamp = std::chrono::high_resolution_clock::now();
            shared_memory.get() = *large_data;
            sem_post(sem_read);
            child_pids.push_back(pid);
        }
    }

    for (pid_t pid : child_pids)
    {
        int status;
        waitpid(pid, &status, 0);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == 0);
    }

    REQUIRE(read_stats.read_count == N);

    auto average_us = read_stats.duration_accumulator.count() / read_stats.read_count;
    fmt::print("Average transfer duration: {} us ({} ms)\n", average_us, average_us / 1000);

    sem_close(sem_write);
    sem_close(sem_read);
    sem_unlink("/doctest_write_sem");
    sem_unlink("/doctest_read_sem");
}
