#include "flat_shared_memory.hpp"
#include "flat_shm_impl.h"
#include "flat_shm_producer_consumer.hpp"
#include <assert.h>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <vector>
int main()
{
    using namespace flat_shm;

    {
        fmt::print("Test SharedMemory with int\n");
        auto shared_memory = SharedMemory<int>::create("int_file_name");
        shared_memory->write_ref() = 42;
        auto read_int = shared_memory->read();
        assert(read_int == 42 && "Failed to read int");
    }

    {
        fmt::print("Test SharedMemory with double\n");
        auto shared_memory = SharedMemory<double>::create("double_file_name");
        shared_memory->write_ref() = 42.42;
        auto read_double = shared_memory->read();
        assert(read_double == 42.42 && "Failed to read double");
    }

    {
        fmt::print("Test SharedMemory with char\n");
        auto shared_memory = SharedMemory<char>::create("char_file_name");
        shared_memory->write_ref() = 'c';
        auto read_char = shared_memory->read();
        assert(read_char == 'c' && "Failed to read char");
    }

    {
        fmt::print("Test SharedMemory with struct\n");
        struct FlatStruct
        {
            int a;
            double b;
            char buffer[50]; // Correct declaration of a fixed-size array
        };
        auto shared_memory = SharedMemory<FlatStruct>::create("struct_file_name");
        shared_memory->write_ref() = {42, 42.42, "Hello, shared memory!"};
        auto read_struct = shared_memory->read();
        assert(read_struct.a == 42 && "Failed to read struct.a");
        assert(read_struct.b == 42.42 && "Failed to read struct.b");
        assert(std::string(read_struct.buffer) == "Hello, shared memory!" && "Failed to read struct.buffer");
        assert(shared_memory->size() == sizeof(FlatStruct) && "Failed to get size of struct");
        assert(shared_memory->path() == "/dev/shm/struct_file_name" && "Failed to get path of struct");
    }

    {
        fmt::print("Test SharedMemory with nested struct\n");

        struct FlatStruct
        {
            int a;
            double b;
            char buffer[50]; // Correct declaration of a fixed-size array
        };
        struct NestedFlatStruct
        {
            FlatStruct inner;
            int c;
        };

        auto shared_memory = SharedMemory<NestedFlatStruct>::create("nested_struct_file_name");
        auto start = std::chrono::high_resolution_clock::now();
        shared_memory->write_ref() = {{42, 42.42, "Hello, shared memory!"}, 42};
        auto read_struct = shared_memory->read();
        auto end = std::chrono::high_resolution_clock::now();
        fmt::print("Time to write and read nested struct: {} us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        assert(read_struct.inner.a == 42 && "Failed to read nested struct.inner.a");
        assert(read_struct.inner.b == 42.42 && "Failed to read nested struct.inner.b");
        assert(std::string(read_struct.inner.buffer) == "Hello, shared memory!" && "Failed to read nested struct.inner.buffer");
        assert(read_struct.c == 42 && "Failed to read nested struct.c");
        assert(shared_memory->size() == sizeof(NestedFlatStruct) && "Failed to get size of nested struct");
        assert(shared_memory->path() == "/dev/shm/nested_struct_file_name" && "Failed to get path of nested struct");
    }

    {
        fmt::print("Test SharedMemory with array\n");
        auto shared_memory = SharedMemory<int[10]>::create("array_file_name");
        int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int i = 0; i < 10; i++)
        {
            shared_memory->write_ref()[i] = data[i];
        }
        auto read_data = shared_memory->read();
        for (int i = 0; i < 10; i++)
        {
            assert(read_data[i] == i && "Failed to read array");
        }
        assert(shared_memory->size() == sizeof(int[10]) && "Failed to get size of array");
        assert(shared_memory->path() == "/dev/shm/array_file_name" && "Failed to get path of array");
    }

    {
        fmt::print("Move constructor test\n");
        auto shared_memory = SharedMemory<int>::create("move_constructor_file_name");
        shared_memory->write_ref() = 42;
        auto shared_memory2 = std::move(shared_memory);
        auto read_int = shared_memory2->read();
        assert(read_int == 42 && "Failed to read int after move constructor");
    }

    {
        fmt::print("Move assignment test\n");
        auto shared_memory = SharedMemory<int>::create("move_assignment_file_name");
        shared_memory->write_ref() = 42;
        auto shared_memory2 = SharedMemory<int>::create("move_assignment_file_name2");
        shared_memory2 = std::move(shared_memory);
        auto read_int = shared_memory2->read();
        assert(read_int == 42 && "Failed to read int after move assignment");
    }

    {
        fmt::print("Test SharedMemory with large image between processes 10K times - timed\n");
        constexpr auto SHARED_MEM_IMAGE_4K_SIZE = 3 * 3840 * 2160; // 4K image size (bytes)
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

        // create image data
        auto large_data = std::make_unique<TimedImage>();
        std::fill(large_data->pixels.begin(), large_data->pixels.end(), std::byte{0x42});

        // create shared memory
        auto shared_memory = SharedMemory<TimedImage>::create("image_shm_test");
        auto shared_stats = SharedMemory<Stats>::create("stats_shm_test");

        constexpr int N = 10; // number of subprocesses
        std::vector<pid_t> child_pids;

        // Create semaphores
        sem_t *sem_write = sem_open("/shm_write_sem", O_CREAT | O_EXCL, 0644, 1);
        sem_t *sem_read = sem_open("/shm_read_sem", O_CREAT | O_EXCL, 0644, 0);

        if (sem_write == SEM_FAILED || sem_read == SEM_FAILED)
        {
            perror("Failed to create semaphores");
            exit(EXIT_FAILURE);
        }

        auto const &read_stats = shared_stats->read();
        auto const &read_data = shared_memory->read();

        for (int i = 0; i < N; ++i)
        {
            pid_t pid = fork();
            if (pid < 0)
            {
                fmt::print("Failed to fork subprocess {}\n", i);
                continue;
            }

            if (pid == 0)
            {
                // child process
                sem_wait(sem_read); // Wait for parent to write

                // verify data
                for (std::size_t j = 0; j < SHARED_MEM_IMAGE_4K_SIZE; ++j)
                {
                    if (read_data.pixels[j] != std::byte{0x42})
                    {
                        fmt::print("Data mismatch in subprocess {} at index {}\n", i, j);
                        _exit(EXIT_FAILURE);
                    }
                }
                auto const now = std::chrono::high_resolution_clock::now();
                auto const read_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - read_data.time_stamp).count();

                shared_stats->write_ref() = {.duration_accumulator = std::chrono::microseconds{read_duration + read_stats.duration_accumulator.count()},
                                             .read_count = read_stats.read_count + 1};

                fmt::print("Subprocess {} completed successfully\n", i);
                sem_post(sem_write); // Signal parent
                _exit(EXIT_SUCCESS);
            }
            else
            {
                // parent process writes data for the child
                sem_wait(sem_write); // Wait for previous child to read

                large_data->time_stamp = std::chrono::high_resolution_clock::now();
                shared_memory->write_ref() = *large_data;

                sem_post(sem_read); // Signal child
                child_pids.push_back(pid);
            }
        }

        // parent process waits for children to complete
        for (pid_t pid : child_pids)
        {
            int status;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            {
                fmt::print("Subprocess failed with PID {}\n", pid);
            }
        }

        // Calculate and print average
        if (read_stats.read_count > 0)
        {
            auto average_read_duration_us = read_stats.duration_accumulator.count() / read_stats.read_count;
            fmt::print("Average transfer duration: {} us\n", average_read_duration_us);
            fmt::print("Average transfer duration: {} ms\n", average_read_duration_us / 1000);
        }
        else
        {
            fmt::print("No successful reads.\n");
        }

        // Cleanup semaphores
        sem_close(sem_write);
        sem_close(sem_read);
        sem_unlink("/shm_write_sem");
        sem_unlink("/shm_read_sem");
    }

    {
        fmt::print("Test SharedMemory with large image between processes - lock-free style\n");

        // 4K image (3 channels * 3840 x 2160)
        constexpr auto SHARED_MEM_IMAGE_4K_SIZE = 3 * 3840 * 2160;
        using ImageDataType = std::array<std::byte, SHARED_MEM_IMAGE_4K_SIZE>;

        // We store one 'TimedImage' in shared memory
        struct TimedImage
        {
            std::chrono::high_resolution_clock::time_point time_stamp;
            ImageDataType pixels;
        };

        // Instead of a plain struct, store atomics directly for lock-free updates
        struct Stats
        {
            // We’ll accumulate times in microseconds
            std::atomic<long long> duration_accumulator_us{0};
            std::atomic<size_t> read_count{0};
        };

        // Our combined shared memory region:
        // - data_ready: child spins on this until the parent publishes
        // - data_read : parent spins on this until the child finishes reading
        // - image     : the data to transfer
        // - stats     : accumulative stats (duration, count)
        struct SharedRegion
        {
            std::atomic<bool> data_ready{false};
            std::atomic<bool> data_read{false};
            TimedImage image{};
            Stats stats{};
        };

        // Create shared memory
        auto shared_mem = SharedMemory<SharedRegion>::create("image_shm_test_lockfree");

        // Prepare the large image data in the parent
        auto large_data = std::make_unique<TimedImage>();
        std::fill(large_data->pixels.begin(), large_data->pixels.end(), std::byte{0x42});

        constexpr int N = 10; // number of subprocesses
        std::vector<pid_t> child_pids;

        // For each child, we:
        // 1) Reset data_ready/data_read to false
        // 2) Parent publishes the data (timestamp + image) => data_ready = true
        // 3) Child waits on data_ready, reads + verifies, updates stats => data_read = true
        // 4) Parent waits on data_read, goes on to next child
        //
        // This ensures no race conditions using only atomic flags.

        for (int i = 0; i < N; ++i)
        {
            // Clear flags for this child’s iteration
            {
                auto &region = shared_mem->write_ref();
                region.data_ready.store(false, std::memory_order_release);
                region.data_read.store(false, std::memory_order_release);
            }

            pid_t pid = fork();
            if (pid < 0)
            {
                fmt::print("Failed to fork subprocess {}\n", i);
                continue;
            }
            else if (pid == 0)
            {
                // Child process: wait until parent sets data_ready = true
                auto &region = shared_mem->write_ref(); // same as readRef but non-const

                while (!region.data_ready.load(std::memory_order_acquire))
                {
                    sched_yield();
                }

                // Child verifies data
                const auto &image = region.image;
                for (std::size_t j = 0; j < SHARED_MEM_IMAGE_4K_SIZE; ++j)
                {
                    if (image.pixels[j] != std::byte{0x42})
                    {
                        fmt::print("Data mismatch in subprocess {} at index {}\n", i, j);
                        _exit(EXIT_FAILURE);
                    }
                }

                // Compute time
                auto now = std::chrono::high_resolution_clock::now();
                auto read_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(now - image.time_stamp).count();

                // Update stats atomically
                region.stats.duration_accumulator_us.fetch_add(read_duration_us, std::memory_order_relaxed);
                region.stats.read_count.fetch_add(1, std::memory_order_relaxed);

                fmt::print("Subprocess {} completed successfully\n", i);

                // Indicate to the parent that data is read
                region.data_read.store(true, std::memory_order_release);

                _exit(EXIT_SUCCESS);
            }
            else
            {
                // Parent process: publish data for the child
                child_pids.push_back(pid);

                // 1) Set the timestamp + copy the large data
                {
                    auto &region = shared_mem->write_ref();
                    region.image.time_stamp = std::chrono::high_resolution_clock::now();
                    region.image.pixels = large_data->pixels;
                }

                // 2) data_ready = true
                {
                    auto &region = shared_mem->write_ref();
                    region.data_ready.store(true, std::memory_order_release);
                }

                // 3) Wait until the child sets data_read = true
                {
                    auto &region = shared_mem->write_ref();
                    while (!region.data_read.load(std::memory_order_acquire))
                    {
                        sched_yield();
                    }
                }
            }
        }

        // Wait for all children to complete
        for (auto pid : child_pids)
        {
            int status;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            {
                fmt::print("Subprocess failed with PID {}\n", pid);
            }
        }

        // Print average duration if we had successful reads
        {
            const auto &region = shared_mem->read();
            auto read_count = region.stats.read_count.load(std::memory_order_relaxed);
            if (read_count > 0)
            {
                auto duration_acc = region.stats.duration_accumulator_us.load(std::memory_order_relaxed);
                auto average_us = duration_acc / read_count;
                fmt::print("Average transfer duration: {} us\n", average_us);
                fmt::print("Average transfer duration: {} ms\n", static_cast<double>(average_us) / 1000.0);
            }
            else
            {
                fmt::print("No successful reads.\n");
            }
        }

        {
            // Producer-consumer move constructor test
            fmt::print("Producer-consumer move constructor test\n");
            auto producer_consumer = FlatShmProducerConsumer<int>::create("producer_consumer_move_constructor_file_name");
            producer_consumer->produce(42);
            auto producer_consumer2 = std::move(producer_consumer);
            auto read_int = producer_consumer2->consume();
            assert(read_int == 42 && "Failed to read int after producer-consumer move constructor");
        }
    }
    fmt::print("All tests passed\n");
    return 0;
}