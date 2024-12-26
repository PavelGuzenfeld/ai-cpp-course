#include "flat_shared_memory.hpp"
#include <assert.h>
#include <chrono>
#include <fcntl.h>
#include <semaphore.h>

#include "flat_shm_impl.h"
#include <sys/wait.h>
#include <vector>

#include <atomic>

int main()
{
    using namespace flat_shm;

    {
        fmt::print("Test SharedMemory with int\n");
        auto shared_memory = SharedMemory<int>::create("int_file_name");
        shared_memory->write(42);
        auto read_int = shared_memory->read();
        assert(read_int == 42 && "Failed to read int");
    }

    {
        fmt::print("Test SharedMemory with double\n");
        auto shared_memory = SharedMemory<double>::create("double_file_name");
        shared_memory->write(42.42);
        auto read_double = shared_memory->read();
        assert(read_double == 42.42 && "Failed to read double");
    }

    {
        fmt::print("Test SharedMemory with char\n");
        auto shared_memory = SharedMemory<char>::create("char_file_name");
        shared_memory->write('c');
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
        shared_memory->write({42, 42.42, "Hello, shared memory!"});
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
        shared_memory->write({{42, 42.42, "Hello, shared memory!"}, 42});
        auto read_struct = shared_memory->read();
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
        shared_memory->write(data);
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
        shared_memory->write(42);
        auto shared_memory2 = std::move(shared_memory);
        auto read_int = shared_memory2->read();
        assert(read_int == 42 && "Failed to read int after move constructor");
    }

    {
        fmt::print("Move assignment test\n");
        auto shared_memory = SharedMemory<int>::create("move_assignment_file_name");
        shared_memory->write(42);
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

                auto const stats = Stats{.duration_accumulator = std::chrono::microseconds{read_duration + read_stats.duration_accumulator.count()},
                                         .read_count = read_stats.read_count + 1};

                shared_stats->write(stats);

                fmt::print("Subprocess {} completed successfully\n", i);
                sem_post(sem_write); // Signal parent
                _exit(EXIT_SUCCESS);
            }
            else
            {
                // parent process writes data for the child
                sem_wait(sem_write); // Wait for previous child to read

                large_data->time_stamp = std::chrono::high_resolution_clock::now();
                shared_memory->write(*large_data);

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

    fmt::print("All tests passed\n");
    return 0;
}