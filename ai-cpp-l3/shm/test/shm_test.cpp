#include "shm/semaphore.hpp" // Include the semaphore header
#include "shm/shm.hpp"       // Include the shared memory header
#include "shm/error.hpp"     // Include the error header
#include <cassert>           // Include the assert header
#include <fmt/core.h>        // Include the fmt header

void shm_test()
{
    // Create a shared memory object
    auto const shm_file_name = "shm_test";
    auto const shm_size = 1024;

    auto const shm_path = shm::path(shm_file_name);
    assert(shm_path == std::string(shm::SHARED_MEM_PATH) + shm_file_name);

    auto shm = shm::Shm(shm_path, shm_size);
    assert(shm.size() == shm_size && "Size mismatch");
    assert(shm.file_path() == std::string(shm_path) && "File path mismatch");

    // Write to shared memory
    std::string data = "Hello, shared memory!";
    std::memcpy(shm.get(), data.data(), data.size());

    // Read from shared memory
    std::string_view view(reinterpret_cast<const char *>(shm.get()), data.size());
    assert(view == data && "Data mismatch");
}

void shm_move_constructor_test()
{
    // Create a shared memory object
    auto const shm_file_name = "shm_test";
    auto const shm_size = 1024;

    auto const shm_path = shm::path(shm_file_name);
    assert(shm_path == std::string(shm::SHARED_MEM_PATH) + shm_file_name && "Path mismatch");

    auto shm = shm::Shm(shm_path, shm_size);
    assert(shm.size() == shm_size && "Size mismatch");
    assert(shm.file_path() == shm_path && "File path mismatch");

    // Move constructor
    auto shm2 = std::move(shm);
    assert(shm2.size() == shm_size && "Size mismatch");
    assert(shm2.file_path() == shm_path && "File path mismatch");
    assert(shm.get() == nullptr && "Data mismatch");
    assert(shm.size() == 0 && "Size mismatch");
    assert(shm.file_path().empty() && "File path mismatch");
}

void shm_move_operator_test()
{
    // Create a shared memory object
    auto const shm_file_name = "shm_test";
    auto const shm_size = 1024;

    auto const shm_path = shm::path(shm_file_name);
    assert(shm_path == std::string(shm::SHARED_MEM_PATH) + shm_file_name && "Path mismatch");

    auto shm = shm::Shm(shm_path, shm_size);
    assert(shm.size() == shm_size && "Size mismatch");
    assert(shm.file_path() == shm_path && "File path mismatch");

    // Move operator
    auto shm2 = shm::Shm("shm_test2", 2048);
    shm2 = std::move(shm);
    assert(shm2.size() == shm_size && "Size mismatch");
    assert(shm2.file_path() == shm_path && "File path mismatch");
    assert(shm.get() == nullptr && "Data mismatch");
    assert(shm.size() == 0 && "Size mismatch");
    assert(shm.file_path().empty() && "File path mismatch");
}

void semaphore_test()
{
    // Create a semaphore object
    auto const sem_name = "sem_test";
    auto const sem_count = 1;

    auto sem = shm::Semaphore(sem_name, sem_count);
    assert(sem.is_valid() && "Semaphore is not valid");
    sem.post();
    sem.wait();
    assert(sem.is_valid() && "Semaphore is not valid");
    sem.destroy();
    assert(!sem.is_valid() && "Semaphore is valid");
}

void semaphore_move_constructor_test()
{
    // Create a semaphore object
    auto const sem_name = "sem_test";
    auto const sem_count = 1;

    auto sem = shm::Semaphore(sem_name, sem_count);
    assert(sem.is_valid() && "Semaphore is not valid");

    sem.post();
    sem.wait();
    assert(sem.is_valid() && "Semaphore is not valid");

    // Move constructor
    auto sem2 = std::move(sem);
    assert(sem2.is_valid() && "Semaphore is not valid");
    assert(!sem.is_valid() && "Semaphore is valid");

    sem2.post();
    sem2.wait();
    assert(sem2.is_valid() && "Semaphore is not valid");

    sem2.destroy();
    assert(!sem2.is_valid() && "Semaphore is valid");
}

void semaphore_move_operator_test()
{
    // Create a semaphore object
    auto const sem_name = "sem_test";
    auto const sem_count = 1;

    auto sem = shm::Semaphore(sem_name, sem_count);
    assert(sem.is_valid() && "Semaphore is not valid");


    sem.post();
    sem.wait();
    assert(sem.is_valid() && "Semaphore is not valid");

    // Move operator
    auto sem2 = shm::Semaphore("sem_test2", 2);
    sem2 = std::move(sem);
    assert(sem2.is_valid() && "Semaphore is not valid");
    assert(!sem.is_valid() && "Semaphore is valid");

    sem2.post();
    sem2.wait();
    assert(sem2.is_valid() && "Semaphore is not valid");

    sem2.destroy();
    assert(!sem2.is_valid() && "Semaphore is valid");
}

void handle_errorno_test()
{
    try
    {
        shm::handle_errorno(-1, "test");
    }
    catch (std::runtime_error const &e)
    {
        assert(e.what() && "No exception thrown");
    }

    try
    {
        shm::handle_errorno(0, "test");
        assert(true && "No exception thrown");
    }
    catch (std::runtime_error const &e)
    {
        assert(e.what() && "No exception thrown");
    }

}

int main()
{
    shm_test();
    shm_move_constructor_test();
    shm_move_operator_test();
    semaphore_test();
    semaphore_move_constructor_test();
    semaphore_move_operator_test();
    handle_errorno_test();
    fmt::print("All tests passed!\n");
    return 0;
}