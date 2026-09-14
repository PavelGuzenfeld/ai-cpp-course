#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/shm_lifecycle.hpp"
#include "safe-shm/shared_memory.hpp"
#include <sys/wait.h>
#include <thread>

TEST_CASE("shm_exists detects created segment")
{
    auto name = "test_lifecycle_exists";

    // Clean up first
    safe_shm::shm_remove(name);
    CHECK_FALSE(safe_shm::shm_exists(name));

    {
        safe_shm::OwnedShm<int> owned(name);
        CHECK(safe_shm::shm_exists(name));
    }
    // After destruction, shm::Shm unlinks
}

TEST_CASE("shm_remove on non-existent segment")
{
    CHECK_FALSE(safe_shm::shm_remove("test_lifecycle_nonexistent_xyz"));
}

TEST_CASE("shm_list finds created segments")
{
    constexpr auto prefix = "test_lifecycle_list_";
    auto name1 = std::string(prefix) + "alpha";
    auto name2 = std::string(prefix) + "beta";

    // Clean up
    safe_shm::shm_remove(name1);
    safe_shm::shm_remove(name2);

    safe_shm::OwnedShm<int> a(name1);
    safe_shm::OwnedShm<int> b(name2);

    auto list = safe_shm::shm_list(prefix);
    CHECK(list.size() >= 2);

    bool found_alpha = false, found_beta = false;
    for (auto const &n : list)
    {
        if (n == name1)
            found_alpha = true;
        if (n == name2)
            found_beta = true;
    }
    CHECK(found_alpha);
    CHECK(found_beta);
}

TEST_CASE("ShmHeader satisfies FlatType")
{
    CHECK(safe_shm::FlatType<safe_shm::ShmHeader>);
    CHECK(sizeof(safe_shm::ShmHeader) == 32);
}

TEST_CASE("OwnedShm initializes header correctly")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_header");

    auto const &hdr = owned.header();
    CHECK(hdr.magic == safe_shm::SHM_HEADER_MAGIC);
    CHECK(hdr.version == safe_shm::SHM_HEADER_VERSION);
    CHECK(hdr.writer_pid == static_cast<int32_t>(getpid()));
    CHECK(hdr.heartbeat_ns > 0);
}

TEST_CASE("OwnedShm data access")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_data");

    // Initially zero
    CHECK(owned.data() == 0);

    // Write
    owned.data() = 42;
    CHECK(owned.data() == 42);
}

TEST_CASE("OwnedShm with struct")
{
    struct Point
    {
        double x, y, z;
    };

    safe_shm::OwnedShm<Point> owned("test_lifecycle_struct");

    owned.data() = {1.0, 2.0, 3.0};
    CHECK(owned.data().x == doctest::Approx(1.0));
    CHECK(owned.data().y == doctest::Approx(2.0));
    CHECK(owned.data().z == doctest::Approx(3.0));
}

TEST_CASE("is_writer_alive with current process")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_alive");

    CHECK(safe_shm::is_writer_alive(owned.header()));
}

TEST_CASE("is_writer_alive with dead process")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_dead");

    // Forge a dead PID
    auto &hdr = owned.header();
    hdr.writer_pid = 999999; // unlikely to exist
    CHECK_FALSE(safe_shm::is_writer_alive(hdr));
}

TEST_CASE("is_writer_alive with bad magic")
{
    safe_shm::ShmHeader hdr{};
    hdr.magic = 0xDEAD;
    CHECK_FALSE(safe_shm::is_writer_alive(hdr));
}

TEST_CASE("is_writer_alive heartbeat check")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_heartbeat");

    // Fresh heartbeat — should be alive
    owned.update_heartbeat();
    CHECK(safe_shm::is_writer_alive(owned.header(), 1'000'000'000)); // 1s tolerance

    // Stale heartbeat — fake it
    owned.header().heartbeat_ns = 1; // epoch
    CHECK_FALSE(safe_shm::is_writer_alive(owned.header(), 1'000'000'000));
}

TEST_CASE("OwnedShm update_heartbeat advances time")
{
    safe_shm::OwnedShm<int> owned("test_lifecycle_hb_advance");

    auto t1 = owned.header().heartbeat_ns;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    owned.update_heartbeat();
    auto t2 = owned.header().heartbeat_ns;

    CHECK(t2 > t1);
}

TEST_CASE("OwnedShm cross-process liveness")
{
    constexpr auto name = "test_lifecycle_xproc";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::OwnedShm<int> owned(name);
        owned.data() = 123;
        // Sleep to let parent read
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        _exit(EXIT_SUCCESS);
    }

    // Let child initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Read header from parent
    safe_shm::SharedMemory<safe_shm::ShmHeader> reader(name);
    auto const &hdr = reader.get();
    CHECK(hdr.magic == safe_shm::SHM_HEADER_MAGIC);
    CHECK(safe_shm::is_writer_alive(hdr));

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));

    // After child exits, writer should be dead
    CHECK_FALSE(safe_shm::is_writer_alive(reader.get()));
}
