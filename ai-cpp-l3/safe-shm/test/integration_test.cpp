#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/dblbuf_loader.hpp"
#include "safe-shm/image.hpp"
#include "safe-shm/image_shm.hpp"
#include "safe-shm/producer_consumer.hpp"
#include "safe-shm/shared_memory.hpp"
#include "safe-shm/storage.hpp"
#include <sys/wait.h>

TEST_CASE("cross-process Storage/DblBufLoader round-trip")
{
    constexpr auto shm_name = "integ_store_load";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        // child: write
        safe_shm::Storage<int> storage(shm_name);
        storage.store(42);
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    // parent: read
    safe_shm::DblBufLoader<int> loader(shm_name);
    auto snapshot = loader.load();
    loader.wait();
    CHECK(*snapshot == 42);
}

TEST_CASE("cross-process struct transfer")
{
    struct Telemetry
    {
        double lat;
        double lon;
        float alt;
        uint32_t seq;
    };

    constexpr auto shm_name = "integ_struct";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::Storage<Telemetry> storage(shm_name);
        storage.store({32.0, -117.0, 150.5f, 1});
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::DblBufLoader<Telemetry> loader(shm_name);
    auto s = loader.load();
    loader.wait();
    CHECK(s->lat == doctest::Approx(32.0));
    CHECK(s->lon == doctest::Approx(-117.0));
    CHECK(s->alt == doctest::Approx(150.5f));
    CHECK(s->seq == 1);
}

TEST_CASE("DoubleBufferShm store-load cycle")
{
    using Image = safe_shm::img::ImageFHD_RGB;
    safe_shm::DoubleBufferShm<Image> shm("integ_dblbuf_img");

    auto img = std::make_unique<Image>();
    img->timestamp = 1000;
    img->frame_number = 1;
    std::fill(img->data.begin(), img->data.end(), 0xFF);

    shm.store(*img);
    auto snap = shm.load();
    shm.wait();

    CHECK(snap->timestamp == 1000);
    CHECK(snap->frame_number == 1);
    CHECK(snap->data[0] == 0xFF);
    CHECK(snap->data[snap->data.size() - 1] == 0xFF);
}

TEST_CASE("ProducerConsumer cross-process")
{
    constexpr auto shm_name = "integ_prodcons";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::ProducerConsumer<int> pc(shm_name);
        pc.produce(777);
        _exit(EXIT_SUCCESS);
    }

    safe_shm::ProducerConsumer<int> pc(shm_name);
    int received = 0;
    pc.consume([&received](int const &val) { received = val; });
    CHECK(received == 777);

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("SharedMemory cross-process read after write")
{
    constexpr auto shm_name = "integ_shmem";

    struct Data
    {
        int values[4];
    };

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::SharedMemory<Data> shm(shm_name);
        shm.get() = {{10, 20, 30, 40}};
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::SharedMemory<Data> shm(shm_name);
    CHECK(shm.get().values[0] == 10);
    CHECK(shm.get().values[1] == 20);
    CHECK(shm.get().values[2] == 30);
    CHECK(shm.get().values[3] == 40);
}
