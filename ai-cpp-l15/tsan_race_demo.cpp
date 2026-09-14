// Standalone executable, built with -DENABLE_TSAN=ON (mirrors
// ai-cpp-l11/asan_example and ai-cpp-l20/ubsan_example). Runs the same
// producer/consumer workload over both ring variants; ThreadSanitizer
// reports a data race only for the broken one.
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "spsc_ring.hpp"

namespace
{
    constexpr std::size_t kRingCapacity = 256;
    constexpr int kItems = 2000;

    template <typename Ring>
    std::vector<int> produce_and_consume()
    {
        Ring ring;
        std::vector<int> received;
        received.reserve(kItems);

        std::jthread producer([&]
                               {
            for (int i = 0; i < kItems; ++i) {
                while (!ring.push(i)) {
                    std::this_thread::yield();
                }
            } });

        std::jthread consumer([&]
                               {
            int count = 0;
            while (count < kItems) {
                if (auto value = ring.try_pop()) {
                    received.push_back(*value);
                    ++count;
                } else {
                    std::this_thread::yield();
                }
            } });

        return received;
    }
} // namespace

int main(int argc, char **argv)
{
    bool const broken = argc > 1 && std::strcmp(argv[1], "broken") == 0;

    std::vector<int> received = broken
                                     ? produce_and_consume<BrokenSpscRing<int, kRingCapacity>>()
                                     : produce_and_consume<SpscRing<int, kRingCapacity>>();

    bool ok = received.size() == static_cast<std::size_t>(kItems);
    for (int i = 0; ok && i < kItems; ++i)
    {
        ok = received[static_cast<std::size_t>(i)] == i;
    }

    std::printf("%s ring: %s\n", broken ? "broken" : "correct", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
