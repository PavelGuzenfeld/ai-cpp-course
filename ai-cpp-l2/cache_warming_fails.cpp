// Cache warming is a HFT-style pattern where you pre-read the data the
// hot path will touch so L1/L2 already hold the relevant lines when it
// fires.  The pattern only pays when the warm-up work is amortised across
// an outer loop; at the micro-level it is usually a net loss because the
// warm-up itself is expensive.
//
// This file reproduces the counter-example from pavelguzenfeld.com —
// warming a 16 MiB array to serve a 64 KiB working set is ~7x slower
// than just doing the reads cold.
//
// Build and run:
//   g++ -O2 -std=c++23 cache_warming_fails.cpp -o cache_warming_fails
//   ./cache_warming_fails

#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// Volatile sink prevents the compiler from eliminating the reduction loops
// once it proves the result is unused.  Without it, -O2 can strip everything.
static volatile long long g_sink = 0;

int main() {
  constexpr std::size_t kN = 1u << 22;   // 16 MiB — bigger than L2
  constexpr std::size_t kK = 1u << 16;   // 64k random reads (= hot path)
  constexpr int kReps = 20;

  std::vector<int> data(kN);
  std::iota(data.begin(), data.end(), 0);

  std::mt19937 rng{42};
  std::uniform_int_distribution<std::size_t> dist(0, kN - 1);
  std::vector<std::size_t> idx(kK);
  for (auto& x : idx) x = dist(rng);

  // --- Cold: just do the reads ---
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < kReps; ++r) {
    long long cold_sum = 0;
    for (auto i : idx) cold_sum += data[i];
    g_sink = cold_sum;   // forces the loop to be emitted
  }
  auto t1 = std::chrono::steady_clock::now();
  auto cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / kReps;

  // --- Warm: first touch all N ints, then do the random reads ---
  auto t2 = std::chrono::steady_clock::now();
  for (int r = 0; r < kReps; ++r) {
    long long warm = 0;
    for (std::size_t i = 0; i < kN; ++i) warm += data[i];    // warm-up
    g_sink = warm;
    long long warm_sum = 0;
    for (auto i : idx) warm_sum += data[i];                  // hot path
    g_sink = warm_sum;
  }
  auto t3 = std::chrono::steady_clock::now();
  auto warm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / kReps;

  std::printf("cold: %lld ns/iter (K=%zu random reads)\n", (long long)cold_ns, kK);
  std::printf("warm: %lld ns/iter (N=%zu walk + K=%zu random reads)\n",
              (long long)warm_ns, kN, kK);
  std::printf("ratio warm/cold = %.2fx  %s\n",
              double(warm_ns) / double(cold_ns),
              (warm_ns > cold_ns) ? "(warming is a NET LOSS here)"
                                  : "(warming paid off)");
  return 0;
}
