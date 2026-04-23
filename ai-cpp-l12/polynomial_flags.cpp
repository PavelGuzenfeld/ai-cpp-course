// Auto-vectorisable 5-term polynomial evaluation over an L1-resident float
// array.  Compile three ways and compare the ns/iter on your CPU:
//
//   g++ -O2 -std=c++23 polynomial_flags.cpp -o poly_o2
//   g++ -O3 -std=c++23 -march=x86-64-v3 polynomial_flags.cpp -o poly_v3
//   g++ -O3 -ffast-math -std=c++23 -march=x86-64-v3 polynomial_flags.cpp -o poly_fast
//
// On GCC 14 the middle build is often *slower* than the baseline because the
// AVX2/FMA scheduler without associative-math permission can't vectorise the
// reduction.  The third build unlocks it and runs ~7× faster than `-O2`.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  constexpr int N = 1 << 12;  // 16 KiB — fits in L1d
  std::vector<float> a(N), b(N);
  std::mt19937 rng{42};
  std::uniform_real_distribution<float> d(0.1f, 1.f);
  for (int i = 0; i < N; ++i) {
    a[i] = d(rng);
    b[i] = d(rng);
  }

  constexpr int REPEAT = 200'000;
  auto t0 = std::chrono::steady_clock::now();

  float s = 0.f;
  for (int r = 0; r < REPEAT; ++r) {
    for (int i = 0; i < N; ++i) {
      float x = a[i], y = b[i];
      float p = ((((x * 0.1f + y) * 0.2f + x) * 0.3f + y) * 0.4f + x) * 0.5f + y;
      s += p;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  std::printf("total = %g  time_per_iter = %.2f ns\n",
              double(s), double(ns) / (REPEAT * N));
  return 0;
}
