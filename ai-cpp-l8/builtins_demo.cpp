// builtins_demo.cpp — C++20/23 standard-library wrappers for GCC/Clang
// builtins.  Every `std::` name in this file lowers to the same single CPU
// instruction as the corresponding `__builtin_*`.  Use the std:: name in new
// code for MSVC portability and constexpr-friendliness.
//
// Paired with `benchmark_builtins.py` which runs each pair and asserts the
// runtimes agree within noise.

#include <bit>              // std::popcount, std::countl_zero, std::byteswap (C++20/23)
#include <cstdint>
#include <utility>          // std::unreachable (C++23)
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// --- Popcount ---------------------------------------------------------------
// __builtin_popcountll(x) → popcntq on x86-64 with POPCNT (every CPU this decade).
// std::popcount is the C++20 <bit> wrapper.
long popcount_builtin(uint64_t x) { return __builtin_popcountll(x); }
long popcount_std(uint64_t x)     { return std::popcount(x); }

// --- Leading-zero count -----------------------------------------------------
// __builtin_clzll(x) → lzcnt or bsr.  UB on x == 0 for the builtin;
// std::countl_zero is well-defined on zero (returns 64 for uint64_t).
long clz_builtin(uint64_t x) { return x ? __builtin_clzll(x) : 64; }
long clz_std(uint64_t x)     { return std::countl_zero(x); }

// --- Byte swap --------------------------------------------------------------
// __builtin_bswap64 → single `bswap` instruction.  GCC already recognises
// the portable shift-OR idiom and folds it to the same instruction, so the
// runtime win is zero — but std::byteswap (C++23) is portable to MSVC and
// documents intent.
uint64_t bswap_builtin(uint64_t x) { return __builtin_bswap64(x); }
uint64_t bswap_std(uint64_t x)     { return std::byteswap(x); }

// --- Unreachable ------------------------------------------------------------
// Guaranteeing a switch default is never reached lets the compiler drop the
// bounds check.  __builtin_unreachable() is GCC/Clang; std::unreachable()
// is C++23.  Both are UB if ever actually reached — pair with a debug assert.
enum class Op : int { Add = 0, Sub = 1, Mul = 2, Xor = 3 };

int apply_builtin(Op op, int a, int b) {
  switch (op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Xor: return a ^ b;
  }
  __builtin_unreachable();
}

int apply_std(Op op, int a, int b) {
  switch (op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Xor: return a ^ b;
  }
  std::unreachable();       // C++23
}

// --- Batch helpers so the Python side has something loop-shaped to time ----
long popcount_sum_builtin(const std::vector<uint64_t>& v) {
  long s = 0; for (auto x : v) s += __builtin_popcountll(x); return s;
}
long popcount_sum_std(const std::vector<uint64_t>& v) {
  long s = 0; for (auto x : v) s += std::popcount(x); return s;
}

uint64_t bswap_xor_builtin(const std::vector<uint64_t>& v) {
  uint64_t a = 0; for (auto x : v) a ^= __builtin_bswap64(x); return a;
}
uint64_t bswap_xor_std(const std::vector<uint64_t>& v) {
  uint64_t a = 0; for (auto x : v) a ^= std::byteswap(x); return a;
}

NB_MODULE(builtins_demo, m) {
  m.doc() = "C++20/23 standard alternatives to GCC/Clang __builtin_* intrinsics";

  m.def("popcount_builtin", &popcount_builtin);
  m.def("popcount_std",     &popcount_std);
  m.def("clz_builtin",      &clz_builtin);
  m.def("clz_std",          &clz_std);
  m.def("bswap_builtin",    &bswap_builtin);
  m.def("bswap_std",        &bswap_std);
  m.def("apply_builtin", [](int op, int a, int b) {
    return apply_builtin(static_cast<Op>(op), a, b);
  });
  m.def("apply_std", [](int op, int a, int b) {
    return apply_std(static_cast<Op>(op), a, b);
  });
  m.def("popcount_sum_builtin", &popcount_sum_builtin);
  m.def("popcount_sum_std",     &popcount_sum_std);
  m.def("bswap_xor_builtin",    &bswap_xor_builtin);
  m.def("bswap_xor_std",        &bswap_xor_std);
}
