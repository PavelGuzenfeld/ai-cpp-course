# Lesson 12: Compiler Flags & clang-tidy — Enforcing Performance in CI

## Goal

Pick the right compiler flags, wire `clang-tidy` into CMake and GitHub Actions,
and turn the performance patterns from earlier lessons into *build failures*
when a teammate regresses them.

L6 taught you to measure. L8 taught you to move work to compile time. This
lesson teaches you to make the fast path the *only* path.

## Background: three concentric rings

Every production C++ codebase has three layers of enforcement:

1. **Source-level opt-ins** — `__builtin_*` intrinsics, `[[likely]]`/`[[unlikely]]`,
   `std::popcount`, `std::byteswap`. One programmer types the right call and
   the compiler emits the right instruction. Covered in L8.
2. **Project-level flags** — `-O2`, `-march=x86-64-v3`, `-ffast-math`,
   `-Wpessimizing-move`. Every file in the binary inherits these. Covered here.
3. **Team-level guardrail** — `clang-tidy` configured via `.clang-tidy`, run on
   every PR via CI. Catches regressions before they merge. Covered here.

## Part A: Compiler flags that matter

### A.1 Optimisation levels (`-O0` … `-Ofast`)

| Flag | Purpose | Use when |
|------|---------|----------|
| `-O0` | No optimisation, every variable on stack | Debugging with `gdb` |
| `-Og` | `-O1` with debugger-friendly choices | Dev builds |
| `-O2` | Production default | Everything that ships |
| `-O3` | + larger inlining budget, + loop unrolling | Hot paths; measure first |
| `-Os` | Smallest code, avoids text growth | Embedded, i-cache-bound |
| `-Ofast` | `-O3 -ffast-math -fno-signed-zeros` ... | Breaks IEEE-754 for speed |

`-O2` is the right default for 95% of production code. `-O3` is *not always
faster* — the bigger inlining and unrolling budgets can overflow the i-cache
in large binaries.

### A.2 Microarchitecture levels (`-march`)

By default, GCC targets the lowest-common-denominator `x86-64` ISA (SSE2, no
AVX). On any server or workstation made since 2013 this leaves real silicon
idle.

| Level | Roughly | Features |
|-------|---------|----------|
| `x86-64` | 2003 | SSE2 |
| `x86-64-v2` | Nehalem (2008) | SSE3, SSSE3, SSE4.1, SSE4.2 |
| `x86-64-v3` | Haswell (2013) | AVX, AVX2, BMI1, BMI2, FMA |
| `x86-64-v4` | Skylake-X (2017) | AVX-512 |
| `native` | This CPU | Whatever `-march=native` detects |

**A gotcha worth measuring yourself:** `-march=x86-64-v3` *without*
`-ffast-math` can be slower than `-O2` on a float reduction, because AVX2
FMA-scheduled without associativity doesn't help a sequential sum. `polynomial_flags.cpp`
in this lesson reproduces the effect.

### A.3 `-ffast-math` — the 2× that ships with a warning

`-ffast-math` bundles six sub-flags, most importantly:

- `-fassociative-math` — sums can reorder (unlocks reduction SIMD)
- `-ffinite-math-only` — assume no NaN/Inf
- `-fno-signed-zeros` — treat `-0 == 0`

When it pays: isolated compute kernels where inputs are bounded.
When it doesn't: anywhere `NaN` has a real meaning, or sums must match
bit-for-bit across runs.

### A.4 `-fno-exceptions` / `-fno-rtti`

Chromium and LLVM both ship with both disabled. The runtime speed gain is
small; the *binary size* reduction is 10–25%, which reduces i-cache pressure
across the whole application. Caveat: a library compiled with exceptions
cannot safely be linked against code compiled without them.

### A.5 Warning flags that catch performance bugs

| Flag | Catches |
|------|---------|
| `-Wpessimizing-move` | `return std::move(local);` — defeats NRVO |
| `-Wrange-loop-construct` | `for (auto x : vec)` — silent copy |
| `-Wexit-time-destructors` | Globals with non-trivial destructors |
| `-Wglobal-constructors` | Globals with non-trivial constructors |
| `-Wswitch-enum` | Missing case in `switch` over enum |
| `-Wnrvo` (GCC 13+) | Spot where NRVO would have fired but didn't |

Add them to your `-Wall -Wextra -Werror` stack. Every warning is a bug that
doesn't land.

## Part B: clang-tidy — the guardrail

`clang-tidy` ships with LLVM, has ~400 named checks, and is the team-level
way to enforce the patterns from earlier lessons. The `.clang-tidy` file in
this directory is a drop-in config you can use on any project.

### B.1 The checks that matter for performance

| Check | Catches | Earlier lesson |
|-------|---------|----------------|
| `performance-unnecessary-value-param` | Non-trivial type by value, used as const-ref | L4 string_view |
| `performance-unnecessary-copy-initialization` | Silent copy from const-returning member | L5 buffers |
| `performance-for-range-copy` | `for (auto x : ...)` over non-trivial container | — |
| `performance-inefficient-vector-operation` | `emplace_back` in loop without `reserve` | L5 pre-alloc |
| `performance-noexcept-move-constructor` | Missing `noexcept` — vector falls back to copy | L5 buffers |
| `performance-move-const-arg` | `std::move` on const — silently copies | — |
| `modernize-use-emplace` | `push_back(T{args...})` — use `emplace_back` | L5 |
| `modernize-pass-by-value` | `const T&` in ctor for a will-own member | L5 |
| `bugprone-implicit-widening-of-multiplication-result` | Overflow before widening | L11 |

### B.2 CMake integration (two lines)

```cmake
find_program(CLANG_TIDY_EXE clang-tidy REQUIRED)
set(CMAKE_CXX_CLANG_TIDY
    ${CLANG_TIDY_EXE}
    -warnings-as-errors=*
    --extra-arg=-std=c++23)
```

CMake then invokes `clang-tidy` as part of every `add_library` /
`add_executable` target. No custom targets, no `.tidy` files. Opt a target
*out* with `set_target_properties(<tgt> PROPERTIES CXX_CLANG_TIDY "")`.

### B.3 The `dirty_code.cpp` ↔ `clean_code.cpp` exercise

`dirty_code.cpp` deliberately contains five `performance-*` / `modernize-*`
violations. `clean_code.cpp` is the fixed version. Run:

```bash
clang-tidy dirty_code.cpp -p build --warnings-as-errors=*
```

and watch the errors fire. Apply the fixes yourself, then compare against
`clean_code.cpp`.

### B.4 The GitHub Actions workflow

`.github/workflows/clang-tidy.yml` is a minimal job that runs `clang-tidy` on
the files a PR changed. Drop it into any project:

```yaml
name: clang-tidy
on: [pull_request]
jobs:
  lint:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - run: sudo apt-get update && sudo apt-get install -y clang-tidy-19
      - run: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      - run: |
          git diff --name-only --diff-filter=AM origin/main...HEAD \
            | grep -E '\.(cpp|h|hpp)$' \
            | xargs -r clang-tidy-19 -p build --warnings-as-errors=*
```

## Exercises

1. **Flag matrix.** Run `./run_benchmarks.sh` which builds `polynomial_flags.cpp`
   three ways: `-O2`, `-O3 -march=x86-64-v3`, `-O3 -ffast-math -march=x86-64-v3`.
   Confirm the v3-without-ffast-math regression on your CPU. Document the
   crossover point.
2. **clang-tidy fixes.** Run clang-tidy on `dirty_code.cpp`, list every
   warning, fix them one by one until the build passes. Compare your fix to
   `clean_code.cpp`.
3. **Enable the guardrail on L4.** Copy `.clang-tidy` into `ai-cpp-l4/`, add
   the CMake stanza to `L4/CMakeLists.txt`, and run the build. Fix any
   violations it surfaces (expect 1–3).
4. **Wire up CI.** Copy `.github/workflows/clang-tidy.yml` to your own
   project. Open a deliberately bad PR (insert `for (auto x : vec)` for a
   vector of strings) and confirm CI fails.

## References

This lesson compiles patterns from two blog posts:

- *C++ Low-Latency Patterns, Benchmarked* — the 15 patterns these tools
  enforce.
- *C++ Low-Latency, Enforced: __builtin_*, Compiler Flags, and clang-tidy* —
  the full writeup with nanobench numbers for every technique.

Both at [pavelguzenfeld.com](https://pavelguzenfeld.com/).

## Files in this directory

| File | Purpose |
|------|---------|
| `polynomial_flags.cpp` | Auto-vectorisable kernel for the flag-matrix exercise |
| `dirty_code.cpp` | Deliberately contains 5 `performance-*` violations |
| `clean_code.cpp` | The fixed version — use to check your work |
| `.clang-tidy` | Drop-in config enabling `performance-*` + `modernize-*` |
| `CMakeLists.txt` | Builds all three binaries with `CMAKE_CXX_CLANG_TIDY` wired in |
| `run_benchmarks.sh` | Compiles `polynomial_flags.cpp` three ways and times each |
| `clang-tidy.yml` | GitHub Actions workflow template (copy to `.github/workflows/`) |
