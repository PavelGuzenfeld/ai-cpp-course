# Lesson 20: Numerical Robustness in Stateful Pipelines

## Goal

In a stateless function a NaN is one wrong answer for one call. In a filter,
a tracker, or any recurrent model it is permanent: the bad value is written
into the state, and every subsequent step reads it. This lesson is about
what you write into persistent state, not just what you detect.

## NaN Poisoning of Persistent State

```cpp
class NaiveFilter {
    double mean_ = 0.0;
public:
    void update(double x) { mean_ = alpha_ * x + (1 - alpha_) * mean_; }
};
```

`nan != nan`, and `finite + nan == nan` for every finite operand. One
non-finite sample writes `nan` into `mean_`, and every later `update()`
computes `finite * finite + finite * nan`, which is still `nan`. The filter
is permanently wedged after a single bad frame.

The fix is not "detect the NaN" — it's deciding what to do once you have:

```cpp
class CoastingFilter {
    double mean_ = 0.0;
public:
    void update(double x) {
        if (!std::isfinite(x)) return;   // coast: hold previous state
        mean_ = alpha_ * x + (1 - alpha_) * mean_;
    }
};
```

`test_integration_robustness.py` replays a 1000-frame stream with one
poisoned sample at frame 900: the naive filter is `nan` from frame 900
onward, the coasting filter is unaffected.

## Underflow That Looks Like a NaN Bug

`exp(x)` underflows to exactly `0.0` in double precision below roughly
`-745`. A mode-probability normalisation that computes `exp(log_likelihood)`
directly can see every mode underflow to `0.0` for a heavy-tailed outlier —
and then divide by a sum of `0.0`, producing `nan` from code that never
looked like it could.

```cpp
double naive_likelihood_sum(std::vector<double> const& ll) {
    double sum = 0.0;
    for (double x : ll) sum += std::exp(x);
    return sum;   // exactly 0.0 if every x < ~-745
}

double log_sum_exp(std::vector<double> const& ll) {
    double const max_ll = *std::max_element(ll.begin(), ll.end());
    double sum = 0.0;
    for (double x : ll) sum += std::exp(x - max_ll);
    return max_ll + std::log(sum);   // never underflows: the largest term is exp(0) = 1
}
```

`test_robustness.py` feeds both functions `[-800, -820, -900]` — real
log-likelihoods, not synthetic edge values — and shows `naive_likelihood_sum`
returns exactly `0.0` while `log_sum_exp` stays finite.

## `assert` Is Not a Guard

```cpp
int validate_dimension(int dim) {
    assert(dim > 0 && "dimension must be positive");
    return dim;
}
```

`assert` compiles to nothing under `-DNDEBUG` — which is what a CMake
`Release` build defines by default. A precondition check that reads like a
guard can ship with no check at all. `assert_example.cpp` is built twice
(see `CMakeLists.txt`): `assert_example_checked` aborts on a negative
dimension, `assert_example_ndebug` silently accepts it.

```bash
./install/lib/nanobind-l20/assert_example_checked -1   # aborts (SIGABRT)
./install/lib/nanobind-l20/assert_example_ndebug -1    # prints "validated dimension: -1"
```

Both variants behave identically on x86-64 and on aarch64 (Orin NX, JP6.2):
exit 134 from `SIGABRT` for the checked build, exit 0 and the printed `-1` for
the `NDEBUG` one. Unlike the cast below, this one really is portable.

## UB From an Out-of-Range Float-to-Int Cast

A non-finite `double` reaching a `static_cast<int>` is undefined behaviour,
not a large number:

```cpp
int cast_unsafely(double x) { return static_cast<int>(x); }
```

Build `ubsan_example` with `-DENABLE_SANITIZERS=ON` (mirrors
[L11](../ai-cpp-l11/)'s `asan_example`) and UBSan traps the cast at the
call site instead of returning garbage. `float-cast-overflow` is **not**
part of GCC's default `-fsanitize=undefined` group — it must be requested
explicitly, which is why `CMakeLists.txt` lists it on its own:

```bash
cd /workspace
mkdir -p build-l20 && cd build-l20
cmake ../ai-cpp-l20 -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug \
    -Dnanobind_DIR=/usr/local/nanobind/cmake
make ubsan_example
./ubsan_example
# ubsan_example.cpp:12:16: runtime error: nan is outside the range of
# representable values of type 'int'
# (process aborts: -fno-sanitize-recover makes this a hard stop, not a print-and-continue)
```

### The same UB, two different wrong answers

"Undefined" here is not a figure of speech, and the value you get is not even
consistent between the machines this course targets. Compiled `-O2` with no
sanitizer, the same casts print:

| expression | x86-64 (i7-12700H) | aarch64 (Orin NX, JP6.2) |
|---|---|---|
| `(int)NaN` | -2147483648 | 0 |
| `(int)+Inf` | -2147483648 | 2147483647 |
| `(int)-Inf` | -2147483648 | -2147483648 |
| `(int)1e300` | -2147483648 | 2147483647 |

x86-64 lowers the cast to `cvttsd2si`, which returns the "integer indefinite"
value `INT_MIN` for every invalid conversion. aarch64 lowers it to `fcvtzs`,
which saturates per case — NaN to 0, positive overflow to `INT_MAX`. Only the
`-Inf` row agrees, and it agrees by coincidence.

The 0 is the dangerous one. A NaN that arrives as `INT_MIN` tends to blow
something up quickly; a NaN that arrives as a plausible-looking 0 travels.

Denormals, by contrast, behave the same on both: neither flushes to zero at
`-O2`, and `numeric_limits<float>::has_denorm` is 1 on both. Flush-to-zero on
aarch64 is an FPCR setting, not a default — you get it from `-ffast-math`,
which is the next section's problem.

## `-ffast-math` and NaN Handling

[L1](../ai-cpp-l1/) and [L2](../ai-cpp-l2/) both ship `-ffast-math` by
default for their SIMD and cache-locality benchmarks. `-ffast-math` permits
the compiler to assume no NaNs and no infinities appear, which means
`std::isfinite` and `x != x` checks can be **optimised away** under that
flag. This lesson's module does not enable `-ffast-math` — the NaN-detection
code above depends on IEEE 754 semantics being honoured, and mixing the two
silently reintroduces the exact bug this lesson fixes.

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l20
source install/setup.bash

pytest ai-cpp-l20/ -v
```

## What You Learned

- A recursive filter's state is permanent: a single non-finite input, once
  written, poisons every future step
- Coast-don't-update is the fix — hold the previous state instead of writing
  a non-finite value into it
- `exp()` of a sufficiently negative log-likelihood underflows to exactly
  `0.0`, which turns a normalisation into a division by zero
- Log-space accumulation with max-subtraction (log-sum-exp) never forms that
  zero
- `assert` is a debug-only check, compiled to nothing under `-DNDEBUG`
- Casting a non-finite `double` to `int` is undefined behaviour, and UBSan
  is how you find it before a user does
- `-ffast-math` (already default in L1/L2) can optimise away the exact
  finiteness checks this lesson relies on

## Exercises

1. **Find the poisoning frame**: given a `NaiveFilter` that has gone `nan`
   after processing a long stream, write the diagnostic that finds the exact
   frame that poisoned it (`test_integration_robustness.py` has a worked
   version — write yours first).

2. **Coast vs. reset**: `CoastingFilter` holds the previous value on a bad
   sample. Implement a variant that resets to a known default instead, and
   argue for which recovery policy a tracker should use.

3. **A second underflow**: find an input to `naive_likelihood_sum` that
   underflows even though no single term looks obviously extreme (hint:
   the exponent threshold is per-term, not per-sum).

4. **NDEBUG in the wild**: `grep -rn NDEBUG` across this repo's `CMakeLists.txt`
   files. Which lessons build Release by default, and therefore already
   compile out every plain `assert`?

## Lesson Files

| File | Description |
|------|-------------|
| [robustness_native.cpp](robustness_native.cpp) | `NaiveFilter`/`CoastingFilter`, naive vs. log-sum-exp |
| [assert_example.cpp](assert_example.cpp) | Built twice: assertions active vs. `-DNDEBUG` |
| [ubsan_example.cpp](ubsan_example.cpp) | Float-to-int UB, trapped with `-DENABLE_SANITIZERS=ON` |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [test_robustness.py](test_robustness.py) | Unit tests: NaN poisoning, coasting, underflow |
| [test_integration_robustness.py](test_integration_robustness.py) | 1000-frame stream, one poisoned sample, diagnostic |
