# Lesson 18: Golden Oracles and Sanitizers on a Real Project

## Goal

Test numerical code whose correct answer you do not know in closed form, and
run ASan/UBSan/TSan on an actual Python extension module — not a standalone
toy binary — including the trap that catches almost everyone the first time.

## Build and Run

Inside the Docker container:

```bash
cd /workspace
colcon build --packages-select nanobind-l18
source install/setup.bash

pytest ai-cpp-l18/test_validation.py -v
pytest ai-cpp-l18/test_integration_validation.py -v
```

## Golden Oracles

`tracker_engine`'s hand-written kernels are checked against a reference
implementation that is *not the code under test* — that is the entire point.
Two flavours appear in production: a well-known CV library as the oracle for
a hand-written kernel, and an fp32 CPU runtime as the oracle for fp16
accelerated inference (matched to IoU 0.97–0.996, not exact equality).

`box_blur()` in `filter_native.cpp` is checked the same way in
`test_validation.py`: against a NumPy implementation that sums the kernel
window in a different order (`np.pad` + slicing) rather than a transcription
of the C++ loop. The tolerance is derived, not picked to pass:

```python
eps32 = np.finfo(np.float32).eps
tolerance = eps32 * (2 * radius + 1) ** 2 * 5
```

`float32` machine epsilon times the number of terms summed per pixel, times a
5x margin for the two implementations disagreeing on summation order. A
tolerance that isn't derived this way is a hole sized to whatever the
implementation happened to produce.

## Metric Choice: max-abs-diff vs mask-disagreement rate

From `docs/ANALYTICS_NO_OPENCV_PLAN.md`: max-abs-diff is right for linear
stages, but wrong for a thresholded one, because *"a hard nonlinearity flips
pixels on ~1e-4 float drift."* A continuous value can be correct to eleven
decimal places and still land on the wrong side of a hard cutoff.

`threshold_mask()` demonstrates this directly. `test_validation.py` places a
few pixels within float32 rounding distance of the threshold, then compares
against an oracle computed in float64. Those specific pixels are allowed to
disagree — that is float rounding, not a bug — but nothing else may:

```python
disagreeing_pixels = set(zip(*np.nonzero(disagreement)))
assert disagreeing_pixels <= set(boundary_pixels)
```

An assertion demanding the two masks match exactly would fail on every run,
for a reason that has nothing to do with correctness. Mask-disagreement rate,
bounded to the known boundary pixels, is the metric that actually tells you
whether the implementation is right.

## Sanitizers on a Python Extension Module

L11's ASan example is a standalone C++ binary — it never has to deal with
`dlopen`. A nanobind module is loaded into the Python interpreter *by*
`dlopen`, and that changes everything.

### The dlopen trap

```bash
colcon build --packages-select nanobind-l18 --cmake-args -DENABLE_SANITIZERS=ON
source install/setup.bash
python3 -c 'import filter_native'
```

```
==10==ASan runtime does not come first in initial library list; you should
either link runtime to your application or manually preload it with
LD_PRELOAD.
```

ASan's runtime has to be the first thing loaded into the process so it can
intercept every allocation from the start. `python3` is a plain executable
that was never linked against ASan — it only picks up ASan's runtime when the
sanitized `.so` is `dlopen`'d partway through, by which point it's too late.
The fix is `LD_PRELOAD`, not a CMake flag:

```bash
ASAN_LIB=$(g++-13 -print-file-name=libasan.so)
LD_PRELOAD=$ASAN_LIB ASAN_OPTIONS=detect_leaks=0 pytest ai-cpp-l18/ -v
```

`ASAN_OPTIONS=detect_leaks=0` is doing real work here, not just silencing
noise: `LD_PRELOAD`ing ASan into the interpreter's own environment
instruments the interpreter's own startup allocations (`PyUnicode_New`,
`type_new`, thread locks) too, and CPython never frees most of them before
`exit()` — LeakSanitizer reports hundreds of them as leaks on every run,
regardless of whether this lesson's code leaks anything. Disabling leak
detection does not disable the rest of ASan: heap-buffer-overflow,
use-after-free and the other error detectors it exists for are still fully
armed in this same run — only the "nothing outstanding at exit" check is
off. A real leak in `box_blur` or `parse_record` would still abort the run.

### Third-party findings: tag, don't suppress

`docs/validation.md`'s policy: *"Anything in our own code should be fixed,
not suppressed."* The interpreter-startup noise above is a case of this —
`detect_leaks=0` is a documented, named workaround for a known non-issue, not
a blanket "ignore whatever comes up." A closed vendor library that
double-locks its own global mutex, or an allocator that OOMs under
shadow-memory reservation, gets a **named, tagged** suppression entry saying
which library and why — never a suite-wide disable that would also hide a
real bug in code you own.

### TSan

```bash
colcon build --packages-select nanobind-l18 --cmake-args -DENABLE_TSAN=ON
source install/setup.bash
TSAN_LIB=$(g++-13 -print-file-name=libtsan.so)
LD_PRELOAD=$TSAN_LIB python3 -m pytest ai-cpp-l18/ -v
```

`test_integration_validation.py` calls `box_blur`, `threshold_mask` and
`parse_record` from 8 Python threads at once — each releases the GIL for its
C++ work (see the `nb::gil_scoped_release` scopes in `filter_native.cpp` and
`record_parser_native.cpp`), so this is real concurrent execution, not GIL-
serialized Python threads taking turns. TSan reports it clean: no shared
mutable state, no race.

Under Docker's default seccomp profile, TSan itself fails to start:

```
FATAL: ThreadSanitizer: encountered an incompatible memory layout but was
unable to disable ASLR (perhaps sandboxing is enabled?).
```

TSan needs to disable ASLR for its fixed shadow-memory layout, which the
default seccomp profile blocks. Run the container with
`--security-opt seccomp=unconfined` for the TSan lane.

### The GIL-release bug this lesson's own tests caught

`box_blur` and `threshold_mask` originally released the GIL for their entire
body via `nb::call_guard<nb::gil_scoped_release>()`, matching what looks like
the natural pattern. `test_integration_validation.py`'s concurrent-threads
test crashed the interpreter immediately — the `nb::capsule` and
`nb::ndarray<nb::numpy, ...>` return value are Python objects, and building
them while the GIL is released is itself a bug, not a false positive. The
fix, matching `ai-cpp-l15`/`ai-cpp-l16`'s convention: a manually-scoped
`nb::gil_scoped_release` around only the pure-compute loop, ending before any
Python object is constructed. Left in as `filter_native.cpp` now reads,
because it's the concrete version of "what breaks if this is called from two
threads" that L18's own interrogation pass should have asked before trusting
`call_guard` blindly.

## Fuzzing: Two Harnesses, One Real

`record_parser.hpp` parses a small binary header (magic, version, a
length field, a checksum) and is exactly the kind of code that should never
trust attacker-controlled input — `length` claims how many payload bytes
follow, and a parser that believes it without checking against what's
actually left in the buffer is a buffer overrun waiting for the right input.

**`test_validation.py`'s parametrized corpus is the deterministic sweep.** It
always runs the same fixed cases (empty, truncated header, bad magic, a
length that overruns the buffer, a bad checksum) through
`record_parser_native.parse_record`. It always passes once the parser is
correct, and it never gets any deeper than these six cases — it is a smoke
test, not a fuzzer.

**`fuzz/fuzz_parser.cpp`, built as `fuzz_parser_libfuzzer`, is the actual
fuzzer.** It links the same `record_parser.hpp` directly (no Python, no
nanobind) under `-fsanitize=fuzzer,address,undefined` and requires clang:

```bash
colcon build --packages-select nanobind-l18 --cmake-args -DCMAKE_CXX_COMPILER=clang++-19
./build/nanobind-l18/fuzz_parser_libfuzzer -max_total_time=60 /tmp/corpus
```

libFuzzer mutates its own inputs under coverage guidance — it found 33 edges
of coverage and 50 features from a handful of seed bytes in under a second in
local testing, discovering length- and checksum-boundary cases no one wrote
by hand. Reinterpreting raw fuzzer bytes as the header's fields (rather than
enumerating "what if length is 0xFFFF" by hand) is what lets it find NaN,
overflow and off-by-one cases organically.

## Exercises

1. **Break the checksum check**: comment out the checksum comparison in
   `record_parser.hpp` and rerun `fuzz_parser_libfuzzer`. How many iterations
   before it finds an input that used to be rejected and now parses?

2. **Widen the golden oracle**: add a `box_blur` radius-2 case to
   `test_validation.py`. Does the tolerance derivation still hold without
   changing the formula, or does the margin need adjusting?

3. **Break the GIL discipline on purpose**: move the `nb::capsule`
   construction in `box_blur` inside the `nb::gil_scoped_release` scope and
   rerun `test_integration_validation.py`. Confirm it crashes the same way
   the original bug did, then put it back.

## What You Learned

- A golden oracle is a second, independent implementation — never the code
  under test reimplemented to look different
- Tolerances come from a derivation (units, term count, error budget), not
  from whatever number happens to pass
- max-abs-diff is the wrong metric for a thresholded stage; use
  mask-disagreement rate and bound it to where drift is expected
- `dlopen`'d Python extensions need `LD_PRELOAD`, not a CMake flag, to run
  under ASan/TSan — and the interpreter's own startup allocations will look
  like leaks unless you know to expect it
- Tag known-external sanitizer findings by name; never blanket-suppress a
  whole class of check
- A deterministic in-suite sweep and a coverage-guided fuzzer are different
  tools — the sweep catches regressions, the fuzzer finds what you didn't
  think to write

## Lesson Files

| File | Description |
|------|-------------|
| [record_parser.hpp](record_parser.hpp) | Bounds-checked wire-format parser, shared by the nanobind module and the fuzzer |
| [record_parser_native.cpp](record_parser_native.cpp) | nanobind binding for the parser |
| [filter_native.cpp](filter_native.cpp) | box_blur (linear) and threshold_mask (nonlinear) stages |
| [fuzz/fuzz_parser.cpp](fuzz/fuzz_parser.cpp) | libFuzzer target for the parser |
| [CMakeLists.txt](CMakeLists.txt) | ENABLE_SANITIZERS / ENABLE_TSAN options, clang-only fuzz target |
| [test_validation.py](test_validation.py) | Golden oracle, mask-disagreement rate, deterministic fuzz sweep |
| [test_integration_validation.py](test_integration_validation.py) | Multi-threaded stress test, the TSan lane's actual target |
