# Lesson 14: Mocking a Vendor C API So CI Runs on a Laptop

## Goal

Shipping to an edge device means writing code against a hardware vendor's C
header — a header you did not write, cannot change, and whose real
implementation lives in a `.so` that only exists on the target board. CI
runs on a laptop with no camera, no Jetson, no vendor SDK. This lesson is
how to test that code anyway, without lying to yourself about what you
tested.

## The Vendor Header

[`real_device_header.h`](real_device_header.h) stands in for a hardware
SDK's header: a struct layout and a set of `extern "C"` function
declarations you do not control.

```cpp
struct DeviceFrame {
    std::uint32_t width;
    std::uint32_t height;
    std::uint64_t timestamp_ns;
    std::uint8_t  data[64];
};
extern "C" {
    DeviceHandle* device_open(char const* path);
    int device_read_frame(DeviceHandle*, DeviceFrame*);
    void device_close(DeviceHandle*);
}
```

## The Mock, and Why Its Layout Must Match Exactly

[`mock_device_header.h`](mock_device_header.h) defines `MockDeviceFrame` —
same fields, same order, same size — and proves it at compile time:

```cpp
static_assert(sizeof(MockDeviceFrame) == sizeof(DeviceFrame), ...);
static_assert(offsetof(MockDeviceFrame, width)  == offsetof(DeviceFrame, width),  ...);
static_assert(offsetof(MockDeviceFrame, height) == offsetof(DeviceFrame, height), ...);
```

A mock that is merely "close" to the real struct is worse than no mock: it
passes its own tests and still misreads every field once real code runs
against it. [`mock_device_header_broken.h`](mock_device_header_broken.h)
swaps `width` and `height` — same `sizeof`, wrong offsets, and the
`static_assert` catches it before a single test runs (Exercise 4).

## One Implementation, Two Headers

[`device_buffer.hpp`](device_buffer.hpp) is the actual point of the lesson:
written once, it compiles against *either* header, selected by a `-D` flag:

```cpp
#if defined(DEVICE_MOCK_API)
  using ActiveFrame = MockDeviceFrame;
#elif defined(DEVICE_REAL_API)
  using ActiveFrame = DeviceFrame;
#endif

class DeviceBuffer {
    explicit DeviceBuffer(ActiveFrame const& frame);
    // ...
};
```

`CMakeLists.txt` builds it twice:

- `device_mock_native` — the nanobind module, `-DDEVICE_MOCK_API`. This is
  what CI actually builds, links, and tests.
- `device_real_layout_check` — an `OBJECT` library, `-DDEVICE_REAL_API`.
  There is no real vendor `.so` to link against on x86, so this target only
  *compiles* — proving `DeviceBuffer` type-checks against the real header's
  layout, without ever running.

## Skip, Never Stub

`docs/extending.md`'s rule: hardware-only components are **skipped, never
stubbed**, on the mock build. `test_mocking.py::TestRealDevice` is marked
`@pytest.mark.skipif` on the absence of a real device node — it does not
run, and does not silently report success. A stub that returns a plausible
answer is worse than a skip: a skip is honest about what wasn't tested; a
green stub isn't.

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l14
source install/setup.bash

pytest ai-cpp-l14/ -v
```

## What You Learned

- A mock's struct layout must match the real header exactly, or bugs in
  real code can slip past every mock test
- One implementation file can — and should — compile against both the mock
  and the real header, selected at build time, so the logic under test is
  never duplicated
- A `-D`-selected `OBJECT` library proves layout compatibility against a
  real vendor header with nothing to link against
- Skipping a hardware-only test is honest; stubbing it to pass is not
- Extracting the algorithm into a header-agnostic class (`DeviceBuffer`) is
  what makes any of this possible — a class that mixed hardware I/O with
  logic couldn't be tested this way at all

## Exercises

1. **Extend the mock**: add a `brightness` field to both headers, keep the
   `static_assert`s passing, and expose it through `DeviceBuffer`.

2. **A layout bug you introduce**: add a field to `MockDeviceFrame` only
   (not the real header) and watch the size `static_assert` fail. Which
   `static_assert` catches a missing field vs. a reordered one?

3. **Write the real path**: sketch (no need to compile against real
   hardware) what `device_open`/`device_read_frame`/`device_close` would
   look like calling into an actual vendor `.so` via `dlopen`.

4. **The broken mock**: temporarily rename `mock_device_header_broken.h` to
   `mock_device_header.h` (back up the original first) and rebuild. Which
   line does the compiler point to, and does it match your prediction from
   the file's comment?

## Lesson Files

| File | Description |
|------|-------------|
| [real_device_header.h](real_device_header.h) | Stand-in for a vendor SDK header |
| [mock_device_header.h](mock_device_header.h) | Mock with a `static_assert`-verified matching layout |
| [mock_device_header_broken.h](mock_device_header_broken.h) | Exercise 4: a deliberately wrong mock |
| [device_buffer.hpp](device_buffer.hpp) | One implementation, compiled against either header |
| [device_mock_native.cpp](device_mock_native.cpp) | nanobind module: the mock device, what CI tests |
| [device_real_layout_check.cpp](device_real_layout_check.cpp) | `-DDEVICE_REAL_API` compile-only layout check |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [test_mocking.py](test_mocking.py) | Unit tests: mock frames, skip-not-stub |
| [test_integration_mocking.py](test_integration_mocking.py) | A small capture loop over the mock device |
