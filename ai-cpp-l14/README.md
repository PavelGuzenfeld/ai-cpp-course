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

`CMakeLists.txt` builds it three ways:

- `device_mock_native` — the nanobind module, `-DDEVICE_MOCK_API`. Synthetic
  frames, no hardware. What CI runs.
- `device_real_native` — the nanobind module, `-DDEVICE_REAL_API`, linked
  against [`real_device_v4l2.cpp`](real_device_v4l2.cpp), which implements
  the vendor API against a `/dev/video*` node. Built everywhere; it is
  *opening* the device that fails without a camera, not building.
- `device_real_layout_check` — an `OBJECT` library, compile-only, proving
  `DeviceBuffer` type-checks against the real header's layout on whatever
  ABI you are targeting.

### The two `DeviceBuffer`s must not share a name

Both modules load into one Python process, and the two compilations of
`DeviceBuffer` have *different layouts* — its member is `MockDeviceFrame` in
one and `DeviceFrame` in the other. Two definitions of `::DeviceBuffer` with
different members is an ODR violation, and the linker will not tell you: it
picks one and both modules use it.

nanobind does tell you, at import:

```
RuntimeWarning: nanobind: type 'DeviceBuffer' was already registered!
```

The fix is the per-API namespace in `device_buffer.hpp` — `device_mock_api::`
and `device_real_api::` — so the two classes have distinct mangled names.
This is [L19](../ai-cpp-l19/)'s failure mode reached from a different
direction: there, one name with two *linkages*; here, one name with two
*layouts*. Both are silent without a tool that happens to be looking.

## Which of Your Mock Tests Are Real Tests?

The acceptance question for this lesson is "do the same tests pass against
the real path". Most of them cannot, and noticing why is the point.

A real camera does not produce 64×48 frames with an arithmetic fill pattern.
So `test_mocking.py` is split:

- `TestTheContractBothImplementationsOwe` is parametrized over every
  implementation and may not name a resolution or a pixel value. It asserts
  what the *API* promises: a nonzero resolution, a resolution stable across
  frames, timestamps that do not go backwards, a frame count that tracks
  reads.
- `TestMockDeviceFrames` asserts the mock's own synthetic fill pattern.
  Those are tests *of the mock*, and they are fine as long as nobody mistakes
  them for tests of the code under it.

Adding an implementation means adding one entry to `IMPLEMENTATIONS`, not
writing a parallel test file. If a contract test fails against the real
device, either the implementation is wrong or the contract was never really
the contract — you were describing the mock.

## Skip, Never Stub

`docs/extending.md`'s rule: hardware-only components are **skipped, never
stubbed**, on the mock build. `test_mocking.py::TestRealDevice` is marked
`@pytest.mark.skipif` on the absence of a real device node — it does not
run, and does not silently report success. A stub that returns a plausible
answer is worse than a skip: a skip is honest about what wasn't tested; a
green stub isn't.

The marker is `/dev/video*`, a node that exists when a camera does. That
matters: an earlier version of this lesson gated on an invented path that no
driver ever creates, so the test was unreachable on *every* machine including
the hardware it named. A skip you can never turn into a run is a deleted test
with extra steps.

And a skip nobody turns into a run is the same thing more slowly. CI now
`modprobe`s `vivid`, the kernel's virtual video test driver — it ships in
`linux-modules-extra-$(uname -r)`, costs about 46 s to install on a GitHub
runner, and gives a `/dev/video0` that answers `QUERYCAP`, `REQBUFS`, `QBUF`
and `DQBUF` for real. The container gets it with `--device`. So the real path
is exercised on every push instead of on whatever day someone next plugs a
camera in. If the `modprobe` fails the job still runs and warns; `pytest -rs`
names the skip in the log, because the quiet version of that is how the path
went unexecuted for the whole life of the lesson.

Note also `TestTheRealModuleWithoutRealHardware`, which is **not** skipped.
Opening a nonexistent node, and opening `/dev/null` — which opens fine and
then fails `VIDIOC_QUERYCAP` — both need to raise, and neither needs a
camera. Without those, the real implementation's error paths would be
untested on every machine that lacks a device, which is most of them. Ask of
any hardware-gated suite: what can I still test without the hardware?

## A Fix With No Test, And Why

`device_open` used to gate on `v4l2_capability.capabilities`. That field is the
union over every node of the physical device, not the node you opened. A UVC
camera exposes a capture node and a metadata node; on both cameras here the
metadata node reads `capabilities=0x84a00001` — capture bit set — while its own
`device_caps=0x04a00000` does not have it. `device_caps` is the per-node field
and is valid exactly when `V4L2_CAP_DEVICE_CAPS` is set, so that is what the
code reads now.

No test covers the change, because none can. Flip the field back and the
metadata node is still rejected: `VIDIOC_G_FMT` with `V4L2_BUF_TYPE_VIDEO_CAPTURE`
returns `EINVAL` on it three lines later. Capture-capable and `G_FMT`-capable are
the same set on every node reachable here, so the caps check is redundant with
the ioctl that follows it and no observable behaviour moves.

A test written against the old field would have passed either way — a test of
`G_FMT`, named after the caps check. That is the same trap as a test that only
passes against the mock, and the honest move is not to write it. Reading the
wrong field is still worth fixing: the next ioctl added above `G_FMT` is the one
that would have paid for it.

## What Running It on a Camera Actually Found

This suite skipped on every machine the lesson was written on, so the real
path shipped unexecuted. Pointed at two USB cameras it failed six tests, for
two unrelated reasons.

**The `DQBUF` wait was a timeout with no time in it.** `device_read_frame`
spun `100000` times on `EAGAIN` and gave up — about 24 ms on the box that
wrote it. Measured time to a frame:

| device | negotiated format | first frame | steady state |
|---|---|---|---|
| USB2.0 HD UVC WebCam | MJPG 1280×720 | 1592 ms | 200 ms |
| Logitech C925e | YUYV 640×480 | 72 ms | 32 ms |

Both cameras, first frame and steady state, need more than the budget allowed:
the real path failed 100% of the time, not intermittently. A spin count is not
a duration. `poll()` with a timeout in milliseconds is, and the number has a
measurement behind it.

The failure cascaded misleadingly, too. Only the first real test reported
`device_read_frame failed`; every later one reported `device_open failed` —
pytest keeps the failing test's traceback, the traceback keeps the temporary
`RealDevice` alive, and the node stays open. The second symptom has nothing to
do with opening devices.

**The payload-varies test was a description of the mock.** It asserted that
`data_checksum` differs across five frames, which the mock guarantees by
construction. On the MJPG camera those 64 bytes are not pixels: they are
`FF D8`, `FF C0`, and the `FF DB` quantization-table marker at byte 21. They
move only when the encoder's rate control does. During auto-exposure
convergence that is every other frame, so the test passed — let the camera
settle and the same 64 bytes repeat for 82 consecutive frames, and it fails.

The contract never promised a varying payload; the mock did. That test now
asserts five reads return five distinct timestamps — a property of the API,
and still a catch for the bug the old one was aiming at, a buffer handed back
without being re-queued.

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
- A skipped test is an unexecuted test: this one's real path was wrong in two
  ways that only a camera could show, and a green suite said nothing about it
- A timeout counted in loop iterations is not a timeout — the number has to be
  in a unit you can measure the hardware against
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
| [real_device_header.h](real_device_header.h) | The vendor-style API: fixed layout, three C entry points |
| [real_device_v4l2.cpp](real_device_v4l2.cpp) | Real implementation of that API against a `/dev/video*` node |
| [device_real_native.cpp](device_real_native.cpp) | nanobind module: the real device, skipped when no camera |
| [mock_device_header.h](mock_device_header.h) | Mock with a `static_assert`-verified matching layout |
| [mock_device_header_broken.h](mock_device_header_broken.h) | Exercise 4: a deliberately wrong mock |
| [device_buffer.hpp](device_buffer.hpp) | One implementation, compiled against either header |
| [device_mock_native.cpp](device_mock_native.cpp) | nanobind module: the mock device, what CI tests |
| [device_real_layout_check.cpp](device_real_layout_check.cpp) | `-DDEVICE_REAL_API` compile-only layout check |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [test_mocking.py](test_mocking.py) | Unit tests: mock frames, skip-not-stub |
| [test_integration_mocking.py](test_integration_mocking.py) | A small capture loop over the mock device |
