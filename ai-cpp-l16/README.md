# Lesson 16: Zero-Copy IPC Across Processes, and Proving the Fast Path Is Live

## Goal

[L3](../ai-cpp-l3/) shares memory between threads in one process. Real
pipelines split work across *processes* — a capture process, an inference
process, a display process — and the naive way to move a frame between them
is to serialize it through a socket or pipe: a real memory copy, twice (once
into the kernel buffer, once out). This lesson is the technique that avoids
it, `SCM_RIGHTS` fd passing, plus the part people skip: **how do you know
the fast path is actually the one that ran?**

## The Naive Path: Copy Through a Socket

```python
send_bytes(socket_fd, frame_bytes)   # producer: copy into the kernel buffer
received = recv_bytes(socket_fd, n)  # consumer: copy out of it
```

Correct, portable, and it pays for the size of every frame, every time.

## The Fast Path: Hand Over the File Descriptor

A POSIX shared-memory segment (`shm_open` + `mmap`, the same mechanism
[L3](../ai-cpp-l3/) uses within one process) is backed by a file descriptor.
`SCM_RIGHTS` is a Unix-domain-socket control message that transfers an open
fd to another process — not a copy of its contents, the same kernel-held
object:

```cpp
// Producer: create the segment, write into it, hand the fd over.
ShmFrame frame("/my_frame", size);
frame.write(payload);
send_fd(client_socket, frame.fd());

// Consumer: receive the fd, map it. No shm_open by name, no copy.
int fd = recv_fd(server_socket);
ShmFrameView view(fd, size);
auto data = view.read();  // the same bytes the producer wrote, zero-copy
```

Only one small control message (`sendmsg`/`recvmsg` with `SCM_RIGHTS`)
crosses the socket regardless of frame size — the frame data never does.

## The Control Size: Proving You Removed the Copy, Not the Work

`benchmark_ipc.py` measures both paths at two sizes: a tiny 64×64 RGBA
control (16 KB) and a 1080p RGB frame (~6 MB). If zero-copy only wins on the
large frame, you've shown you avoided copying bytes. If it *also* wins by
roughly the same margin on the tiny control, the "win" is dominated by pure
synchronization overhead, not the copy you think you removed — measure both
sizes or you can't tell which one you're looking at.

## Falsification: How Would You Know If You Had Silently Fallen Back?

A benchmark that reports plausible zero-copy numbers proves nothing if the
code actually fell back to copying without telling you. `ipc_native`
instruments the real fast path:

```cpp
int zero_copy_transfer_count() noexcept;  // incremented once per recv_fd()
```

`test_integration_ipc.py::TestZeroCopyVerification` checks this directly:
the zero-copy consumer's count is always ≥1, and — the discriminating
half — the *copy* path's count is always exactly 0. A verification that
can't fail on the thing it's supposed to catch isn't verifying anything.

## Pixel-Perfect Roundtrip

Fast is not the same claim as correct. `test_integration_ipc.py` sends a
deterministic byte pattern through both paths across two real processes and
asserts the consumer's bytes are identical to what the producer wrote,
byte for byte.

## Build and Run

```bash
cd /workspace
colcon build --packages-select nanobind-l16
source install/setup.bash

pytest ai-cpp-l16/ -v

# Benchmark (CSV to stdout)
python3 ai-cpp-l16/benchmark_ipc.py
```

## What You Learned

- `SCM_RIGHTS` transfers an open file descriptor between processes — the
  receiver maps the same kernel object, not a copy
- A shared-memory segment's fd is the handle; passing it is what makes the
  transfer zero-copy, not the shared-memory segment by itself
- A control-size benchmark separates "we removed the copy" from "we removed
  the work" — measure both, not just the frame you care about
- A benchmark number is not proof the fast path ran; instrument the actual
  call site and assert on it, including that the *other* path's counter
  stays at zero
- Zero-copy and correct are different claims — verify both

## Exercises

1. **Break the verification on purpose**: change the consumer to always use
   `recv_bytes` regardless of `mode`. Does `test_zerocopy_mode_actually_receives_an_fd`
   catch it? What would a benchmark-only test suite have shown instead?

2. **Multiple consumers**: extend the producer to `send_fd` to two accepted
   clients. Do both consumers see the same data? What POSIX guarantee makes
   that true?

3. **Name what "zero-copy" covers**: `ShmFrame::write()` still does one
   `memcpy` from the producer's payload into the shared segment. Is that
   part covered by the term "zero-copy" here? Compare against how L3's
   README defines the boundary.

4. **Latency vs. size**: from the benchmark CSV, does `copy`'s
   `us_per_transfer` scale with frame size the way you'd expect from a
   `memcpy`-bound cost? Does zero-copy's?

## Lesson Files

| File | Description |
|------|-------------|
| [ipc_native.cpp](ipc_native.cpp) | `ShmFrame`/`ShmFrameView`, Unix-socket + `SCM_RIGHTS` primitives |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [_ipc_producer_proc.py](_ipc_producer_proc.py) / [_ipc_consumer_proc.py](_ipc_consumer_proc.py) | Two-process demo, both paths |
| [benchmark_ipc.py](benchmark_ipc.py) | Copy vs. zero-copy at two frame sizes, CSV output |
| [test_ipc.py](test_ipc.py) | Unit tests: primitives in-process |
| [test_integration_ipc.py](test_integration_ipc.py) | Two-process roundtrip and fast-path verification |

Depends on [L3](../ai-cpp-l3/) (shared memory) and [L15](../ai-cpp-l15/) (atomics/ordering).
