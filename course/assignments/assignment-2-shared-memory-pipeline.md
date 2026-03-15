# Assignment 2: Shared Memory Image Pipeline

## Objective

Build a two-process image processing pipeline where a C++ producer writes
frames to shared memory and a Python consumer reads them with zero-copy access.

## Background

Lesson 3 introduced POSIX shared memory, the FlatType concept, and lock-free
patterns. This assignment puts them together in a realistic producer-consumer
pipeline similar to what runs on UAV tracking systems and Jetson edge devices.

## Requirements

### Part A: Producer Process (40 points)

Write a C++ program `frame_producer` that:

1. Creates a shared memory region using the `shm` library
2. Generates synthetic 640x480 RGB frames (gradient pattern with a moving box)
3. Writes frames to shared memory using a double-buffer pattern
4. Targets 30 FPS — measure and print actual achieved FPS
5. Runs for 300 frames (10 seconds), then exits cleanly

The frame struct must satisfy `FlatType`:
```cpp
struct Frame {
    uint64_t timestamp_ns;
    uint32_t frame_number;
    uint32_t width, height, channels;
    uint8_t data[640 * 480 * 3];
};
static_assert(std::is_trivially_copyable_v<Frame>);
```

### Part B: Consumer Process (40 points)

Write a Python script `frame_consumer.py` that:

1. Opens the same shared memory region (using nanobind bindings)
2. Reads frames as they arrive
3. Computes a simple metric per frame (e.g., mean pixel value, histogram)
4. Prints per-frame latency (time between producer write and consumer read)
5. Reports: average latency, max latency, frames received, frames dropped

### Part C: Latency Analysis (20 points)

Run both processes and produce a report:
- Average producer-to-consumer latency
- 99th percentile latency
- Comparison vs socket-based transfer (use `localhost` TCP for baseline)
- On Jetson: compare latency with and without CPU frequency scaling

## Deliverables

- `frame_producer.cpp` + `CMakeLists.txt`
- `frame_consumer.py`
- `benchmark_ipc.py` — latency comparison (shm vs TCP)
- `report.md` — analysis with numbers and conclusions

## Grading

| Criteria | Points |
|----------|--------|
| Producer runs at 30 FPS | 15 |
| Consumer receives all frames | 15 |
| Double-buffer prevents torn reads | 10 |
| Latency measurement is accurate | 15 |
| Clean shutdown (no shm leaks) | 10 |
| TCP baseline comparison | 15 |
| Report with analysis | 20 |
