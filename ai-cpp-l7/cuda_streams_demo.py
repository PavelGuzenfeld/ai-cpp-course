#!/usr/bin/env python3
"""
CUDA Streams Overlap Demo — Practical addition to Lesson 7.

Demonstrates how CUDA streams allow overlapping data transfer and compute,
a critical optimisation in tracker_engine's GPU pipeline.

Uses PyTorch operations as a proxy:
  - .cuda() / .cpu()  for host<->device transfer
  - torch.matmul       for compute

Run:
    python cuda_streams_demo.py
"""

import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: torch is required.  pip install torch")
    sys.exit(1)

if not torch.cuda.is_available():
    print("No CUDA GPU detected — running a CPU-only simulation to show the concept.")
    print("On a real GPU the stream overlap gives measurable speedup.\n")
    HAS_CUDA = False
else:
    HAS_CUDA = True
    GPU = torch.device("cuda:0")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_CHUNKS = 8         # number of data chunks to process
MATRIX_DIM = 1024      # square matrix side (controls compute time)
TRANSFER_DIM = 4096    # size of transfer payload


# ---------------------------------------------------------------------------
# CPU-only simulation (illustrates the concept without a GPU)
# ---------------------------------------------------------------------------

def cpu_simulate_sequential(chunks: list[torch.Tensor]) -> float:
    """Simulate: transfer -> compute -> transfer -> compute (no overlap)."""
    t0 = time.perf_counter()
    for chunk in chunks:
        # "transfer to device"
        _ = chunk.clone()
        # "compute"
        _ = torch.matmul(chunk, chunk)
        # "transfer back"
        _ = chunk.clone()
    return (time.perf_counter() - t0) * 1000.0


def cpu_simulate_pipelined(chunks: list[torch.Tensor]) -> float:
    """
    Simulate pipelined processing.
    In a real GPU scenario, stream overlap hides transfer latency.
    Here we just show the structure.
    """
    t0 = time.perf_counter()
    # With real CUDA streams, transfer[i+1] overlaps with compute[i].
    # On CPU we can only illustrate the pattern.
    prev_result = None
    for i, chunk in enumerate(chunks):
        _ = chunk.clone()  # "transfer to device"
        result = torch.matmul(chunk, chunk)  # "compute"
        if prev_result is not None:
            _ = prev_result.clone()  # "transfer back" previous result
        prev_result = result
    if prev_result is not None:
        _ = prev_result.clone()
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Real GPU implementation
# ---------------------------------------------------------------------------

def gpu_sequential(chunks_cpu: list[torch.Tensor],
                   weight: torch.Tensor) -> float:
    """No streams: transfer -> compute -> transfer back, serially."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for chunk in chunks_cpu:
        # H2D
        d = chunk.to(GPU, non_blocking=False)
        # Compute
        out = torch.matmul(d, weight)
        # D2H
        _ = out.cpu()

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def gpu_streamed(chunks_cpu: list[torch.Tensor],
                 weight: torch.Tensor,
                 num_streams: int = 2) -> float:
    """
    With streams: overlap transfer[i+1] with compute[i].

    Stream 0: H2D[0] -> compute[0] -> D2H[0]
    Stream 1:           H2D[1] -> compute[1] -> D2H[1]
    Stream 0:                      H2D[2] -> compute[2] -> D2H[2]
    ...
    """
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # Pin memory for async transfers
    pinned_chunks = []
    for c in chunks_cpu:
        pinned = torch.empty_like(c, pin_memory=True)
        pinned.copy_(c)
        pinned_chunks.append(pinned)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    d_inputs = [None] * num_streams
    d_outputs = [None] * num_streams
    results_cpu = [None] * len(pinned_chunks)

    for i, chunk in enumerate(pinned_chunks):
        si = i % num_streams
        stream = streams[si]

        with torch.cuda.stream(stream):
            # H2D (async because source is pinned)
            d_inputs[si] = chunk.to(GPU, non_blocking=True)
            # Compute
            d_outputs[si] = torch.matmul(d_inputs[si], weight)
            # D2H (async)
            results_cpu[i] = d_outputs[si].cpu()

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("CUDA Streams Overlap Demo")
    print(f"  Chunks     : {NUM_CHUNKS}")
    print(f"  Matrix dim : {MATRIX_DIM}")
    print()

    if not HAS_CUDA:
        # CPU-only conceptual demo
        chunks = [torch.randn(MATRIX_DIM, MATRIX_DIM) for _ in range(NUM_CHUNKS)]

        ms_seq = cpu_simulate_sequential(chunks)
        ms_pipe = cpu_simulate_pipelined(chunks)

        print(f"{'Approach':<25} {'Time (ms)':>10}")
        print("-" * 37)
        print(f"{'Sequential (no streams)':<25} {ms_seq:>10.2f}")
        print(f"{'Pipelined (simulated)':<25} {ms_pipe:>10.2f}")
        print()
        print("NOTE: On CPU, pipelining shows no real overlap benefit.")
        print("      On a real GPU with CUDA streams, transfer and compute")
        print("      happen on separate hardware engines and truly overlap.")
        print()
        print("Pipeline structure with 2 streams:")
        print("  Stream 0: [H2D_0][compute_0][D2H_0]          [H2D_2][compute_2][D2H_2]")
        print("  Stream 1:        [H2D_1][compute_1][D2H_1]          [H2D_3]...")
        print("                    ^^^^^^^^^^^^^^^^^ overlapped!")
        return

    # --- Real GPU benchmark ---
    chunks_cpu = [torch.randn(TRANSFER_DIM, MATRIX_DIM) for _ in range(NUM_CHUNKS)]
    weight = torch.randn(MATRIX_DIM, MATRIX_DIM, device=GPU)

    # Warmup
    for _ in range(3):
        gpu_sequential(chunks_cpu, weight)
        gpu_streamed(chunks_cpu, weight)

    # Benchmark
    REPEATS = 10
    seq_times = []
    stream_times = []
    for _ in range(REPEATS):
        seq_times.append(gpu_sequential(chunks_cpu, weight))
        stream_times.append(gpu_streamed(chunks_cpu, weight))

    seq_times.sort()
    stream_times.sort()
    ms_seq = seq_times[REPEATS // 2]
    ms_stream = stream_times[REPEATS // 2]

    print(f"{'Approach':<30} {'Time (ms)':>10} {'Speedup':>10}")
    print("-" * 52)
    print(f"{'Sequential (no streams)':<30} {ms_seq:>10.2f} {'1.0x':>10}")
    print(f"{'2 CUDA streams (overlap)':<30} {ms_stream:>10.2f} {ms_seq/ms_stream:>9.2f}x")
    print()

    # Vary number of streams
    print("Effect of stream count:")
    print(f"  {'Streams':<10} {'Time (ms)':>10} {'Speedup':>10}")
    print(f"  {'-'*32}")
    for ns in [1, 2, 4, 8]:
        times = []
        for _ in range(REPEATS):
            times.append(gpu_streamed(chunks_cpu, weight, num_streams=ns))
        times.sort()
        ms = times[REPEATS // 2]
        print(f"  {ns:<10} {ms:>10.2f} {ms_seq/ms:>9.2f}x")

    print()
    print("Key takeaways:")
    print("  - CUDA streams allow H2D, compute, and D2H to overlap")
    print("  - Pinned (page-locked) memory is required for async transfers")
    print("  - 2 streams is usually enough to hide transfer latency")
    print("  - More streams help when individual operations are very short")
    print("  - tracker_engine uses this pattern in its TRT inference pipeline")
    print()


if __name__ == "__main__":
    main()
