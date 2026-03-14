"""
GPU timing utilities — accurate measurement of CUDA operations
using torch.cuda.Event, with CPU fallback.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

# Check for CUDA availability at import time
_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None


@dataclass
class TimingResult:
    """Result of a GPU timing measurement."""
    name: str
    elapsed_ms: float
    device: str  # "cuda" or "cpu"

    def __repr__(self):
        return f"TimingResult({self.name}: {self.elapsed_ms:.3f} ms on {self.device})"


class GpuTimer:
    """Context manager for timing GPU operations.

    Uses torch.cuda.Event for accurate GPU timing when CUDA is available.
    Falls back to time.perf_counter_ns for CPU timing.

    Usage:
        with GpuTimer("inference") as t:
            output = model(input_tensor)
        print(t.result)  # TimingResult(inference: 12.345 ms on cuda)
    """

    def __init__(self, name: str = "unnamed", device: str | None = None):
        self.name = name
        self._result: TimingResult | None = None

        # Determine device
        if device is not None:
            self._use_cuda = (device == "cuda" and _HAS_CUDA)
        else:
            self._use_cuda = _HAS_CUDA

        # State for timing
        self._start_event = None
        self._end_event = None
        self._cpu_start_ns = 0

    @property
    def result(self) -> TimingResult | None:
        """Get the timing result after the context manager exits."""
        return self._result

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._result.elapsed_ms if self._result else 0.0

    def __enter__(self):
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self._result = TimingResult(
                name=self.name,
                elapsed_ms=elapsed_ms,
                device="cuda",
            )
        else:
            elapsed_ns = time.perf_counter_ns() - self._cpu_start_ns
            self._result = TimingResult(
                name=self.name,
                elapsed_ms=elapsed_ns / 1e6,
                device="cpu",
            )
        return False


class GpuTimerCollection:
    """Collect multiple GPU timing measurements."""

    def __init__(self):
        self._results: list[TimingResult] = []

    def timer(self, name: str, device: str | None = None) -> GpuTimer:
        """Create a GpuTimer that records into this collection."""
        return _CollectedGpuTimer(self, name, device)

    def record(self, result: TimingResult):
        """Add a timing result."""
        self._results.append(result)

    @property
    def results(self) -> list[TimingResult]:
        return list(self._results)

    def total_ms(self) -> float:
        return sum(r.elapsed_ms for r in self._results)

    def print_summary(self):
        """Print a summary of all collected timings."""
        if not self._results:
            print("No GPU timings recorded.")
            return

        print()
        print(f"{'Operation':<25} {'Time (ms)':>10} {'Device':>8}")
        print("-" * 45)
        for r in self._results:
            print(f"{r.name:<25} {r.elapsed_ms:>10.3f} {r.device:>8}")
        print("-" * 45)
        print(f"{'TOTAL':<25} {self.total_ms():>10.3f}")
        print()


class _CollectedGpuTimer(GpuTimer):
    """GpuTimer that automatically records into a GpuTimerCollection."""

    def __init__(self, collection: GpuTimerCollection, name: str,
                 device: str | None = None):
        super().__init__(name, device)
        self._collection = collection

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if self._result:
            self._collection.record(self._result)
        return False


def demo_gpu_timing():
    """Demo: measure GPU vs CPU timing for a torch operation."""
    if torch is None:
        print("PyTorch not installed. Install with: pip install torch")
        return

    print("GPU Timer Demo")
    print(f"CUDA available: {_HAS_CUDA}")
    print()

    size = 4096

    # CPU timing
    print(f"Matrix multiply ({size}x{size}) on CPU:")
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # Warm up
    _ = torch.mm(a_cpu, b_cpu)

    with GpuTimer("matmul_cpu", device="cpu") as cpu_timer:
        _ = torch.mm(a_cpu, b_cpu)
    print(f"  {cpu_timer.result}")

    # GPU timing (if available)
    if _HAS_CUDA:
        print(f"\nMatrix multiply ({size}x{size}) on GPU:")
        a_gpu = torch.randn(size, size, device="cuda")
        b_gpu = torch.randn(size, size, device="cuda")

        # Warm up (first op triggers CUDA init, kernel compilation, etc.)
        for _ in range(3):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        with GpuTimer("matmul_gpu", device="cuda") as gpu_timer:
            _ = torch.mm(a_gpu, b_gpu)
        print(f"  {gpu_timer.result}")

        # Show why CPU timers are wrong for GPU
        print("\nDemonstrating why CPU timers fail for GPU ops:")
        cpu_start = time.perf_counter_ns()
        _ = torch.mm(a_gpu, b_gpu)  # async — returns immediately
        cpu_elapsed_ms = (time.perf_counter_ns() - cpu_start) / 1e6
        torch.cuda.synchronize()
        print(f"  CPU timer (wrong):  {cpu_elapsed_ms:.3f} ms — only measures launch overhead")
        print(f"  CUDA Event (right): {gpu_timer.elapsed_ms:.3f} ms — measures actual GPU work")
    else:
        print("\nNo CUDA GPU available. Showing CPU-only fallback.")
        print("On a CUDA system, GpuTimer uses torch.cuda.Event for accurate timing.")


if __name__ == "__main__":
    demo_gpu_timing()
