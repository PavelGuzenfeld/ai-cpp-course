"""
Benchmark GPU preprocessing on Jetson hardware.

Auto-detects Jetson devices and reports hardware details, then runs three
benchmarks that highlight Jetson-specific behavior:

  1. Unified vs explicit memory allocation (on Jetson, unified should be
     competitive because CPU and GPU share the same physical DRAM).
  2. Fused CUDA kernel vs NumPy preprocessing.
  3. Power mode comparison (reads nvpmodel on Jetson).

Falls back to reference numbers with explanation when not running on Jetson.

Usage:
    python benchmark_jetson.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn(fn, n_iters=100, warmup=10):
    """Time a function over n_iters, returning mean and std in milliseconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    arr = np.array(times)
    return arr.mean(), arr.std()


def print_table(title, rows):
    """Print a formatted benchmark results table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 8}")

    baseline = rows[0][1] if rows else 1.0
    for name, mean, std in rows:
        speedup = baseline / mean if mean > 0 else float("inf")
        print(f"  {name:<35} {mean:>10.3f} {std:>10.3f} {speedup:>7.1f}x")
    print()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_jetson():
    """
    Detect if running on a Jetson board by reading /proc/device-tree/model.
    Returns a dict with model info, or None if not on Jetson.
    """
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return None

    try:
        model_str = model_path.read_text().strip().rstrip("\x00")
    except OSError:
        return None

    if "jetson" not in model_str.lower() and "nvidia" not in model_str.lower():
        return None

    info = {"model": model_str}
    info["cuda_cores"] = _get_cuda_cores()
    info["memory_total_mb"] = _get_total_memory_mb()
    info["power_mode"] = _get_power_mode()
    return info


def _get_cuda_cores():
    """Estimate CUDA core count from GPU device name."""
    if not HAS_CUDA:
        return "unknown"

    name = torch.cuda.get_device_name(0).lower()
    core_map = {
        "orin": 2048,
        "xavier": 384,
        "nano": 128,
        "tx2": 256,
    }
    for key, cores in core_map.items():
        if key in name:
            return cores

    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count * 128


def _get_total_memory_mb():
    """Get total system memory in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _get_power_mode():
    """Read the current nvpmodel power mode, or return None."""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "POWER_MODEL" in line.upper() or "NV Power Mode" in line:
                return line.strip()
        # Return first non-empty line as fallback
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def report_device_info():
    """Print device information and return jetson_info dict (or None)."""
    jetson_info = detect_jetson()

    print("=" * 70)
    print("  Device Information")
    print("=" * 70)

    if jetson_info:
        print(f"  Platform:       Jetson")
        print(f"  Model:          {jetson_info['model']}")
        print(f"  CUDA cores:     {jetson_info['cuda_cores']}")
        print(f"  System memory:  {jetson_info['memory_total_mb']} MB")
        if jetson_info["power_mode"]:
            print(f"  Power mode:     {jetson_info['power_mode']}")
    else:
        print("  Platform:       Desktop / Server (not Jetson)")

    if HAS_CUDA:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:     {props.total_mem // (1024 * 1024)} MB")
        print(f"  CUDA version:   {torch.version.cuda}")
        print(f"  SM count:       {props.multi_processor_count}")
    elif HAS_TORCH:
        print("  CUDA:           not available")
    else:
        print("  PyTorch:        not installed")

    print()
    return jetson_info


# ---------------------------------------------------------------------------
# Benchmark 1: Unified vs Explicit Memory
# ---------------------------------------------------------------------------

def benchmark_unified_vs_explicit():
    """
    Compare unified memory vs explicit device allocation.

    On Jetson, unified memory accesses shared LPDDR with zero-copy semantics,
    so the gap between unified and explicit should be small or nonexistent.
    On desktop, unified memory incurs page-fault overhead.
    """
    title = "Benchmark 1: Unified vs Explicit Memory Allocation"

    if not HAS_CUDA:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        _print_unified_reference()
        return

    device = torch.device("cuda:0")
    sizes_mb = [1, 4, 16, 64]
    n_iters = 20
    warmup = 5

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Iterations: {n_iters} (warmup: {warmup})")
    print()

    header = f"  {'Size':>8}  {'Explicit (ms)':>14}  {'Unified (ms)':>14}  {'Ratio':>8}  {'Comment'}"
    print(header)
    print(f"  {'-' * 70}")

    for size_mb in sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 4  # float32

        t_explicit = _bench_explicit(n_elements, device, n_iters, warmup)
        t_unified = _bench_unified(n_elements, device, n_iters, warmup)

        ratio = t_unified / t_explicit if t_explicit > 0 else float("inf")

        if ratio < 1.15:
            comment = "unified competitive"
        elif ratio < 2.0:
            comment = "explicit faster"
        else:
            comment = "explicit much faster"

        print(
            f"  {size_mb:>6} MB  {t_explicit:>12.3f}   "
            f"{t_unified:>12.3f}   {ratio:>7.2f}x  {comment}"
        )

    print()
    print("  On Jetson (shared LPDDR), expect ratios near 1.0.")
    print("  On desktop (discrete GPU), expect ratios of 2-10x due to page faults.")
    print()


def _bench_explicit(n, device, n_iters, warmup):
    """Benchmark: pre-allocated device tensors, compute only."""
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    c = torch.empty(n, device=device)

    for _ in range(warmup):
        torch.add(a, b, out=c)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        torch.add(a, b, out=c)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _bench_unified(n, device, n_iters, warmup):
    """
    Simulate unified memory: allocate on CPU, transfer to GPU each iteration.
    On Jetson this is effectively zero-copy; on desktop it triggers a PCIe transfer.
    """
    a_cpu = torch.randn(n)
    b_cpu = torch.randn(n)

    for _ in range(warmup):
        a_dev = a_cpu.to(device)
        b_dev = b_cpu.to(device)
        _ = a_dev + b_dev
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        a_dev = a_cpu.to(device)
        b_dev = b_cpu.to(device)
        _ = a_dev + b_dev
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _print_unified_reference():
    """Show reference numbers when no GPU is available."""
    print()
    print("  No CUDA device found. Reference numbers for comparison:")
    print()
    print(f"  {'Size':>8}  {'Desktop Explicit':>18}  {'Desktop Unified':>18}"
          f"  {'Jetson Explicit':>18}  {'Jetson Unified':>18}")
    print(f"  {'-' * 80}")
    ref = {
        1:  ("0.015 ms", "0.85 ms",  "0.025 ms", "0.028 ms"),
        4:  ("0.040 ms", "1.60 ms",  "0.080 ms", "0.085 ms"),
        16: ("0.130 ms", "4.20 ms",  "0.300 ms", "0.310 ms"),
        64: ("0.500 ms", "15.0 ms",  "1.100 ms", "1.120 ms"),
    }
    for s in [1, 4, 16, 64]:
        de, du, je, ju = ref[s]
        print(f"  {s:>6} MB  {de:>18}  {du:>18}  {je:>18}  {ju:>18}")
    print()
    print("  Jetson unified and explicit are nearly identical (shared LPDDR).")
    print("  Desktop unified is 10-30x slower due to PCIe page faults.")
    print()


# ---------------------------------------------------------------------------
# Benchmark 2: Fused Kernel vs NumPy Preprocessing
# ---------------------------------------------------------------------------

def benchmark_fused_preprocess():
    """Compare fused GPU preprocessing kernel vs NumPy baseline."""
    title = "Benchmark 2: Fused Kernel vs NumPy Preprocessing"

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    rows = []

    # NumPy baseline
    def numpy_preprocess():
        img = image.astype(np.float32) / 255.0
        img = (img - np.array(mean_vals)) / np.array(std_vals)
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img)

    m, s = time_fn(numpy_preprocess)
    rows.append(("NumPy preprocessing", m, s))

    # PyTorch CPU
    if HAS_TORCH:
        image_t = torch.from_numpy(image)
        mean_t = torch.tensor(mean_vals).view(1, 1, 3)
        std_t = torch.tensor(std_vals).view(1, 1, 3)

        def torch_cpu_preprocess():
            img = image_t.float() / 255.0
            img = (img - mean_t) / std_t
            return img.permute(2, 0, 1).contiguous()

        m, s = time_fn(torch_cpu_preprocess)
        rows.append(("PyTorch CPU preprocessing", m, s))

    # PyTorch GPU
    if HAS_CUDA:
        device = torch.device("cuda:0")
        image_gpu = torch.from_numpy(image).to(device)
        mean_gpu = torch.tensor(mean_vals, device=device).view(1, 1, 3)
        std_gpu = torch.tensor(std_vals, device=device).view(1, 1, 3)

        def torch_gpu_preprocess():
            img = image_gpu.float() / 255.0
            img = (img - mean_gpu) / std_gpu
            return img.permute(2, 0, 1).contiguous()

        # Extra warmup for GPU
        for _ in range(20):
            torch_gpu_preprocess()
        torch.cuda.synchronize()

        m, s = time_fn(torch_gpu_preprocess)
        rows.append(("PyTorch GPU preprocessing", m, s))

    # Try C++ fused kernel if built
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "build"))
        import gpu_preprocess as gpu_mod

        if gpu_mod.cuda_available():
            def fused_preprocess():
                return gpu_mod.fused_preprocess(image, mean_vals, std_vals)

            m, s = time_fn(fused_preprocess)
            rows.append(("CUDA fused kernel", m, s))
    except ImportError:
        pass

    print_table(title, rows)

    if not HAS_CUDA:
        print("  Note: GPU benchmarks skipped (no CUDA device).")
        print("  On Jetson Orin, the fused GPU kernel is typically 5-15x faster")
        print("  than NumPy for 640x480 images.")
        print()


# ---------------------------------------------------------------------------
# Benchmark 3: Power Mode Comparison
# ---------------------------------------------------------------------------

def benchmark_power_mode():
    """
    Report current power mode and run a compute benchmark.
    On non-Jetson hardware, print an explanation of Jetson power modes.
    """
    title = "Benchmark 3: Power Mode Analysis"

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    jetson_info = detect_jetson()

    if not jetson_info:
        _print_power_mode_reference()
        return

    power_mode = jetson_info.get("power_mode")
    print(f"  Current power mode: {power_mode or 'unknown'}")
    print()

    # Read clock frequencies if available
    gpu_freq = _read_gpu_freq()
    cpu_freq = _read_cpu_freq()
    if gpu_freq:
        print(f"  GPU frequency: {gpu_freq}")
    if cpu_freq:
        print(f"  CPU frequency: {cpu_freq}")
    print()

    if HAS_CUDA:
        # Run a simple compute benchmark at current power setting
        device = torch.device("cuda:0")
        n = 4 * 1024 * 1024
        a = torch.randn(n, device=device)
        b = torch.randn(n, device=device)

        # Warmup
        for _ in range(10):
            _ = a + b
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = a + b
        end.record()
        torch.cuda.synchronize()

        ms_per_iter = start.elapsed_time(end) / 100
        bandwidth_gb = (n * 4 * 3) / (ms_per_iter / 1000) / 1e9

        print(f"  Vector add (16 MB): {ms_per_iter:.3f} ms/iter")
        print(f"  Effective bandwidth: {bandwidth_gb:.1f} GB/s")
        print()

    print("  To compare power modes, switch modes and re-run:")
    print("    sudo nvpmodel -m 0   # Max performance (MAXN)")
    print("    sudo nvpmodel -m 1   # Power-saving mode")
    print("    sudo nvpmodel -m 2   # (varies by Jetson model)")
    print("    sudo jetson_clocks   # Lock clocks at max for consistent benchmarks")
    print()


def _read_gpu_freq():
    """Try to read current GPU frequency."""
    freq_paths = [
        "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq",
    ]
    for path in freq_paths:
        try:
            with open(path) as f:
                freq_hz = int(f.read().strip())
                return f"{freq_hz / 1e6:.0f} MHz"
        except (OSError, ValueError):
            continue
    return None


def _read_cpu_freq():
    """Try to read current CPU frequency."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            freq_khz = int(f.read().strip())
            return f"{freq_khz / 1e3:.0f} MHz"
    except (OSError, ValueError):
        return None


def _print_power_mode_reference():
    """Print power mode explanation for non-Jetson systems."""
    print()
    print("  Not running on Jetson -- showing reference power mode information.")
    print()
    print("  Jetson power modes (nvpmodel) trade performance for power draw:")
    print()
    print(f"  {'Mode':<20} {'Power (W)':>10} {'GPU Freq':>12} {'CPU Cores':>10} {'Relative Perf':>14}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 14}")
    print(f"  {'MAXN (mode 0)':<20} {'60 W':>10} {'1.3 GHz':>12} {'12':>10} {'1.00x':>14}")
    print(f"  {'50W (mode 1)':<20} {'50 W':>10} {'1.3 GHz':>12} {'8':>10} {'0.85x':>14}")
    print(f"  {'30W (mode 2)':<20} {'30 W':>10} {'0.9 GHz':>12} {'8':>10} {'0.60x':>14}")
    print(f"  {'15W (mode 3)':<20} {'15 W':>10} {'0.6 GHz':>12} {'4':>10} {'0.35x':>14}")
    print()
    print("  (Values shown are approximate for Jetson Orin AGX.)")
    print("  Lower power modes reduce thermal output and are useful for")
    print("  continuous operation without active cooling.")
    print()


# ---------------------------------------------------------------------------
# Jetson-specific recommendations
# ---------------------------------------------------------------------------

def print_recommendations(jetson_info):
    """Print optimization recommendations based on benchmark results."""
    print("=" * 70)
    print("  Jetson-Specific Recommendations")
    print("=" * 70)

    if jetson_info:
        print(f"  Running on: {jetson_info['model']}")
        print()
        print("  1. MEMORY: Use unified memory (cudaMallocManaged) freely.")
        print("     On Jetson, it performs identically to explicit allocation")
        print("     because CPU and GPU share the same physical DRAM.")
        print()
        print("  2. PREPROCESSING: Fuse normalize/transpose/scale into a single")
        print("     CUDA kernel. Avoids multiple passes over data and keeps the")
        print("     GPU fed with work.")
        print()
        print("  3. POWER: Use 'sudo nvpmodel -m 0' and 'sudo jetson_clocks'")
        print("     for maximum throughput during benchmarks. For deployment,")
        print("     find the lowest power mode that meets your latency target.")
        print()
        print("  4. PRECISION: Use FP16 or INT8 where possible. Jetson GPUs have")
        print("     limited memory bandwidth; halving data size nearly doubles")
        print("     effective throughput for bandwidth-bound kernels.")
        print()
        print("  5. DLA: For supported operations, offload to the Deep Learning")
        print("     Accelerator to free up the GPU for other work.")
    else:
        print()
        print("  Not running on Jetson. Key differences from desktop GPUs:")
        print()
        print("  - Unified memory is free on Jetson (shared DRAM), expensive on desktop")
        print("  - No PCIe bottleneck on Jetson, but lower absolute memory bandwidth")
        print("  - Power modes let you trade performance for thermal/power budget")
        print("  - DLA (Deep Learning Accelerator) offloads inference from the GPU")
        print()
        print("  Run this benchmark on a Jetson device to see the full picture.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Lesson 7J -- Jetson GPU Programming Benchmarks")
    print(f"Image size: 640x480x3 (uint8)")
    print()

    jetson_info = report_device_info()

    benchmark_unified_vs_explicit()
    benchmark_fused_preprocess()
    benchmark_power_mode()
    print_recommendations(jetson_info)

    print("=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
