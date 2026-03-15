"""
Demonstrates Jetson power/performance tradeoffs.

Shows current power mode and clock frequencies, runs a compute benchmark
at the current settings, and prints instructions for switching power modes.
Reads tegrastats if available to show GPU/CPU utilization, power draw, and
temperature.

Falls back gracefully on non-Jetson hardware.

Usage:
    python power_mode_demo.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Jetson detection (lightweight version)
# ---------------------------------------------------------------------------

def is_jetson():
    """Return True if running on a Jetson device."""
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return False
    try:
        model_str = model_path.read_text().strip().rstrip("\x00").lower()
        return "jetson" in model_str or "nvidia" in model_str
    except OSError:
        return False


def get_jetson_model():
    """Return the Jetson model string, or None."""
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return None
    try:
        return model_path.read_text().strip().rstrip("\x00")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Power mode and clock info
# ---------------------------------------------------------------------------

def get_power_mode():
    """Read current nvpmodel power mode."""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_clock_frequencies():
    """
    Read current GPU and CPU clock frequencies.
    Returns a dict with available frequency info.
    """
    freqs = {}

    # GPU frequency -- try several known sysfs paths
    gpu_freq_paths = [
        "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq",
    ]
    for path in gpu_freq_paths:
        try:
            with open(path) as f:
                freq_hz = int(f.read().strip())
                freqs["gpu_freq_mhz"] = freq_hz / 1e6
                break
        except (OSError, ValueError):
            continue

    # GPU max frequency
    gpu_max_paths = [p.replace("cur_freq", "max_freq") for p in gpu_freq_paths]
    for path in gpu_max_paths:
        try:
            with open(path) as f:
                freq_hz = int(f.read().strip())
                freqs["gpu_max_freq_mhz"] = freq_hz / 1e6
                break
        except (OSError, ValueError):
            continue

    # CPU frequency (core 0)
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            freq_khz = int(f.read().strip())
            freqs["cpu_freq_mhz"] = freq_khz / 1e3
    except (OSError, ValueError):
        pass

    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
            freq_khz = int(f.read().strip())
            freqs["cpu_max_freq_mhz"] = freq_khz / 1e3
    except (OSError, ValueError):
        pass

    # Count online CPU cores
    try:
        with open("/sys/devices/system/cpu/online") as f:
            freqs["cpu_online"] = f.read().strip()
    except OSError:
        pass

    return freqs


def show_current_settings():
    """Display current power mode and clock frequencies."""
    print("=" * 60)
    print("  Current Power Settings")
    print("=" * 60)

    model = get_jetson_model()
    if model:
        print(f"  Device:           {model}")
    else:
        print("  Device:           Not a Jetson (or model not readable)")

    power_mode = get_power_mode()
    if power_mode:
        for line in power_mode.splitlines():
            stripped = line.strip()
            if stripped:
                print(f"  Power mode:       {stripped}")
    else:
        print("  Power mode:       unavailable (nvpmodel not found)")

    freqs = get_clock_frequencies()
    if "gpu_freq_mhz" in freqs:
        gpu_str = f"{freqs['gpu_freq_mhz']:.0f} MHz"
        if "gpu_max_freq_mhz" in freqs:
            gpu_str += f" (max: {freqs['gpu_max_freq_mhz']:.0f} MHz)"
        print(f"  GPU frequency:    {gpu_str}")

    if "cpu_freq_mhz" in freqs:
        cpu_str = f"{freqs['cpu_freq_mhz']:.0f} MHz"
        if "cpu_max_freq_mhz" in freqs:
            cpu_str += f" (max: {freqs['cpu_max_freq_mhz']:.0f} MHz)"
        print(f"  CPU frequency:    {cpu_str}")

    if "cpu_online" in freqs:
        print(f"  CPU cores online: {freqs['cpu_online']}")

    if not freqs and not power_mode:
        print("  (No Jetson power/frequency data available on this system)")

    print()


# ---------------------------------------------------------------------------
# Compute benchmark
# ---------------------------------------------------------------------------

def run_compute_benchmark():
    """
    Run a compute benchmark at current power settings.
    Uses both CPU (NumPy) and GPU (PyTorch CUDA) if available.
    """
    print("=" * 60)
    print("  Compute Benchmark at Current Settings")
    print("=" * 60)

    n = 4 * 1024 * 1024  # 4M elements
    n_iters = 100

    # CPU benchmark (NumPy matrix operations)
    a_np = np.random.randn(1024, 1024).astype(np.float32)
    b_np = np.random.randn(1024, 1024).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = a_np @ b_np

    times_cpu = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = a_np @ b_np
        t1 = time.perf_counter()
        times_cpu.append((t1 - t0) * 1000)

    cpu_mean = np.mean(times_cpu)
    cpu_std = np.std(times_cpu)
    cpu_gflops = (2 * 1024 ** 3) / (cpu_mean / 1000) / 1e9

    print(f"  CPU (1024x1024 matmul): {cpu_mean:.2f} +/- {cpu_std:.2f} ms")
    print(f"  CPU throughput:         {cpu_gflops:.1f} GFLOPS")

    # GPU benchmark
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            a_gpu = torch.randn(1024, 1024, device=device)
            b_gpu = torch.randn(1024, 1024, device=device)

            # Warmup
            for _ in range(10):
                _ = a_gpu @ b_gpu
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_iters):
                _ = a_gpu @ b_gpu
            end.record()
            torch.cuda.synchronize()

            gpu_mean = start.elapsed_time(end) / n_iters
            gpu_gflops = (2 * 1024 ** 3) / (gpu_mean / 1000) / 1e9

            print(f"  GPU (1024x1024 matmul): {gpu_mean:.3f} ms")
            print(f"  GPU throughput:         {gpu_gflops:.1f} GFLOPS")
            print(f"  GPU/CPU speedup:        {cpu_mean / gpu_mean:.1f}x")

            # Vector add for bandwidth measurement
            va = torch.randn(n, device=device)
            vb = torch.randn(n, device=device)
            for _ in range(10):
                _ = va + vb
            torch.cuda.synchronize()

            start.record()
            for _ in range(n_iters):
                _ = va + vb
            end.record()
            torch.cuda.synchronize()

            va_ms = start.elapsed_time(end) / n_iters
            bandwidth_gb = (n * 4 * 3) / (va_ms / 1000) / 1e9
            print(f"  Memory bandwidth:       {bandwidth_gb:.1f} GB/s (vector add, 16 MB)")
        else:
            print("  GPU: CUDA not available")
    except ImportError:
        print("  GPU: PyTorch not installed")

    print()


# ---------------------------------------------------------------------------
# Tegrastats reading
# ---------------------------------------------------------------------------

def read_tegrastats(duration_seconds=3):
    """
    Run tegrastats for a short duration and parse the output.
    Shows GPU/CPU utilization, power draw, and temperature.
    """
    print("=" * 60)
    print("  Tegrastats Snapshot")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "500"],
            capture_output=True, text=True,
            timeout=duration_seconds + 2,
        )
        lines = result.stdout.strip().splitlines()
    except subprocess.TimeoutExpired as e:
        # tegrastats runs continuously; timeout is expected
        output = e.stdout
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        lines = output.strip().splitlines() if output else []
    except FileNotFoundError:
        print("  tegrastats not found. This is expected on non-Jetson systems.")
        print()
        print("  On Jetson, tegrastats shows real-time metrics including:")
        print("    - CPU and GPU utilization percentages")
        print("    - Memory usage (RAM and swap)")
        print("    - Power draw per rail (VDD_CPU, VDD_GPU, VDD_SOC)")
        print("    - Temperature for CPU, GPU, and other thermal zones")
        print()
        return
    except subprocess.SubprocessError as e:
        print(f"  tegrastats error: {e}")
        print()
        return

    if not lines:
        print("  No tegrastats output captured.")
        print()
        return

    # Parse and display the last few lines
    print(f"  Captured {len(lines)} sample(s):")
    print()
    for line in lines[-3:]:
        _parse_tegrastats_line(line)
    print()


def _parse_tegrastats_line(line):
    """Parse a single tegrastats output line and display key metrics."""
    # tegrastats output format varies by JetPack version.
    # Common fields: RAM, CPU, GPU, EMC, temperatures, power
    parts = line.split()

    metrics = {}
    i = 0
    while i < len(parts):
        token = parts[i]

        # RAM usage
        if token == "RAM":
            if i + 1 < len(parts):
                metrics["RAM"] = parts[i + 1]
                i += 2
                continue

        # GPU utilization
        if token.startswith("GR3D_FREQ"):
            if i + 1 < len(parts):
                metrics["GPU util"] = parts[i + 1]
                i += 2
                continue

        # CPU utilization
        if token == "CPU":
            if i + 1 < len(parts):
                metrics["CPU"] = parts[i + 1]
                i += 2
                continue

        # Temperature fields contain @
        temp_keys = ("cpu", "gpu", "soc", "tj")
        if "@" in token and any(k in token.lower() for k in temp_keys):
            metrics[token.split("@")[0]] = token.split("@")[1] if "@" in token else token

        # Power fields (VDD_*)
        if token.startswith("VDD_"):
            if i + 1 < len(parts):
                metrics[token] = parts[i + 1]
                i += 2
                continue

        i += 1

    if metrics:
        for key, value in metrics.items():
            print(f"    {key:<20} {value}")
    else:
        # Could not parse; print raw line
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Power mode switching instructions
# ---------------------------------------------------------------------------

def print_power_mode_instructions():
    """Print instructions for switching power modes on Jetson."""
    print("=" * 60)
    print("  Power Mode Management")
    print("=" * 60)
    print()
    print("  List available power modes:")
    print("    sudo nvpmodel -p --verbose")
    print()
    print("  Query current mode:")
    print("    sudo nvpmodel -q")
    print()
    print("  Switch power modes:")
    print("    sudo nvpmodel -m 0    # MAXN -- maximum performance")
    print("    sudo nvpmodel -m 1    # mode 1 (varies by device)")
    print("    sudo nvpmodel -m 2    # mode 2 (varies by device)")
    print()
    print("  Lock clocks at maximum for consistent benchmarks:")
    print("    sudo jetson_clocks")
    print()
    print("  Restore default clock scaling:")
    print("    sudo jetson_clocks --restore")
    print()
    print("  Monitor system in real time:")
    print("    tegrastats --interval 1000")
    print()

    if is_jetson():
        print("  Typical power modes for this device:")
        power_info = get_power_mode()
        if power_info:
            for line in power_info.splitlines():
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped}")
            print()

        # Try to list all modes
        try:
            result = subprocess.run(
                ["nvpmodel", "-p", "--verbose"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                print("  Available modes on this device:")
                for line in result.stdout.splitlines():
                    stripped = line.strip()
                    if stripped:
                        print(f"    {stripped}")
                print()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    else:
        print("  Example power modes (Jetson Orin AGX):")
        print()
        print(f"    {'Mode':<8} {'Name':<15} {'Power':<10} {'GPU Cores':<12} {'CPU Cores':<10}")
        print(f"    {'-' * 8} {'-' * 15} {'-' * 10} {'-' * 12} {'-' * 10}")
        print(f"    {'0':<8} {'MAXN':<15} {'60 W':<10} {'2048':<12} {'12':<10}")
        print(f"    {'1':<8} {'50W':<15} {'50 W':<10} {'2048':<12} {'8':<10}")
        print(f"    {'2':<8} {'30W':<15} {'30 W':<10} {'1024':<12} {'8':<10}")
        print(f"    {'3':<8} {'15W':<15} {'15 W':<10} {'512':<12} {'4':<10}")
        print()
        print("  Lower power modes reduce thermal output at the cost of throughput.")
        print("  Choose the lowest mode that meets your real-time latency requirements.")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Lesson 7J -- Jetson Power/Performance Tradeoffs")
    print()

    show_current_settings()
    run_compute_benchmark()

    if is_jetson():
        read_tegrastats(duration_seconds=3)

    print_power_mode_instructions()

    print("=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
