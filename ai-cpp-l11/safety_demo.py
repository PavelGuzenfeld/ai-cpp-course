"""Lesson 11: Memory Safety Without Sacrifice — Python Demo

Demonstrates std::span, std::optional, RAII, and smart pointers from Python.
"""

import safe_views
import memory_safety_demo


def demo_span():
    """std::span — safe views without copies."""
    print("=" * 60)
    print("std::span — Safe Views Without Copies")
    print("=" * 60)

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Sum using std::span (safe)
    total = safe_views.span_sum(data)
    print(f"span_sum({data}) = {total}")

    # Slice using std::span (zero-cost view internally)
    sliced = safe_views.span_slice(data, 2, 5)
    print(f"span_slice(data, offset=2, count=5) = {sliced}")

    # Bounds-checked access
    val = safe_views.safe_at(data, 3)
    print(f"safe_at(data, 3) = {val}")

    # Out-of-bounds access raises IndexError
    try:
        safe_views.safe_at(data, 100)
    except IndexError as e:
        print(f"safe_at(data, 100) raised IndexError: {e}")

    # Out-of-bounds slice raises IndexError
    try:
        safe_views.span_slice(data, 8, 5)
    except IndexError as e:
        print(f"span_slice(data, 8, 5) raised IndexError: {e}")

    # Raw pointer sum (for comparison)
    raw_total = safe_views.raw_pointer_sum(data)
    print(f"raw_pointer_sum({data}) = {raw_total}")
    print()


def demo_benchmark():
    """Benchmark: std::span vs raw pointer (should be identical in release)."""
    print("=" * 60)
    print("Benchmark: span vs raw pointer")
    print("=" * 60)

    data = list(range(100_000))
    data = [float(x) for x in data]

    span_us, raw_us = safe_views.benchmark_span_vs_raw(data, 100)
    print(f"  span_sum:        {span_us:10.1f} us ({100} iterations)")
    print(f"  raw_pointer_sum: {raw_us:10.1f} us ({100} iterations)")

    if raw_us > 0:
        ratio = span_us / raw_us
        print(f"  Ratio (span/raw): {ratio:.3f}x")
        if 0.8 < ratio < 1.2:
            print("  -> Essentially identical (zero-cost abstraction)")
        elif ratio > 1.2:
            print("  -> span is slower (likely debug build with bounds checks)")
        else:
            print("  -> span is faster (measurement noise)")
    print()


def demo_optional():
    """std::optional — no more null pointers."""
    print("=" * 60)
    print("std::optional — No More Null Pointers")
    print("=" * 60)

    # Detection present
    det = memory_safety_demo.OptionalDetection(10.0, 20.0, 100.0, 80.0)
    print(f"Detection present: {det}")
    print(f"  has_value() = {det.has_value()}")
    print(f"  area() = {det.area()}")
    bbox = det.value()
    print(f"  value() = {bbox}")

    # No detection
    empty = memory_safety_demo.OptionalDetection()
    print(f"\nEmpty detection: {empty}")
    print(f"  has_value() = {empty.has_value()}")
    print(f"  area() = {empty.area()}")

    # value() on empty raises
    try:
        empty.value()
    except RuntimeError as e:
        print(f"  value() raised RuntimeError: {e}")

    # value_or provides a default
    fallback = empty.value_or(0.0, 0.0, 0.0, 0.0)
    print(f"  value_or(0,0,0,0) = {fallback}")

    # Simulate detector over multiple frames
    print("\n  Simulating detector over 10 frames:")
    for frame_id in range(10):
        det = memory_safety_demo.detect_in_frame(frame_id)
        status = "DETECTED" if det.has_value() else "empty"
        print(f"    Frame {frame_id}: {status} (area={det.area():.0f})")
    print()


def demo_raii():
    """RAII — resources clean themselves."""
    print("=" * 60)
    print("RAII — Resources Clean Themselves")
    print("=" * 60)

    print(f"Active buffers before: {memory_safety_demo.RAIIBuffer.active_count()}")

    buf = memory_safety_demo.RAIIBuffer(1000)
    print(f"Created buffer: {buf}")
    print(f"Active buffers after create: {memory_safety_demo.RAIIBuffer.active_count()}")

    # Set and get values
    buf.set(0, 42.0)
    buf.set(999, 99.0)
    print(f"  buf.get(0) = {buf.get(0)}")
    print(f"  buf.get(999) = {buf.get(999)}")
    print(f"  buf.size() = {buf.size()}")

    # Out-of-bounds access raises
    try:
        buf.get(1000)
    except IndexError as e:
        print(f"  buf.get(1000) raised IndexError: {e}")

    # When buf goes out of scope (in C++), destructor frees memory automatically.
    # In Python, the ref count drop triggers the destructor.
    del buf
    print(f"Active buffers after del: {memory_safety_demo.RAIIBuffer.active_count()}")
    print()


def demo_smart_pointers():
    """Smart pointers — ownership is clear."""
    print("=" * 60)
    print("Smart Pointers — Ownership Is Clear")
    print("=" * 60)

    # unique_ptr: single ownership
    print("--- std::unique_ptr (single ownership) ---")
    model = memory_safety_demo.create_model("YOLOv8", 25_000_000)
    print(f"  Created: {model.info()}")
    print(f"  Name: {model.name()}")
    print(f"  Params: {model.param_count()}")
    print(f"  Valid: {model.is_valid()}")

    # shared_ptr: reference-counted ownership
    print("\n--- std::shared_ptr (reference-counted ownership) ---")
    buf = memory_safety_demo.SharedBuffer(4096, "frame_buffer")
    print(f"  Created: {buf}")
    print(f"  use_count = {buf.use_count()}")

    # Share the buffer (increments reference count)
    shared_copy = buf.share()
    print(f"  After share(): original use_count = {buf.use_count()}")
    print(f"  After share(): copy use_count = {shared_copy.use_count()}")
    print(f"  Same label? {buf.label()} == {shared_copy.label()}")

    # Drop one reference
    del shared_copy
    print(f"  After del copy: use_count = {buf.use_count()}")
    print()


def main():
    print("Lesson 11: Memory Safety Without Sacrifice")
    print("=" * 60)
    print()

    demo_span()
    demo_benchmark()
    demo_optional()
    demo_raii()
    demo_smart_pointers()

    print("All demos completed.")


if __name__ == "__main__":
    main()
