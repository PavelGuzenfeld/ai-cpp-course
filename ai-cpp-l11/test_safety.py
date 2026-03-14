"""Unit tests for Lesson 11: Memory Safety Without Sacrifice.

Tests std::span, std::optional, RAII, and smart pointer wrappers.
"""

import pytest
import safe_views
import memory_safety_demo


# ============================================================================
# std::span tests
# ============================================================================

class TestSpanSum:
    def test_basic_sum(self):
        assert safe_views.span_sum([1.0, 2.0, 3.0]) == 6.0

    def test_empty_sum(self):
        assert safe_views.span_sum([]) == 0.0

    def test_single_element(self):
        assert safe_views.span_sum([42.0]) == 42.0

    def test_large_sum(self):
        data = [1.0] * 10_000
        assert safe_views.span_sum(data) == 10_000.0

    def test_negative_values(self):
        assert safe_views.span_sum([-1.0, -2.0, 3.0]) == 0.0


class TestSpanSlice:
    def test_basic_slice(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = safe_views.span_slice(data, 1, 3)
        assert result == [20.0, 30.0, 40.0]

    def test_slice_from_start(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = safe_views.span_slice(data, 0, 2)
        assert result == [1.0, 2.0]

    def test_slice_to_end(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = safe_views.span_slice(data, 2, 2)
        assert result == [3.0, 4.0]

    def test_full_slice(self):
        data = [1.0, 2.0, 3.0]
        result = safe_views.span_slice(data, 0, 3)
        assert result == data

    def test_out_of_bounds_raises(self):
        data = [1.0, 2.0, 3.0]
        with pytest.raises(IndexError):
            safe_views.span_slice(data, 2, 5)

    def test_offset_past_end_raises(self):
        data = [1.0, 2.0]
        with pytest.raises(IndexError):
            safe_views.span_slice(data, 10, 1)


class TestSafeAt:
    def test_valid_index(self):
        data = [10.0, 20.0, 30.0]
        assert safe_views.safe_at(data, 0) == 10.0
        assert safe_views.safe_at(data, 1) == 20.0
        assert safe_views.safe_at(data, 2) == 30.0

    def test_out_of_bounds_raises(self):
        data = [10.0, 20.0, 30.0]
        with pytest.raises(IndexError):
            safe_views.safe_at(data, 3)

    def test_large_index_raises(self):
        data = [1.0]
        with pytest.raises(IndexError):
            safe_views.safe_at(data, 999)

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            safe_views.safe_at([], 0)


class TestRawPointerSum:
    def test_matches_span_sum(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert safe_views.raw_pointer_sum(data) == safe_views.span_sum(data)

    def test_empty(self):
        assert safe_views.raw_pointer_sum([]) == 0.0


class TestBenchmark:
    def test_benchmark_runs(self):
        data = [float(i) for i in range(1000)]
        span_us, raw_us = safe_views.benchmark_span_vs_raw(data, 10)
        assert span_us > 0
        assert raw_us > 0


# ============================================================================
# std::optional tests
# ============================================================================

class TestOptionalDetection:
    def test_present(self):
        det = memory_safety_demo.OptionalDetection(10.0, 20.0, 100.0, 80.0)
        assert det.has_value() is True
        assert det.area() == 8000.0

    def test_empty(self):
        det = memory_safety_demo.OptionalDetection()
        assert det.has_value() is False
        assert det.area() == 0.0

    def test_value_when_present(self):
        det = memory_safety_demo.OptionalDetection(5.0, 10.0, 50.0, 40.0)
        bbox = det.value()
        assert bbox.x == 5.0
        assert bbox.y == 10.0
        assert bbox.w == 50.0
        assert bbox.h == 40.0

    def test_value_when_empty_raises(self):
        det = memory_safety_demo.OptionalDetection()
        with pytest.raises(RuntimeError):
            det.value()

    def test_value_or_when_present(self):
        det = memory_safety_demo.OptionalDetection(1.0, 2.0, 3.0, 4.0)
        bbox = det.value_or(0.0, 0.0, 0.0, 0.0)
        assert bbox.x == 1.0  # returns the actual value, not the default

    def test_value_or_when_empty(self):
        det = memory_safety_demo.OptionalDetection()
        bbox = det.value_or(99.0, 99.0, 10.0, 10.0)
        assert bbox.x == 99.0
        assert bbox.area() == 100.0

    def test_detect_in_frame(self):
        # Frame 0 is divisible by 3 -> empty
        det0 = memory_safety_demo.detect_in_frame(0)
        assert det0.has_value() is False

        # Frame 1 is not divisible by 3 -> detection
        det1 = memory_safety_demo.detect_in_frame(1)
        assert det1.has_value() is True

    def test_repr(self):
        det = memory_safety_demo.OptionalDetection(1.0, 2.0, 3.0, 4.0)
        assert "OptionalDetection" in repr(det)

        empty = memory_safety_demo.OptionalDetection()
        assert "empty" in repr(empty)


# ============================================================================
# RAII tests
# ============================================================================

class TestRAIIBuffer:
    def test_create_and_size(self):
        buf = memory_safety_demo.RAIIBuffer(100)
        assert buf.size() == 100

    def test_set_and_get(self):
        buf = memory_safety_demo.RAIIBuffer(10)
        buf.set(0, 42.0)
        buf.set(9, 99.0)
        assert buf.get(0) == 42.0
        assert buf.get(9) == 99.0

    def test_initial_value_is_zero(self):
        buf = memory_safety_demo.RAIIBuffer(5)
        for i in range(5):
            assert buf.get(i) == 0.0

    def test_out_of_bounds_get_raises(self):
        buf = memory_safety_demo.RAIIBuffer(5)
        with pytest.raises(IndexError):
            buf.get(5)

    def test_out_of_bounds_set_raises(self):
        buf = memory_safety_demo.RAIIBuffer(5)
        with pytest.raises(IndexError):
            buf.set(5, 1.0)

    def test_active_count_tracks_lifecycle(self):
        initial = memory_safety_demo.RAIIBuffer.active_count()
        buf1 = memory_safety_demo.RAIIBuffer(10)
        assert memory_safety_demo.RAIIBuffer.active_count() == initial + 1
        buf2 = memory_safety_demo.RAIIBuffer(20)
        assert memory_safety_demo.RAIIBuffer.active_count() == initial + 2
        del buf1
        assert memory_safety_demo.RAIIBuffer.active_count() == initial + 1
        del buf2
        assert memory_safety_demo.RAIIBuffer.active_count() == initial

    def test_repr(self):
        buf = memory_safety_demo.RAIIBuffer(42)
        assert "42" in repr(buf)


# ============================================================================
# Smart pointer tests
# ============================================================================

class TestUniqueModel:
    def test_create(self):
        model = memory_safety_demo.UniqueModel("YOLOv8", 25_000_000)
        assert model.is_valid() is True
        assert model.name() == "YOLOv8"
        assert model.param_count() == 25_000_000

    def test_create_via_factory(self):
        model = memory_safety_demo.create_model("ResNet50", 25_000_000)
        assert model.is_valid() is True
        assert model.name() == "ResNet50"

    def test_info(self):
        model = memory_safety_demo.UniqueModel("test", 100)
        info = model.info()
        assert "test" in info
        assert "100" in info


class TestSharedBuffer:
    def test_create(self):
        buf = memory_safety_demo.SharedBuffer(1024, "test_buffer")
        assert buf.size() == 1024
        assert buf.label() == "test_buffer"
        assert buf.use_count() == 1

    def test_share_increments_count(self):
        buf = memory_safety_demo.SharedBuffer(512, "shared")
        assert buf.use_count() == 1

        copy = buf.share()
        assert buf.use_count() == 2
        assert copy.use_count() == 2
        assert copy.label() == "shared"

    def test_drop_shared_decrements_count(self):
        buf = memory_safety_demo.SharedBuffer(256, "drop_test")
        copy = buf.share()
        assert buf.use_count() == 2

        del copy
        assert buf.use_count() == 1

    def test_multiple_shares(self):
        buf = memory_safety_demo.SharedBuffer(128, "multi")
        copies = [buf.share() for _ in range(5)]
        assert buf.use_count() == 6  # original + 5 copies

        del copies
        assert buf.use_count() == 1

    def test_repr(self):
        buf = memory_safety_demo.SharedBuffer(64, "repr_test")
        r = repr(buf)
        assert "repr_test" in r
        assert "64" in r
