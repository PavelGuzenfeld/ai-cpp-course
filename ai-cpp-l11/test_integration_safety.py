"""Integration tests for Lesson 11: Memory Safety Without Sacrifice.

Simulates a tracking loop using safe types. Processes 1000 frames with
optional detections to verify no memory issues within Python.
"""

import pytest
import safe_views
import memory_safety_demo


class TestTrackingLoopIntegration:
    """Simulate a full tracking loop using safe memory patterns."""

    def test_1000_frames_with_optional_detections(self):
        """Process 1000 frames. Some have detections, some do not.
        Verify correct handling of present and absent detections."""
        detected_count = 0
        empty_count = 0
        total_area = 0.0

        for frame_id in range(1000):
            det = memory_safety_demo.detect_in_frame(frame_id)

            if det.has_value():
                detected_count += 1
                bbox = det.value()
                total_area += bbox.area()
                assert bbox.w > 0
                assert bbox.h > 0
            else:
                empty_count += 1
                # value_or provides a safe default
                fallback = det.value_or(0.0, 0.0, 0.0, 0.0)
                assert fallback.area() == 0.0

        # Every 3rd frame is empty (frame 0, 3, 6, ...)
        # Frames 0..999: 334 frames divisible by 3 (0, 3, 6, ..., 999)
        assert empty_count == 334
        assert detected_count == 666
        assert total_area > 0

    def test_buffer_lifecycle_in_loop(self):
        """Create and destroy buffers in a loop. Verify no leaks via
        the active_count tracker."""
        initial_count = memory_safety_demo.RAIIBuffer.active_count()

        for i in range(100):
            buf = memory_safety_demo.RAIIBuffer(1000)
            buf.set(0, float(i))
            assert buf.get(0) == float(i)
            del buf

        assert memory_safety_demo.RAIIBuffer.active_count() == initial_count

    def test_many_buffers_active_simultaneously(self):
        """Create many buffers at once, then release them all."""
        initial_count = memory_safety_demo.RAIIBuffer.active_count()
        buffers = []

        for i in range(50):
            buf = memory_safety_demo.RAIIBuffer(500)
            buf.set(0, float(i))
            buffers.append(buf)

        assert memory_safety_demo.RAIIBuffer.active_count() == initial_count + 50

        # Release all
        buffers.clear()
        assert memory_safety_demo.RAIIBuffer.active_count() == initial_count

    def test_shared_buffer_in_pipeline(self):
        """Simulate a pipeline where multiple stages share a buffer."""
        # Create a shared buffer (simulating frame data)
        frame_buf = memory_safety_demo.SharedBuffer(1920 * 1080, "frame_0")
        assert frame_buf.use_count() == 1

        # Simulate pipeline stages sharing the buffer
        detector_ref = frame_buf.share()
        tracker_ref = frame_buf.share()
        visualizer_ref = frame_buf.share()
        assert frame_buf.use_count() == 4

        # Detector finishes first
        del detector_ref
        assert frame_buf.use_count() == 3

        # Tracker finishes
        del tracker_ref
        assert frame_buf.use_count() == 2

        # Visualizer finishes
        del visualizer_ref
        assert frame_buf.use_count() == 1

        # All stages done, only original reference remains
        assert frame_buf.label() == "frame_0"

    def test_span_operations_on_large_data(self):
        """Process large buffers through span operations repeatedly."""
        data = [float(i) for i in range(10_000)]

        for _ in range(100):
            total = safe_views.span_sum(data)
            assert total == sum(data)

            # Slice the middle 1000 elements
            sliced = safe_views.span_slice(data, 4500, 1000)
            assert len(sliced) == 1000
            assert sliced[0] == 4500.0
            assert sliced[-1] == 5499.0

    def test_detection_to_tracking_pipeline(self):
        """Full pipeline: detect -> filter -> track using safe types."""
        tracked_boxes = []
        min_area = 2000.0

        for frame_id in range(200):
            det = memory_safety_demo.detect_in_frame(frame_id)

            if det.has_value():
                bbox = det.value()
                # Filter by minimum area
                if bbox.area() >= min_area:
                    tracked_boxes.append({
                        "frame": frame_id,
                        "x": bbox.x,
                        "y": bbox.y,
                        "w": bbox.w,
                        "h": bbox.h,
                        "area": bbox.area(),
                    })

        # Should have tracked some boxes
        assert len(tracked_boxes) > 0

        # All tracked boxes should meet minimum area
        for box in tracked_boxes:
            assert box["area"] >= min_area

    def test_model_ownership_transfer(self):
        """Create models via factory function (ownership transfer)."""
        models = []
        for i in range(10):
            model = memory_safety_demo.create_model(f"model_{i}", (i + 1) * 1000)
            assert model.is_valid()
            assert model.name() == f"model_{i}"
            models.append(model)

        # All models should still be valid
        for i, model in enumerate(models):
            assert model.is_valid()
            assert model.param_count() == (i + 1) * 1000

    def test_mixed_operations_stress(self):
        """Stress test mixing all safe types in a realistic workflow."""
        initial_buffers = memory_safety_demo.RAIIBuffer.active_count()

        for iteration in range(50):
            # Allocate working buffer
            work_buf = memory_safety_demo.RAIIBuffer(2048)

            # Process frames
            detections = []
            frame_data = [float(x) for x in range(256)]

            for frame_id in range(20):
                # Compute statistics on frame data
                total = safe_views.span_sum(frame_data)
                work_buf.set(frame_id, total)

                # Run detection
                det = memory_safety_demo.detect_in_frame(
                    iteration * 20 + frame_id
                )
                if det.has_value():
                    detections.append(det.value())

            # Verify buffer contents
            assert work_buf.get(0) == safe_views.span_sum(frame_data)

            # Clean up
            del work_buf
            del detections

        assert memory_safety_demo.RAIIBuffer.active_count() == initial_buffers
