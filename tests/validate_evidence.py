"""
Evidence Validation: Verify lesson claims match actual runtime behavior.

This script checks that every factual claim in the lesson markdown files
is supported by actual code execution results.
"""
import sys
import os
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-cpp-l5'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-cpp-l4'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-cpp-l8'))

passed = 0
failed = 0


def check(claim, result, detail=""):
    global passed, failed
    if result:
        print(f"  CONFIRMED: {claim}")
        passed += 1
    else:
        print(f"  CONTRADICTED: {claim} -- {detail}")
        failed += 1


print("=" * 60)
print("EVIDENCE VALIDATION: Lesson claims vs actual results")
print("=" * 60)

# =====================================================================
# L5: __slots__ memory savings
# =====================================================================
print("\n--- L5: __slots__ memory savings ---")
from bbox_slots import BboxSlow, BboxSlots, BboxDataclass

n = 100_000

tracemalloc.start()
slow_boxes = [BboxSlow(float(i), float(i), 10.0, 10.0) for i in range(n)]
mem_slow, _ = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
fast_boxes = [BboxSlots(float(i), float(i), 10.0, 10.0) for i in range(n)]
mem_fast, _ = tracemalloc.get_traced_memory()
tracemalloc.stop()

savings_pct = (1 - mem_fast / mem_slow) * 100
print(f"  BboxSlow: {mem_slow / 1024:.0f} KB, BboxSlots: {mem_fast / 1024:.0f} KB")
print(f"  Actual savings: {savings_pct:.0f}%")
check("__slots__ saves significant memory (lesson claims ~56%)",
      savings_pct > 30,
      f"only {savings_pct:.0f}%")

# Keep refs alive
del slow_boxes, fast_boxes

# =====================================================================
# L5: __slots__ eliminates __dict__
# =====================================================================
print("\n--- L5: __slots__ eliminates __dict__ ---")
check("BboxSlow has __dict__", hasattr(BboxSlow(1, 2, 3, 4), "__dict__"))
check("BboxSlots has no __dict__", not hasattr(BboxSlots(1, 2, 3, 4), "__dict__"))
check("BboxDataclass has no __dict__", not hasattr(BboxDataclass(1, 2, 3, 4), "__dict__"))

# =====================================================================
# L5: numpy views share memory, copies don't
# =====================================================================
print("\n--- L5: numpy views vs copies ---")
from numpy_views import HistoryCopy, HistoryView

hc = HistoryCopy(capacity=20, cols=4)
hv = HistoryView(capacity=20, cols=4)
for i in range(10):
    row = np.array([i, i + 1, i + 2, i + 3], dtype=np.float64)
    hc.push(row)
    hv.push(row)

view_result = hv.latest(5)
copy_result = hc.latest(5)

check("HistoryView.latest() shares memory with buffer",
      np.shares_memory(hv._buf, view_result))
check("HistoryCopy.latest() does NOT share memory with buffer",
      not np.shares_memory(hc._buf, copy_result))

# =====================================================================
# L5: VelocityTracker slow == fast
# =====================================================================
print("\n--- L5: VelocityTracker correctness ---")
from preallocated_buffers import VelocityTrackerSlow, VelocityTrackerFast

rng = np.random.default_rng(42)
positions = np.cumsum(rng.normal(3.0, 0.5, size=(50, 2)), axis=0)

slow_vt = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
fast_vt = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

v_slow = slow_vt.compute_velocity(positions)
v_fast = fast_vt.compute_velocity(positions)

check("Slow and fast produce identical velocity",
      np.isclose(v_slow, v_fast),
      f"slow={v_slow}, fast={v_fast}")
check("Both agree on is_target_moving",
      slow_vt.is_target_moving(positions) == fast_vt.is_target_moving(positions))

# =====================================================================
# L5: ThreadPool saves all images
# =====================================================================
print("\n--- L5: ThreadPool correctness ---")
from thread_pool_io import save_images_threads, save_images_pool

items = [(f"/tmp/frame_{i}.png", b"\x00" * 64) for i in range(30)]
results_t = save_images_threads(items)
results_p = save_images_pool(items, max_workers=4)

check("Thread-per-task saves all 30 images", len(results_t) == 30,
      f"saved {len(results_t)}")
check("ThreadPool saves all 30 images", len(results_p) == 30,
      f"saved {len(results_p)}")

# =====================================================================
# L4: BBox properties
# =====================================================================
print("\n--- L4: BBox properties ---")
from bbox_slow import BBox

b = BBox(0, 0, 10, 10)
check("cx = x + w/2", b.cx == 5.0, f"got {b.cx}")
check("cy = y + h/2", b.cy == 5.0, f"got {b.cy}")
check("area = w * h", b.area == 100.0, f"got {b.area}")

b1 = BBox(0, 0, 10, 10)
b2 = BBox(5, 5, 10, 10)
expected_iou = 25.0 / 175.0
check("IOU(overlapping boxes) = intersection/union",
      abs(b1.iou(b2) - expected_iou) < 1e-10,
      f"got {b1.iou(b2)}, expected {expected_iou}")
check("IOU is symmetric", abs(b1.iou(b2) - b2.iou(b1)) < 1e-10)
check("IOU(identical) = 1.0", abs(b1.iou(b1) - 1.0) < 1e-10)

b3 = BBox(20, 20, 10, 10)
check("IOU(non-overlapping) = 0.0", b1.iou(b3) == 0.0)

check("contains_point inside", b.contains_point(5, 5))
check("contains_point outside", not b.contains_point(15, 15))

# =====================================================================
# L8: State machine transitions
# =====================================================================
print("\n--- L8: State machine transitions ---")
from state_machine_slow import StringStateMachine

sm = StringStateMachine()
check("Initial state = idle", sm.state == "idle")

sm.update(True, 10, 20, 30, 40)
check("idle + detection -> tracking", sm.state == "tracking")

sm.update(True, 15, 25, 35, 45)
check("tracking + detection -> tracking (update target)", sm.state == "tracking")

sm.update(False)
check("tracking + no detection -> lost", sm.state == "lost")
check("lost_frames = 1 after first loss", sm.lost_frames == 1)

for _ in range(30):
    sm.update(False)
check("lost for 31 frames -> search", sm.state == "search")

sm.update(False)
check("search + no detection -> search (stays)", sm.state == "search")

sm.update(True, 50, 60, 70, 80)
check("search + detection -> tracking", sm.state == "tracking")
check("lost_frames resets to 0", sm.lost_frames == 0)

# =====================================================================
# L5 lesson table claims
# =====================================================================
print("\n--- L5: Lesson summary table claims ---")
import sys as _sys

slow_sizeof = _sys.getsizeof(BboxSlow(1, 2, 3, 4))
if hasattr(BboxSlow(1, 2, 3, 4), "__dict__"):
    slow_sizeof += _sys.getsizeof(BboxSlow(1, 2, 3, 4).__dict__)
slots_sizeof = _sys.getsizeof(BboxSlots(1, 2, 3, 4))

print(f"  BboxSlow total size: {slow_sizeof} bytes")
print(f"  BboxSlots total size: {slots_sizeof} bytes")
check("Slots version is smaller than dict version", slots_sizeof < slow_sizeof)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} confirmed, {failed} contradicted")
print("=" * 60)

if failed > 0:
    sys.exit(1)
