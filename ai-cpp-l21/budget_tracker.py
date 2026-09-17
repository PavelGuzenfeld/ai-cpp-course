"""Worked example: budget one frame of the tracker loop against its floor.

Run it, then argue with it. The machine numbers below are from an Orin NX
(L6's cache table, L21 Part 1's tax table); the *node* numbers are compulsory
work counted off the algorithm. Both are inputs you are supposed to replace
with your own.

    python3 budget_tracker.py
"""

import sys

sys.path.insert(0, "/workspace/ai-cpp-l21")

from sol import (  # noqa: E402
    Machine, Node, budget_s, derive_node_sol, efficiency, fits_window,
    pipeline_sol_s, verdict,
)

FRAME_W, FRAME_H = 1920, 1080
RGB = 3
FLOAT32 = 4
WINDOW_S = 1.0 / 30.0  # 30 fps

# Orin NX. ops_per_s and bytes_per_s are sustained and under the contention
# this graph produces -- see machine_model.md sections 3 and 5. The tax is
# L21 Part 1's measured nanobind crossing, 55.8 ns.
ORIN = Machine(
    ops_per_s={"cpu": 8e9, "gpu": 400e9},
    bytes_per_s={"cpu": 12e9, "gpu": 60e9},
    tax_s_per_crossing=55.8e-9,
    dispatch_s=20e-6,
    completion_s=10e-6,
)

# Compulsory work only. The frame is read once and the tensor written once;
# anything the implementation re-reads is a design cost and would be its own
# node, which is exactly what makes those nodes visible.
PIPELINE = [
    Node("preprocess", "gpu",
         ops=FRAME_W * FRAME_H * RGB * 4,
         byte_count=FRAME_W * FRAME_H * RGB + FRAME_W * FRAME_H * RGB * FLOAT32,
         crossings=1),
    Node("inference", "gpu",
         ops=2 * 3.5e9,
         byte_count=FRAME_W * FRAME_H * RGB * FLOAT32,
         crossings=1),
    Node("postprocess", "cpu",
         ops=100 * 200.0,
         byte_count=100 * 6 * FLOAT32,
         crossings=1),
    # The node people forget: one syscall per detection.
    Node("publish (one sendto per box)", "cpu",
         ops=0.0, byte_count=100 * 64.0, crossings=100),
]

# What the pipeline actually costs today, measured per node. Replace with
# your own p99 -- these stand in for a real measurement run.
#
# `inference` is deliberately left inconsistent: 9 ms measured against a
# 17.5 ms floor. That is impossible, and the report says so rather than
# celebrating it. The op count here assumes 7 GFLOP per frame at a 400 GFLOP/s
# sustained rate; one of those two is wrong, and the exercise is to find out
# which before trusting any other row in the table.
MEASURED_S = {
    "preprocess": 3.0e-3,
    "inference": 9.0e-3,
    "postprocess": 1.2e-3,
    "publish (one sendto per box)": 4.0e-3,
}


def main():
    print(f"window {WINDOW_S * 1e3:.2f} ms at 30 fps, 25% margin reserved\n")
    print(f"{'node':<30} {'regime':<8} {'SOL ms':>8} {'meas ms':>8} {'of SOL':>7}")
    print("-" * 66)

    for node in PIPELINE:
        s = derive_node_sol(node, ORIN)
        m = MEASURED_S[node.name]
        print(f"{node.name:<30} {s.regime.value:<8} {s.sol_s * 1e3:>8.3f} "
              f"{m * 1e3:>8.3f} {efficiency(s, m):>7.0%}")

    total_sol = pipeline_sol_s(PIPELINE, ORIN)
    total_measured = sum(MEASURED_S.values())
    print("-" * 66)
    print(f"{'critical path':<30} {'':<8} {total_sol * 1e3:>8.3f} "
          f"{total_measured * 1e3:>8.3f}")
    print(f"\nfloor fits window minus margin: "
          f"{fits_window(PIPELINE, ORIN, WINDOW_S)}")

    print("\nper-node verdicts:")
    for node in PIPELINE:
        s = derive_node_sol(node, ORIN)
        print(f"  {node.name}:\n    {verdict(s, MEASURED_S[node.name])}")
        print(f"    budget at 70% of SOL: {budget_s(s) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
