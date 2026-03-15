# Promo Video Script (~2 minutes)

## Opening (0:00 - 0:15)

[Screen: Python code running a tracker at 8 FPS, visibly laggy]

"Your Python tracker works. But at 8 frames per second, it is useless for
production. The conventional wisdom says rewrite it in C++. That is the wrong
answer."

## The Problem (0:15 - 0:35)

[Screen: side-by-side code — the 4 anti-patterns from L7]

"The real problem is not Python itself. It is four specific bottlenecks:
per-frame memory allocation, CPU-side preprocessing, copy-heavy data structures,
and string-based state machines. Fix these four things and your Python code
runs at 60 FPS."

## The Solution (0:35 - 1:00)

[Screen: terminal showing benchmark output]

"In this course you will write surgical C++ replacements for each bottleneck.
Not a rewrite — Python still runs the show. The C++ parts slot in as regular
Python imports."

[Show: `from fast_tracker_utils import FastPreprocessor`]

"Your fused CUDA kernel replaces three NumPy operations with one GPU launch.
Result: 30x faster preprocessing."

[Show: benchmark numbers scrolling]

## Jetson (1:00 - 1:20)

[Screen: Jetson Orin running the tracker]

"Deploying on NVIDIA Jetson? The optimization strategy is different. Jetson
shares memory between CPU and GPU — no PCIe bottleneck. This course covers
Jetson-specific techniques: unified memory, power modes, multi-process GPU
pipelines. Everything runs in Docker, on any Jetson from Nano to Orin."

## What You Get (1:20 - 1:45)

[Screen: course overview scrolling]

"11 lessons, 4 assignments, a capstone project, and 300 automated tests that
verify your work. Docker environment — zero setup. Real code from a real UAV
tracking system, not toy examples."

[Screen: test output — "306 passed"]

## Close (1:45 - 2:00)

[Screen: before/after FPS comparison]

"Stop rewriting everything in C++. Learn to replace only what matters."

[Title card: "Production C++ for CV/AI Python Developers — Enroll Now"]
