# Target Audience Profiles

## Primary: Python CV/AI Developer

**Who they are**: Software engineers at companies building computer vision or ML
products. They write Python daily (OpenCV, PyTorch, TensorFlow). Their code works
but does not meet latency or throughput requirements for production.

**Pain point**: "My Python tracker runs at 8 FPS. Product needs 30 FPS. My manager
says rewrite in C++ but that would take 6 months and I do not know C++."

**What they want**: Keep their Python workflow, make the slow parts fast.

**Why they buy**: The course shows them how to get 10-50x speedups on specific hot
paths without rewriting their entire codebase.

## Secondary: Jetson Edge Developer

**Who they are**: Engineers deploying AI models on NVIDIA Jetson devices for
drones, robots, autonomous vehicles, smart cameras, or industrial inspection.

**Pain point**: "My inference pipeline runs fine on a desktop GPU but is too slow
on Jetson Orin. I need to squeeze every drop of performance from 8GB of shared
memory and 1024 CUDA cores."

**What they want**: Jetson-specific optimization techniques. Unified memory
strategies. Multi-process pipelines that maximize GPU utilization on constrained
hardware.

**Why they buy**: Most C++ and CUDA courses target desktop/server GPUs. This course
explicitly covers Jetson hardware, power modes, and the unified memory architecture
that makes Jetson optimization fundamentally different.

## Tertiary: ML Ops / Infrastructure Engineer

**Who they are**: Engineers who package and deploy ML models. They deal with
Docker, CI/CD, wheel building, and production reliability.

**Pain point**: "The ML team hands me a Python script and says 'deploy this.'
The script imports C++ extensions that only build on their laptop."

**What they want**: Learn how to package C++ extensions as proper pip wheels,
build multi-platform Docker images, and set up CI that catches build failures
before production.

**Why they buy**: Lessons 9 (packaging) and the capstone project teach exactly
this workflow end-to-end.

## Marketing Angles by Audience

| Audience | Hook | Key sections |
|----------|------|-------------|
| Python CV dev | "Make your Python tracker 10x faster without leaving Python" | L1-L5, L10 |
| Jetson developer | "Squeeze maximum FPS from your Jetson Orin" | L3, L7, Jetson guide |
| ML Ops engineer | "Package C++ extensions as pip wheels that actually work" | L9, Capstone |

## Keywords for Udemy SEO

- Python C++ performance optimization
- pybind11 nanobind tutorial
- CUDA programming for Python developers
- NVIDIA Jetson optimization
- Computer vision real-time processing
- Shared memory IPC Python C++
- scikit-build-core wheel packaging
- GPU programming Python
- Edge AI deployment Jetson Orin
- Production C++ for Python developers
