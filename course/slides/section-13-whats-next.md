# Section 13: What's Next -- Wrap-Up and Further Resources

## Video 13.1: Course Recap (~10 min)

### Slides
- Slide 1: The journey -- From "Python developer who needs performance" to "developer who can surgically replace hot paths with C++ and ship the result as a pip-installable package."
- Slide 2: Foundation skills (Sections 1-4) -- SIMD vector operations (50-100x over Python loops), cache-aware image processing (row-major traversal), shared memory IPC (0.1 us latency), nanobind zero-copy bindings (3-5x smaller than pybind11).
- Slide 3: Optimization skills (Sections 5-7) -- Python optimization (__slots__, views, pre-allocation), hardware measurement (perf_counter_ns, torch.cuda.Event, cache benchmarks), GPU programming (fused kernels, pinned memory, CUDA streams, CUDA IPC).
- Slide 4: Production skills (Sections 8-11) -- Compile-time concepts (constexpr LUTs, variant state machines), scikit-build-core packaging (pip install, type stubs, Docker), profiling workflow (Amdahl's Law, the optimization loop), memory safety (span, optional, RAII, smart pointers, ASAN).
- Slide 5: The capstone -- All skills applied to a real tracker pipeline. Profile, identify bottlenecks, implement C++ replacements, package as pip library, measure >2x speedup per component.
- Slide 6: Jetson throughout -- Unified memory changes GPU programming fundamentally. ARM NEON vs x86 AVX for SIMD. Thermal throttling affects benchmarks. Smaller caches demand more cache-aware code. Same C++ source, different optimization priorities.

### Key Takeaway
- You now have a complete toolbox: profile, identify, optimize with C++, package for production -- the full cycle from measurement to deployment.

## Video 13.2: Further Resources and Deep Dives (~8 min)

### Slides
- Slide 1: C++ learning resources -- "A Tour of C++" by Bjarne Stroustrup (concise overview), cppreference.com (the definitive reference), "C++ Core Guidelines" (best practices by Stroustrup and Sutter), Compiler Explorer (godbolt.org) for seeing generated assembly.
- Slide 2: Performance and optimization -- "What Every Programmer Should Know About Memory" by Ulrich Drepper (free paper on memory hierarchy), Brendan Gregg's performance analysis tools and books, Denis Bakhvalov's "Performance Analysis and Tuning on Modern CPUs."
- Slide 3: GPU programming -- NVIDIA CUDA Programming Guide (official documentation), "Programming Massively Parallel Processors" by Kirk and Hwu, NVIDIA Nsight Systems and Nsight Compute documentation, Jetson developer forums and JetPack documentation.
- Slide 4: Python packaging and bindings -- nanobind documentation and examples, scikit-build-core documentation, PyPA (Python Packaging Authority) guides, pybind11 documentation (for legacy projects).
- Slide 5: Communities -- CppCon talks (YouTube, annual conference), Python performance discussions on discuss.python.org, NVIDIA developer forums for GPU and Jetson topics, r/cpp and r/Python subreddits.

### Key Takeaway
- This course is a foundation -- the resources listed here let you go deeper into any area that matters most for your specific project.

## Video 13.3: Career Paths and Applying What You Learned (~8 min)

### Slides
- Slide 1: Career path A -- CV/AI performance engineer. Companies deploying real-time computer vision (autonomous vehicles, drones, robotics, security cameras) need engineers who bridge Python ML and production C++. This course is directly applicable.
- Slide 2: Career path B -- Embedded AI engineer (Jetson/edge). NVIDIA Jetson, Qualcomm, Intel Movidius platforms. Deploy models on constrained hardware. Optimize for power, thermal, and latency budgets. Unified memory, ARM NEON, TensorRT optimization.
- Slide 3: Career path C -- ML infrastructure engineer. Build the tools and pipelines that ML researchers use. Package management, Docker distribution, CI/CD for compiled extensions, profiling infrastructure.
- Slide 4: Applying this to your own projects -- Step 1: Profile your existing Python pipeline (Lesson 10 workflow). Step 2: Identify the dominant cost (Amdahl's Law). Step 3: Choose the right technique (is it a Python optimization or does it need C++?). Step 4: Implement, measure, iterate. Step 5: Package and ship (Lesson 9).
- Slide 5: Final advice -- Do not rewrite everything in C++. Surgical optimization beats total rewrites. Profile first, always. Ship early, iterate. The best optimization is the one that ships.

### Key Takeaway
- The most valuable skill from this course is not any single C++ technique -- it is the discipline of profiling first, optimizing surgically, and knowing when to stop.

## Video 13.4: Thank You and Next Steps (~5 min)

### Slides
- Slide 1: What to do right now -- If you have not completed the capstone, do it. Apply the optimization loop to a real project of your own. Profile something you work on daily.
- Slide 2: Stay connected -- Course repository for updates and community contributions. File issues for bugs or unclear content. Contribute exercises or real-world examples.
- Slide 3: The one thing to remember -- Measure, optimize, measure again. The difference between a good engineer and a great one is not knowing more tricks -- it is knowing which trick to apply and when to stop.

### Key Takeaway
- Measure, optimize, measure again -- this discipline is more valuable than any individual technique.
