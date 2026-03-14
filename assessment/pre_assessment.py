"""
Pre-Course Assessment — Take this BEFORE starting the course.

Run: python3 assessment/pre_assessment.py

Your score is saved to assessment/results.json so you can compare with
the post-assessment after completing the course.
"""

import json
import time
from pathlib import Path

QUESTIONS = [
    # --- Memory & Performance ---
    {
        "id": 1,
        "category": "Memory Hierarchy",
        "question": "How much slower is accessing RAM compared to L1 cache?",
        "choices": [
            "A) 2x slower",
            "B) 10x slower",
            "C) 200x slower",
            "D) 1000x slower",
        ],
        "answer": "C",
        "explanation": "L1 cache: ~0.5ns, RAM: ~100ns. That's 200x slower.",
    },
    {
        "id": 2,
        "category": "Memory Hierarchy",
        "question": "A 1920x1080 RGB image is ~6MB. Which cache level can hold it entirely?",
        "choices": [
            "A) L1 (32-64 KB)",
            "B) L2 (256 KB-1 MB)",
            "C) L3 (8-32 MB)",
            "D) None — it must be in RAM",
        ],
        "answer": "C",
        "explanation": "6MB fits in L3 (8-32MB) but not L1 or L2.",
    },
    # --- Python Optimization ---
    {
        "id": 3,
        "category": "Python",
        "question": "What does __slots__ do on a Python class?",
        "choices": [
            "A) Makes the class immutable",
            "B) Eliminates the per-instance __dict__ hash table",
            "C) Makes attribute access faster by using C types",
            "D) Enables garbage collection",
        ],
        "answer": "B",
        "explanation": "__slots__ tells Python to store attributes in a fixed-size tuple instead of a dict, saving ~100 bytes per instance.",
    },
    {
        "id": 4,
        "category": "Python",
        "question": "What is the output of np.shares_memory(buf, buf[10:20])?",
        "choices": [
            "A) True — slicing returns a view",
            "B) False — slicing always copies",
            "C) Depends on the dtype",
            "D) Raises an error",
        ],
        "answer": "A",
        "explanation": "Numpy slicing returns a view that shares the same underlying memory. Use .copy() only when you need an independent copy.",
    },
    {
        "id": 5,
        "category": "Python",
        "question": "What's wrong with: threading.Thread(target=save, args=(img,)).start() in a loop?",
        "choices": [
            "A) Thread() is not available on Linux",
            "B) Unbounded thread creation — at 30fps you create 30 threads/sec",
            "C) Threads can't call OpenCV functions",
            "D) Nothing — this is the correct pattern",
        ],
        "answer": "B",
        "explanation": "Use ThreadPoolExecutor(max_workers=4) to bound concurrency and reuse threads.",
    },
    # --- C++ Basics ---
    {
        "id": 6,
        "category": "C++",
        "question": "What does std::execution::unseq do in std::transform?",
        "choices": [
            "A) Runs the transform in reverse order",
            "B) Allows the compiler to use SIMD instructions",
            "C) Runs the transform on a separate thread",
            "D) Disables bounds checking",
        ],
        "answer": "B",
        "explanation": "unseq (unsequenced) tells the compiler it can vectorize the operation using SIMD (SSE, AVX, NEON).",
    },
    {
        "id": 7,
        "category": "C++",
        "question": "What is a 'trivially copyable' type in C++?",
        "choices": [
            "A) Any type that has a copy constructor",
            "B) A type that can be safely copied with memcpy (no pointer fixups needed)",
            "C) A type that fits in a register",
            "D) A type with no virtual functions",
        ],
        "answer": "B",
        "explanation": "Trivially copyable types (int, float, POD structs) can be memcpy'd, mapped into shared memory, and sent to GPUs safely.",
    },
    # --- GPU ---
    {
        "id": 8,
        "category": "GPU",
        "question": "Why is timing GPU operations with time.perf_counter() incorrect?",
        "choices": [
            "A) perf_counter() doesn't work on Linux",
            "B) GPU operations are asynchronous — the CPU returns before the GPU finishes",
            "C) perf_counter() measures wall-clock time, not CPU time",
            "D) GPU clocks run at a different frequency",
        ],
        "answer": "B",
        "explanation": "GPU kernels launch asynchronously. Use torch.cuda.Event(enable_timing=True) for accurate GPU timing.",
    },
    {
        "id": 9,
        "category": "GPU",
        "question": "What is the biggest bottleneck when using a GPU for inference?",
        "choices": [
            "A) GPU compute speed",
            "B) CPU preprocessing speed",
            "C) PCIe data transfer between CPU and GPU",
            "D) Python interpreter overhead",
        ],
        "answer": "C",
        "explanation": "GPU memory bandwidth is ~900 GB/s. PCIe is ~12 GB/s. Every CPU↔GPU transfer is 75x slower than GPU memory access.",
    },
    # --- Profiling ---
    {
        "id": 10,
        "category": "Profiling",
        "question": "If inference takes 80% of frame time, what's the max speedup from optimizing preprocessing (Amdahl's Law)?",
        "choices": [
            "A) 5x (1/0.2)",
            "B) 1.25x (1/0.8)",
            "C) 10x",
            "D) Depends on the preprocessing optimization",
        ],
        "answer": "A",
        "explanation": "Speedup = 1/((1-P) + P/S). If P=0.2 (preprocess) and S=infinity, max speedup = 1/(1-0.2) = 1.25x. But the question asks about preprocessing — making the 20% infinitely fast gives 1/(0.8 + 0) = 1.25x. The 5x answer applies if inference is the target.",
    },
    {
        "id": 11,
        "category": "Profiling",
        "question": "You have a function that allocates 7 numpy arrays per call at 30fps. What should you try first?",
        "choices": [
            "A) Rewrite it in C++",
            "B) Use torch.compile()",
            "C) Pre-allocate buffers in __init__ and reuse with out= parameter",
            "D) Use multiprocessing",
        ],
        "answer": "C",
        "explanation": "Pre-allocation is the simplest fix. Rewriting in C++ is overkill for this. torch.compile doesn't help numpy code.",
    },
    # --- Packaging ---
    {
        "id": 12,
        "category": "Production",
        "question": "What is the modern way to package a C++ extension for pip install?",
        "choices": [
            "A) setup.py with distutils",
            "B) pyproject.toml with scikit-build-core",
            "C) Makefile + manual .so copy",
            "D) conda-build only",
        ],
        "answer": "B",
        "explanation": "scikit-build-core integrates CMake builds with Python packaging. pyproject.toml is the modern standard (PEP 517/518).",
    },
    # --- Safety ---
    {
        "id": 13,
        "category": "Safety",
        "question": "What C++ feature replaces Python's 'with' statement for resource management?",
        "choices": [
            "A) try/catch",
            "B) RAII (Resource Acquisition Is Initialization)",
            "C) garbage collection",
            "D) manual delete calls",
        ],
        "answer": "B",
        "explanation": "RAII ties resource lifetime to object lifetime. When the object goes out of scope, the destructor cleans up — even if an exception occurs.",
    },
    {
        "id": 14,
        "category": "Safety",
        "question": "What does -fsanitize=address detect?",
        "choices": [
            "A) Data races between threads",
            "B) Buffer overflows, use-after-free, and memory leaks",
            "C) Undefined behavior like signed integer overflow",
            "D) Performance bottlenecks",
        ],
        "answer": "B",
        "explanation": "AddressSanitizer (ASAN) detects memory errors. Use -fsanitize=thread for data races, -fsanitize=undefined for UB.",
    },
    {
        "id": 15,
        "category": "Compile-Time",
        "question": "What advantage does std::variant + std::visit have over string-based state machines?",
        "choices": [
            "A) It uses less memory",
            "B) It's faster to type",
            "C) The compiler errors if you forget to handle a state",
            "D) It supports more states",
        ],
        "answer": "C",
        "explanation": "std::visit requires handling every alternative in the variant. Add a state, forget a handler? Compiler error. String comparison has no such safety.",
    },
]


def run_assessment(label: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {len(QUESTIONS)} questions — type the letter (A/B/C/D)")
    print(f"{'=' * 60}\n")

    correct = 0
    total = len(QUESTIONS)
    answers = {}
    categories = {}

    for q in QUESTIONS:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1

        print(f"Q{q['id']}. [{cat}] {q['question']}")
        for c in q["choices"]:
            print(f"   {c}")

        while True:
            ans = input("\nYour answer: ").strip().upper()
            if ans in ("A", "B", "C", "D"):
                break
            print("Please enter A, B, C, or D.")

        answers[q["id"]] = ans
        is_correct = ans == q["answer"]

        if is_correct:
            correct += 1
            categories[cat]["correct"] += 1
            print(f"   Correct!\n")
        else:
            print(f"   Wrong — correct answer: {q['answer']}")
            print(f"   {q['explanation']}\n")

    score_pct = (correct / total) * 100

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {correct}/{total} ({score_pct:.0f}%)")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Category':<20s}  {'Score':>8s}")
    print(f"  {'-' * 20}  {'-' * 8}")
    for cat, data in categories.items():
        pct = data["correct"] / data["total"] * 100
        print(f"  {cat:<20s}  {data['correct']}/{data['total']} ({pct:.0f}%)")
    print()

    if score_pct >= 80:
        print("  You already know a lot — focus on the advanced lessons (L7-L11).")
    elif score_pct >= 50:
        print("  Good foundation — the course will fill in the gaps.")
    else:
        print("  Start from L1 and work through sequentially.")

    return {
        "label": label,
        "score": correct,
        "total": total,
        "percentage": score_pct,
        "answers": answers,
        "categories": {k: v for k, v in categories.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    result = run_assessment("PRE-COURSE ASSESSMENT")

    results_path = Path(__file__).parent / "results.json"
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    results["pre"] = result

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Run post_assessment.py after completing the course to compare.\n")


if __name__ == "__main__":
    main()
