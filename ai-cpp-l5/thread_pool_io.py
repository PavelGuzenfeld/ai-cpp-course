"""
Lesson 5: Thread pools vs. thread-per-task

Two approaches to saving images from a tracking loop:
  - save_images_threads(): spawns Thread() per image (tracker_engine style)
  - save_images_pool(): uses ThreadPoolExecutor with bounded concurrency

Uses simulated image saves (sleep) so no filesystem or OpenCV dependency.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


def _simulate_save(path: str, data: bytes, delay: float = 0.01) -> str:
    """Simulate an image save with a small delay (like cv2.imwrite)."""
    time.sleep(delay)
    return path


# ---------------------------------------------------------------------------
# 1. Thread-per-task (tracker_engine pattern)
# ---------------------------------------------------------------------------

def save_images_threads(
    items: list[tuple[str, bytes]],
    save_fn: Callable[[str, bytes], str] | None = None,
) -> list[str]:
    """Spawn a new Thread() for each image save.

    Problems:
      - Unbounded thread creation (30+ threads/second at 30 fps)
      - Thread creation overhead (~1 ms per thread on Linux)
      - No backpressure if I/O is slower than frame rate
      - Threads are not joined, results are lost
    """
    if save_fn is None:
        save_fn = _simulate_save

    threads: list[threading.Thread] = []
    results: list[str] = []
    lock = threading.Lock()

    def _worker(path: str, data: bytes) -> None:
        result = save_fn(path, data)
        with lock:
            results.append(result)

    for path, data in items:
        t = threading.Thread(target=_worker, args=(path, data))
        t.start()
        threads.append(t)

    # Join all threads (tracker_engine often doesn't do this)
    for t in threads:
        t.join()

    return results


# ---------------------------------------------------------------------------
# 2. ThreadPoolExecutor with bounded concurrency
# ---------------------------------------------------------------------------

def save_images_pool(
    items: list[tuple[str, bytes]],
    save_fn: Callable[[str, bytes], str] | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Use a thread pool with bounded concurrency.

    Benefits:
      - At most max_workers threads exist at any time
      - Thread reuse: no creation overhead after warmup
      - Backpressure: tasks queue when all workers are busy
      - Clean error handling via futures
    """
    if save_fn is None:
        save_fn = _simulate_save

    results: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(save_fn, path, data): path
            for path, data in items
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results


# ---------------------------------------------------------------------------
# Reusable ImageSaver class (production pattern)
# ---------------------------------------------------------------------------

class ImageSaver:
    """Long-lived thread pool for image saving.

    Usage:
        saver = ImageSaver(max_workers=4)
        for frame in video:
            saver.save(f"frame_{i}.png", image_bytes)
        saver.shutdown()  # wait for pending saves
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list = []

    def save(self, path: str, data: bytes) -> None:
        future = self._pool.submit(_simulate_save, path, data)
        self._pending.append(future)

    def shutdown(self, wait: bool = True) -> list[str]:
        results = []
        for f in self._pending:
            results.append(f.result())
        self._pool.shutdown(wait=wait)
        self._pending.clear()
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    # Generate fake image save tasks
    n_images = 100
    items = [(f"/tmp/frame_{i:04d}.png", b"\x00" * 1024) for i in range(n_images)]

    print(f"Saving {n_images} images...\n")

    # Thread-per-task
    t0 = time.perf_counter()
    results_threads = save_images_threads(items)
    dt_threads = time.perf_counter() - t0

    # Thread pool (4 workers)
    t0 = time.perf_counter()
    results_pool = save_images_pool(items, max_workers=4)
    dt_pool = time.perf_counter() - t0

    print(f"Thread-per-task: {dt_threads:.3f}s  ({len(results_threads)} saved)")
    print(f"ThreadPool(4):   {dt_pool:.3f}s  ({len(results_pool)} saved)")

    # Thread-per-task is actually faster here because all run in parallel,
    # but it uses 100 threads vs 4. The real cost shows up under load.
    print(f"\nThread-per-task peak threads: {n_images}")
    print(f"ThreadPool peak threads:      4")
    print(f"\nIn production at 30 fps for 1 minute:")
    print(f"  Thread-per-task: up to 1800 threads created")
    print(f"  ThreadPool(4):   4 threads total, tasks queued")


if __name__ == "__main__":
    _demo()
