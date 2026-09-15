"""
Whether this test passes or fails depends entirely on how the CMake option
LINKING_REGISTRY_SHARED was set at build time -- see README.md. Built
LINKING_REGISTRY_SHARED=OFF (the default), it fails, deterministically, on
every run: that failure is the point of this lesson, not a flake to chase.
"""

import threading

import module_a_native
import module_b_native


def test_touch_from_both_modules_lands_in_one_global_count():
    """An explicit barrier, not timing luck, forces both modules to touch
    the registry before either reads it back -- the assertion below is
    then either always true or always false, never a race.

    Baselined against current_count() before touching, not an absolute
    value: the underlying counter is process-global and does not reset
    between tests."""
    baseline = module_a_native.current_count()
    barrier = threading.Barrier(2)
    results = {}

    def touch(module, key):
        barrier.wait()
        results[key] = module.register_touch()

    t_a = threading.Thread(target=touch, args=(module_a_native, "a"))
    t_b = threading.Thread(target=touch, args=(module_b_native, "b"))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    # register_touch()'s own return value is the post-increment count -- it
    # must never come back equal to the pre-increment value.
    assert results["a"] != baseline
    assert results["b"] != baseline

    # Two touches happened. A single shared counter reflects both of them
    # from either module's view. Two independent counters each reflect only
    # their own module's touch -- current_count() reads back baseline + 1.
    assert module_a_native.current_count() == baseline + 2
    assert module_b_native.current_count() == baseline + 2
