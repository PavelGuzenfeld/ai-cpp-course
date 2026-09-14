"""
Integration tests for Lesson 13: a small pipeline that owns a pool of C
handles while transform functions borrow them without taking ownership --
the pattern from PRODUCTION_PLAN.md 1.1, and its inverse (the bug).
"""

import sys

sys.path.insert(0, ".")

from ownership_native import (  # noqa: E402
    OwnedHandle,
    double_destroy_count,
    live_count,
    reset_for_test,
    use_borrowed,
    wrap_borrowed_correctly,
    wrap_borrowed_incorrectly,
)


class TestPipelineOwnsPoolTransformsBorrow:
    def test_many_borrow_cycles_never_double_free(self):
        reset_for_test()
        pool = [OwnedHandle() for _ in range(8)]
        assert live_count() == 8

        for _ in range(50):
            for owner in pool:
                # A transform stage borrows the handle to do work, then
                # discards its (non-owning) wrapper -- the pool still owns it.
                use_borrowed(owner.get())
                borrowed_view = wrap_borrowed_correctly(owner.get())
                assert borrowed_view.owns() is False
                del borrowed_view

        assert live_count() == 8
        assert double_destroy_count() == 0

        del owner  # the loop variable above still refs pool[-1]
        pool.clear()
        assert live_count() == 0
        assert double_destroy_count() == 0

    def test_one_buggy_transform_corrupts_the_pool(self):
        """The delayed-symptom case: the bug fires when the pool releases
        its own handles later, not at the buggy transform call site."""
        reset_for_test()
        pool = [OwnedHandle() for _ in range(4)]

        # One transform stage wraps a borrowed handle as if it owned it.
        buggy = wrap_borrowed_incorrectly(pool[2].get())
        del buggy  # frees pool[2]'s handle out from under the pool

        assert live_count() == 3  # one handle gone, and the pool doesn't know

        pool.clear()  # the pool tries to free all 4 of its handles
        assert double_destroy_count() == 1  # pool[2] was freed twice
