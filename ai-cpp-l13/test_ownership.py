"""
Unit tests for Lesson 13: ownership of a C handle.

Tests:
  - Default construction owns and destroys exactly once
  - Move steals ownership and nulls the source
  - release() hands ownership back without destroying
  - The borrowed-pointer double-free bug, and its fix
"""

import sys

sys.path.insert(0, ".")

from ownership_native import (  # noqa: E402
    OwnedHandle,
    create_handle,
    double_destroy_count,
    live_count,
    move_assign_steals_and_nulls_source,
    move_ctor_steals_and_nulls_source,
    reset_for_test,
    wrap_borrowed_correctly,
    wrap_borrowed_incorrectly,
)


class TestOwnedHandleLifecycle:
    def test_construction_creates_one_live_handle(self):
        reset_for_test()
        h = OwnedHandle()
        assert h.owns() is True
        assert live_count() == 1

    def test_destruction_frees_exactly_once(self):
        reset_for_test()
        h = OwnedHandle()
        del h
        assert live_count() == 0
        assert double_destroy_count() == 0

    def test_move_ctor_steals_and_nulls_source(self):
        assert move_ctor_steals_and_nulls_source() is True

    def test_move_assign_steals_and_nulls_source(self):
        assert move_assign_steals_and_nulls_source() is True

    def test_release_prevents_destruction(self):
        reset_for_test()
        h = OwnedHandle()
        raw = h.release()
        assert h.owns() is False

        del h
        assert live_count() == 1  # release()'d handle was not destroyed

        # Clean up manually since release() handed ownership out.
        wrap_borrowed_incorrectly(raw)
        assert live_count() == 0


class TestBorrowedPointerDoubleFree:
    def test_wrapping_a_borrowed_handle_as_owned_double_frees(self):
        reset_for_test()
        real_owner = OwnedHandle()
        borrowed = real_owner.get()

        buggy_wrapper = wrap_borrowed_incorrectly(borrowed)
        del buggy_wrapper  # destroys the handle real_owner still thinks it owns

        del real_owner  # destroys the same handle again
        assert double_destroy_count() == 1

    def test_wrapping_a_borrowed_handle_as_non_owning_is_safe(self):
        reset_for_test()
        real_owner = OwnedHandle()
        borrowed = real_owner.get()

        correct_wrapper = wrap_borrowed_correctly(borrowed)
        assert correct_wrapper.owns() is False
        del correct_wrapper  # no-op: does not own the handle

        del real_owner  # the only destroy call for this handle
        assert double_destroy_count() == 0


class TestFreeStandingHandles:
    def test_create_handle_is_independent_of_wrapper(self):
        reset_for_test()
        h = create_handle()
        assert live_count() == 1

        wrap_borrowed_correctly(h)  # does not own -> no destroy on drop
        assert live_count() == 1
