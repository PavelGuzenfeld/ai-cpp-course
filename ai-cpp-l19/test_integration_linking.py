"""
Integration test for Lesson 19: several rounds of concurrent registration
from both modules, not just one -- the divergence under static linking
compounds round over round rather than showing up once.
"""

import threading

import module_a_native
import module_b_native


def test_five_rounds_of_concurrent_touches_land_in_one_global_count():
    baseline = module_a_native.current_count()
    rounds = 5
    for round_index in range(rounds):
        barrier = threading.Barrier(2)

        def touch(module):
            barrier.wait()
            module.register_touch()

        t_a = threading.Thread(target=touch, args=(module_a_native,))
        t_b = threading.Thread(target=touch, args=(module_b_native,))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        expected = baseline + (round_index + 1) * 2
        assert module_a_native.current_count() == expected
        assert module_b_native.current_count() == expected
