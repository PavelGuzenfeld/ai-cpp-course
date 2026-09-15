"""
Integration tests for Lesson 18: concurrent use of the GIL-released native
modules, the scenario the TSan lane (see README) actually has to be clean
under -- a single-threaded pytest run gives TSan nothing to check.
"""

import threading

import numpy as np

from filter_native import box_blur, threshold_mask
from record_parser_native import parse_record

_VALID_RECORD = bytes([ord("R"), ord("C"), ord("1"), 0, 0, 3, 0, 6, 1, 2, 3])


def test_filter_functions_survive_concurrent_use_from_multiple_threads():
    rng = np.random.default_rng(1)
    image = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    errors = []

    def worker():
        try:
            for _ in range(200):
                blurred = box_blur(image, 2)
                threshold_mask(blurred, 0.5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_parser_survives_concurrent_use_from_multiple_threads():
    errors = []

    def worker():
        try:
            for _ in range(500):
                assert parse_record(_VALID_RECORD) is not None
                assert parse_record(b"bad") is None
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
