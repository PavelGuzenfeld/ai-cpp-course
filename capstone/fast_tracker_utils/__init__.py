"""
fast_tracker_utils — High-performance tracking utilities with C++ backends.

Four nanobind extension modules, one per component:
kalman_native, preprocess_native, history_native, state_machine_native.
See solution/tracker_fast.py for the Python-facing wrapper classes that
consume them (FastKalmanFilter, FastPreprocessor, FastHistoryBuffer,
FastStateMachine).
"""

from . import history_native, kalman_native, preprocess_native, state_machine_native

__all__ = [
    "kalman_native",
    "preprocess_native",
    "history_native",
    "state_machine_native",
]

__version__ = "1.0.0"
