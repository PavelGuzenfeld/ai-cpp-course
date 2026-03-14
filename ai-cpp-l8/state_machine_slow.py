"""
Pure Python string-based state machine — the 'before' version.

This mirrors the pattern found in tracker_engine where state is a string
and transitions are if/elif chains comparing strings. Every state check
is a string comparison, and typos in state names are silent bugs.
"""


class StringStateMachine:
    __slots__ = ("_state", "_lost_frames", "_target", "_last_known")

    def __init__(self):
        self._state = "idle"
        self._lost_frames = 0
        self._target = (0.0, 0.0, 0.0, 0.0)
        self._last_known = (0.0, 0.0, 0.0, 0.0)

    def update(
        self,
        has_detection: bool,
        det_x: float = 0.0,
        det_y: float = 0.0,
        det_w: float = 0.0,
        det_h: float = 0.0,
    ):
        if self._state == "idle":
            if has_detection:
                self._state = "tracking"
                self._target = (det_x, det_y, det_w, det_h)

        elif self._state == "tracking":
            if has_detection:
                self._target = (det_x, det_y, det_w, det_h)
            else:
                self._state = "lost"
                self._lost_frames = 1
                self._last_known = self._target

        elif self._state == "lost":
            if has_detection:
                self._state = "tracking"
                self._target = (det_x, det_y, det_w, det_h)
                self._lost_frames = 0
            else:
                self._lost_frames += 1
                if self._lost_frames > 30:
                    self._state = "search"

        elif self._state == "search":
            if has_detection:
                self._state = "tracking"
                self._target = (det_x, det_y, det_w, det_h)
                self._lost_frames = 0
            # else: stay in search

    @property
    def state(self) -> str:
        return self._state

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    @property
    def target(self) -> tuple:
        return self._target
