# =============================================================================
# head_down_tracker.py
# EvilEye — Protect your sessions from evil doing | Head-Down Time Tracker
#
# Tracks:
#   - How many times the student's head went down (event count)
#   - Total cumulative seconds of head-down time
#   - Current continuous head-down streak duration
#   - Per-event log (start_time, end_time, duration_sec)
#
# Uses sustained-frame confirmation (same pattern as AlertTracker) to avoid
# counting brief accidental tilts.
#
# HEAD-DOWN EVENT LIFECYCLE:
#   CLEAR → PENDING (head=DOWN for >= MIN_DOWN_FRAMES) → ACTIVE
#   ACTIVE → ends when head=UP for >= MIN_UP_FRAMES consecutive frames
#
# Compatible with: Python 3.6 / no external dependencies
# =============================================================================

import time
from typing import Optional

# ============================================================================ #
#   CONFIGURABLE
# ============================================================================ #

MIN_DOWN_FRAMES = 10   # frames of DOWN before event starts (~0.8 s @ 12 FPS)
MIN_UP_FRAMES   = 8    # frames of UP before event ends  (~0.6 s)

# ============================================================================ #

class HeadDownEvent(object):
    """One confirmed head-down episode."""
    __slots__ = ['start_time', 'end_time', 'duration_sec']

    def __init__(self, start_time):
        self.start_time   = start_time
        self.end_time     = None
        self.duration_sec = 0.0


class HeadDownTracker(object):
    """
    Tracks head-down time for one student.

    Call update(head_state_str) once per frame.
    head_state_str: "UP" | "DOWN" | "UNKNOWN"

    Usage:
        tracker = HeadDownTracker()
        tracker.update("DOWN")   # call each frame
        print(tracker.get_summary())
    """

    def __init__(self):
        self._down_count   = 0     # consecutive DOWN frames
        self._up_count     = 0     # consecutive UP  frames (while event active)
        self._event_active = False
        self._current_ev   = None  # type: Optional[HeadDownEvent]

        self.events              = []    # list of completed HeadDownEvent
        self.total_down_secs     = 0.0
        self.event_count         = 0
        self.head_currently_down = False

        print("[HeadDownTracker] Initialised.")

    # ----------------------------------------------------------------------- #
    def update(self, head_state):
        """
        head_state : "UP" | "DOWN" | "UNKNOWN"
        Returns True if head is currently in an active down-event.
        """
        now = time.time()
        is_down = (head_state == "DOWN")

        if is_down:
            self._down_count += 1
            self._up_count    = 0
        else:
            self._up_count   += 1
            self._down_count  = 0

        # --- Start event ---
        if not self._event_active and self._down_count >= MIN_DOWN_FRAMES:
            self._event_active = True
            # Validate backdate is not negative (prevents race condition with time.time())
            backdate = max(0.0, now - MIN_DOWN_FRAMES / 12.0)
            ev = HeadDownEvent(backdate)
            self._current_ev = ev
            self.event_count += 1
            print("[HeadDownTracker] Head-down event #{} started.".format(
                self.event_count))

        # --- Update running duration ---
        if self._event_active and self._current_ev is not None:
            self._current_ev.duration_sec = now - self._current_ev.start_time

        # --- End event ---
        if self._event_active and self._up_count >= MIN_UP_FRAMES:
            ev = self._current_ev
            if ev is not None:
                ev.end_time      = now - MIN_UP_FRAMES / 12.0
                ev.duration_sec  = max(0, ev.end_time - ev.start_time)
                self.total_down_secs += ev.duration_sec
                self.events.append(ev)
                print("[HeadDownTracker] Head-down event #{} ended. "
                      "Duration: {:.1f}s".format(self.event_count,
                                                  ev.duration_sec))
            self._event_active = False
            self._current_ev   = None

        self.head_currently_down = self._event_active
        return self._event_active

    # ----------------------------------------------------------------------- #
    def get_current_streak_secs(self):
        """Seconds in the current active head-down streak, or 0."""
        if self._event_active and self._current_ev is not None:
            return time.time() - self._current_ev.start_time
        return 0.0

    # ----------------------------------------------------------------------- #
    def get_summary(self):
        """Return summary dict for the dashboard."""
        return {
            'event_count'       : self.event_count,
            'total_secs'        : round(self.total_down_secs, 1),
            'total_mins'        : round(self.total_down_secs / 60.0, 2),
            'current_active'    : self._event_active,
            'current_streak_sec': round(self.get_current_streak_secs(), 1),
            'events'            : [
                {
                    'start': time.strftime('%H:%M:%S',
                             time.localtime(ev.start_time)),
                    'duration_sec': round(ev.duration_sec, 1),
                }
                for ev in self.events
            ],
        }

    # ----------------------------------------------------------------------- #
    def reset(self):
        self._down_count     = 0
        self._up_count       = 0
        self._event_active   = False
        self._current_ev     = None
        self.events          = []
        self.total_down_secs = 0.0
        self.event_count     = 0
        self.head_currently_down = False
