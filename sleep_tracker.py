# sleep_tracker.py
# EvilEye — Protect your sessions from evil doing | Sleep Event Tracker
#
# Tracks:
#   - How many times the participant fell asleep (event count)
#   - Total cumulative seconds of sleep time
#   - Current continuous sleep streak duration
#   - Per-event log (start_time, end_time, duration_sec)
#
# Uses sustained-frame confirmation to avoid counting brief blinks as sleep.
#
# SLEEP EVENT LIFECYCLE:
#   CLEAR → PENDING (sleep_state=SLEEPING for >= MIN_SLEEP_FRAMES) → ACTIVE
#   ACTIVE → ends when sleep_state=AWAKE for >= MIN_AWAKE_FRAMES consecutive frames
#
# Memory-efficient for Jetson Nano: no image storage, just timestamps and counters.
# Compatible with: Python 3.6 / no external dependencies
# =============================================================================

import time

# ============================================================================ #
#   CONFIGURABLE
# ============================================================================ #

MIN_SLEEP_FRAMES = 12   # frames of SLEEPING before event starts (~1s @ 12 FPS) - behavior_analyzer already confirmed 25s
MIN_AWAKE_FRAMES = 12   # frames of AWAKE before event ends  (~1s) - fast recovery from temporary head movement

# ============================================================================ #

class SleepEvent(object):
    """One confirmed sleep episode."""
    __slots__ = ['start_time', 'end_time', 'duration_sec']

    def __init__(self, start_time):
        self.start_time   = start_time
        self.end_time     = None
        self.duration_sec = 0.0


class SleepTracker(object):
    """
    Tracks sleep time for one participant.

    Call update(sleep_state_str) once per frame.
    sleep_state_str: "SLEEPING" | "AWAKE" | "UNKNOWN"

    Usage:
        tracker = SleepTracker()
        tracker.update("SLEEPING")   # call each frame
        print(tracker.get_summary())
    """

    def __init__(self):
        self._sleep_count    = 0     # consecutive SLEEPING frames
        self._awake_count    = 0     # consecutive AWAKE frames (while event active)
        self._event_active   = False
        self._current_ev     = None  # SleepEvent or None

        self.events                    = []    # list of completed SleepEvent
        self.total_sleep_secs          = 0.0
        self.event_count               = 0
        self.participant_currently_sleeping = False

        print("[SleepTracker] Initialised.")

    # ----------------------------------------------------------------------- #
    def update(self, sleep_state):
        """
        sleep_state : "SLEEPING" | "AWAKE" | "UNKNOWN"
        Returns True if participant is currently in an active sleep event.
        """
        now = time.time()
        is_sleeping = (sleep_state == "SLEEPING")

        if is_sleeping:
            self._sleep_count += 1
            self._awake_count  = 0
        else:
            self._awake_count += 1
            self._sleep_count  = 0

        # --- Start event ---
        if not self._event_active and self._sleep_count >= MIN_SLEEP_FRAMES:
            self._event_active = True
            ev = SleepEvent(now - MIN_SLEEP_FRAMES / 12.0)  # backdate
            self._current_ev = ev
            self.event_count += 1
            print("[SleepTracker] ✓ Sleep event #{} STARTED (frames: {}≥{}).".format(
                self.event_count, self._sleep_count, MIN_SLEEP_FRAMES))

        # --- Update running duration ---
        if self._event_active and self._current_ev is not None:
            self._current_ev.duration_sec = now - self._current_ev.start_time

        # --- End event ---
        if self._event_active and self._awake_count >= MIN_AWAKE_FRAMES:
            ev = self._current_ev
            if ev is not None:
                ev.end_time      = now - MIN_AWAKE_FRAMES / 12.0
                ev.duration_sec  = max(0, ev.end_time - ev.start_time)
                self.total_sleep_secs += ev.duration_sec
                self.events.append(ev)
                print("[SleepTracker] Sleep event #{} ended. "
                      "Duration: {:.1f}s".format(self.event_count,
                                                  ev.duration_sec))
            self._event_active = False
            self._current_ev   = None

        self.participant_currently_sleeping = self._event_active
        return self._event_active

    # ----------------------------------------------------------------------- #
    def get_current_streak_secs(self):
        """Seconds in the current active sleep streak, or 0."""
        if self._event_active and self._current_ev is not None:
            return time.time() - self._current_ev.start_time
        return 0.0

    # ----------------------------------------------------------------------- #
    def get_summary(self):
        """Return summary dict for the dashboard."""
        return {
            'event_count'       : self.event_count,
            'total_secs'        : round(self.total_sleep_secs, 1),
            'total_mins'        : round(self.total_sleep_secs / 60.0, 2),
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
        self._sleep_count         = 0
        self._awake_count         = 0
        self._event_active        = False
        self._current_ev          = None
        self.events               = []
        self.total_sleep_secs     = 0.0
        self.event_count          = 0
        self.participant_currently_sleeping = False
