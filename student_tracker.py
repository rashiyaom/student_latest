# =============================================================================
# student_tracker.py
# ClassPulse AI — Student Session Orchestrator
#
# This is the central state object for Phase 3 (single-student mode).
# It owns and coordinates:
#   - BehaviourAnalyzer    (yawn, head, eye, attention)
#   - PhoneDetector        (phone usage events + screenshots)
#   - HeadDownTracker      (cumulative head-down time)
#
# It also maintains:
#   - Session-level statistics (yawn count, total attentive %, phone minutes)
#   - Timeseries for the dashboard (attention over time, yawn markers)
#   - A JSON-serialisable state dict pushed to the dashboard every N frames
#
# OUTPUT FORMAT — state dict published to dashboard:
#   {
#     "ts"              : "14:32:05",
#     "session_secs"    : 1234,
#     "present"         : true,
#     "face_found"      : true,
#
#     "attention_score" : 0.78,
#     "attention_label" : "ATTENTIVE",
#     "attention_pct"   : 78,       # session-average attention %
#
#     "eye_state"       : "OPEN",
#     "head_state"      : "DOWN",
#     "yawn_state"      : "YAWNING",
#
#     "yawn_count"      : 4,        # total confirmed yawns this session
#     "yawn_events"     : [...],    # [{time, duration_sec}]
#
#     "head_down_count" : 2,
#     "head_down_total_mins" : 1.3,
#     "head_down_current_secs": 0.0,
#
#     "phone_active"    : false,
#     "phone_events"    : [...],
#     "phone_total_secs": 45.0,
#     "phone_current_secs": 0.0,
#
#     "attention_ts"    : [{"t":"14:31:00","v":0.82}, ...],  # timeseries
#     "yawn_markers"    : ["14:31:45", ...],                 # yawn timestamps
#   }
#
# Compatible with: Python 3.6 / no extra dependencies
# =============================================================================

import time
import json
import os
import tempfile
import shutil

from behavior_analyzer import BehaviourAnalyzer
from phone_detector    import PhoneDetector
from head_down_tracker import HeadDownTracker
from sleep_tracker     import SleepTracker

# ============================================================================ #
#   CONFIGURABLE
# ============================================================================ #

TIMESERIES_MAX_LEN  = 120   # keep last 120 attention samples (~2 min at 1/s)
PUBLISH_INTERVAL_S  = 1.0   # seconds between dashboard state publishes
STATE_FILE          = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'outputs', 'state.json')

# Yawn event — must last this many frames to count as ONE yawn
# (handled inside BehaviourAnalyzer; this is for dedup at tracker level)

# ============================================================================ #

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)


class StudentTracker(object):
    """
    Owns all per-student tracking modules and publishes JSON state.

    Usage:
        tracker = StudentTracker(student_name="Alex")
        tracker.start_session()

        # Each frame:
        state = tracker.update(
            person_roi_bgr,   # crop of detected person (or None)
            person_box,       # [x1,y1,x2,y2] or None
            all_detections,   # full list including phone class
            frame_bgr         # full webcam frame (for screenshots)
        )

        tracker.end_session()
    """

    def __init__(self, student_name="Student"):
        self.student_name = student_name

        self._behavior  = BehaviourAnalyzer()
        self._phone     = PhoneDetector()
        self._head_down = HeadDownTracker()
        self._sleep     = SleepTracker()

        self._session_start = None
        self._session_active= False

        # Cumulative attention tracking
        self._total_frames      = 0
        self._attentive_frames  = 0
        self._present_frames    = 0

        # Yawn dedup — track last yawn state to count transitions
        self._prev_yawn_state   = "NO_YAWN"
        self._yawn_count        = 0
        self._yawn_events       = []   # list of {time, duration_sec}
        self._yawn_markers      = []   # timestamps for chart markers

        # Attention timeseries
        self._attention_ts      = []   # deque-like list, capped at max len

        # Dashboard publish
        self._last_publish_time = 0.0

        _ensure_dir(os.path.dirname(STATE_FILE))
        print("[StudentTracker] Initialised for student: {}".format(student_name))

    # ----------------------------------------------------------------------- #
    def start_session(self):
        self._session_start  = time.time()
        self._session_active = True
        self._behavior.reset()
        self._phone.reset()
        self._head_down.reset()
        self._sleep.reset()
        self._total_frames     = 0
        self._attentive_frames = 0
        self._present_frames   = 0
        self._yawn_count       = 0
        self._yawn_events      = []
        self._yawn_markers     = []
        self._attention_ts     = []
        self._prev_yawn_state  = "NO_YAWN"
        print("[StudentTracker] Session started.")

    # ----------------------------------------------------------------------- #
    def update(self, person_roi_bgr, person_box, all_detections, frame_bgr):
        """
        Call once per frame.

        person_roi_bgr : BGR crop of the person (or None if not present)
        person_box     : [x1,y1,x2,y2] in frame coords (or None)
        all_detections : list of [x1,y1,x2,y2,conf,class_id]
        frame_bgr      : full BGR frame (for screenshots)

        Returns the current state dict.
        """
        now = time.time()
        person_present = person_roi_bgr is not None

        self._total_frames += 1
        if person_present:
            self._present_frames += 1

        # ------------------------------------------------------------------ #
        # 1. Behaviour analysis
        # ------------------------------------------------------------------ #
        br = self._behavior.analyse(person_roi_bgr, person_present)

        # ------------------------------------------------------------------ #
        # 2. Yawn counting — count transitions NO_YAWN → YAWNING
        # ------------------------------------------------------------------ #
        if (br.yawn_state == "YAWNING" and
                self._prev_yawn_state == "NO_YAWN"):
            self._yawn_count += 1
            # Use time with milliseconds to prevent collisions
            timestamp = time.strftime('%H:%M:%S') + f'.{int((time.time() % 1) * 1000):03d}'
            self._yawn_markers.append(timestamp)
            self._yawn_events.append({
                'time'        : timestamp,
                'duration_sec': getattr(br, 'yawn_duration_sec', 0.5),  # Use actual duration if available
            })
            print("[StudentTracker] Yawn #{} detected at {} (duration: {:.2f}s)".format(
                self._yawn_count, self._yawn_markers[-1], 
                getattr(br, 'yawn_duration_sec', 0.5)))
        
        # Update last event duration if yawn ended
        if (br.yawn_state == "NO_YAWN" and 
                self._prev_yawn_state == "YAWNING" and 
                self._yawn_events and 
                hasattr(br, 'yawn_duration_sec')):
            self._yawn_events[-1]['duration_sec'] = br.yawn_duration_sec
        
        self._prev_yawn_state = br.yawn_state

        # ------------------------------------------------------------------ #
        # 3. Head-down tracking
        # ------------------------------------------------------------------ #
        self._head_down.update(br.head_state)

        # ------------------------------------------------------------------ #
        # 3b. Sleep tracking
        # ------------------------------------------------------------------ #
        self._sleep.update(br.sleep_state)

        # ------------------------------------------------------------------ #
        # 4. Phone detection
        # ------------------------------------------------------------------ #
        phone_visible, new_phone_event = self._phone.update(
            all_detections, person_box, frame_bgr)

        # ------------------------------------------------------------------ #
        # 5. Attention accumulation
        # ------------------------------------------------------------------ #
        if person_present and br.face_found:
            if br.attention == "ATTENTIVE":
                self._attentive_frames += 1

        # ------------------------------------------------------------------ #
        # 6. Attention timeseries (throttled to ~1 sample/sec)
        # ------------------------------------------------------------------ #
        if now - self._last_publish_time >= PUBLISH_INTERVAL_S:
            sample = {
                't': time.strftime('%H:%M:%S'),
                'v': round(br.attention_score, 3),
            }
            self._attention_ts.append(sample)
            if len(self._attention_ts) > TIMESERIES_MAX_LEN:
                self._attention_ts.pop(0)

        # ------------------------------------------------------------------ #
        # 7. Build state dict
        # ------------------------------------------------------------------ #
        session_secs    = int(now - self._session_start) if self._session_start else 0
        pf              = self._present_frames or 1
        att_pct_session = round(self._attentive_frames / pf * 100, 1)

        head_summary  = self._head_down.get_summary()
        sleep_summary = self._sleep.get_summary()
        phone_summary = self._phone.get_summary()

        state = {
            'ts'                    : time.strftime('%H:%M:%S'),
            'session_secs'          : session_secs,
            'student_name'          : self.student_name,
            'present'               : person_present,
            'face_found'            : br.face_found,

            'attention_score'       : br.attention_score,
            'attention_label'       : br.attention,
            'attention_pct'         : att_pct_session,

            'eye_state'             : br.eye_state,
            'head_state'            : br.head_state,
            'sleep_state'           : br.sleep_state,
            'yawn_state'            : br.yawn_state,
            'mouth_open_raw'        : round(br.mouth_open_raw, 3),
            'eye_open_raw'          : round(br.eye_open_raw, 3),
            'eyes_closed_frames'    : br.eyes_closed_frames,

            'yawn_count'            : self._yawn_count,
            'yawn_markers'          : self._yawn_markers[-20:],   # last 20

            'head_down_count'       : head_summary['event_count'],
            'head_down_total_mins'  : head_summary['total_mins'],
            'head_down_total_secs'  : head_summary['total_secs'],
            'head_down_current_secs': head_summary['current_streak_sec'],
            'head_currently_down'   : head_summary['current_active'],
            'head_down_events'      : head_summary['events'][-10:],

            'sleep_count'           : sleep_summary['event_count'],
            'sleep_total_mins'      : sleep_summary['total_mins'],
            'sleep_total_secs'      : sleep_summary['total_secs'],
            'sleep_current_secs'    : sleep_summary['current_streak_sec'],
            'student_currently_sleeping' : sleep_summary['current_active'],
            'sleep_events'          : sleep_summary['events'][-10:],

            'phone_active'          : phone_summary['current_active'],
            'phone_events_count'    : phone_summary['total_events'],
            'phone_total_secs'      : phone_summary['total_secs'],
            'phone_current_secs'    : phone_summary['current_duration'],
            'phone_events'          : phone_summary['events'],

            'attention_ts'          : self._attention_ts,
        }

        # ------------------------------------------------------------------ #
        # 8. Publish to file (for dashboard polling)
        # ------------------------------------------------------------------ #
        if now - self._last_publish_time >= PUBLISH_INTERVAL_S:
            self._publish(state)
            self._last_publish_time = now

        return state

    # ----------------------------------------------------------------------- #
    def end_session(self):
        self._session_active = False
        # Force final publish
        state = self._build_final_report()
        self._publish(state)
        print("[StudentTracker] Session ended. Final report published.")
        return state

    # ----------------------------------------------------------------------- #
    def _build_final_report(self):
        """Build the final end-of-session report dict."""
        session_secs = (int(time.time() - self._session_start)
                        if self._session_start else 0)
        pf = self._present_frames or 1
        att_pct = round(self._attentive_frames / pf * 100, 1)

        head_summary  = self._head_down.get_summary()
        phone_summary = self._phone.get_summary()

        return {
            'final'                 : True,
            'ts'                    : time.strftime('%H:%M:%S'),
            'session_secs'          : session_secs,
            'student_name'          : self.student_name,

            'attention_pct'         : att_pct,
            'total_frames'          : self._total_frames,
            'present_frames'        : self._present_frames,
            'attentive_frames'      : self._attentive_frames,

            'yawn_count'            : self._yawn_count,
            'yawn_markers'          : self._yawn_markers,

            'head_down_count'       : head_summary['event_count'],
            'head_down_total_mins'  : head_summary['total_mins'],
            'head_down_events'      : head_summary['events'],

            'phone_events_count'    : phone_summary['total_events'],
            'phone_total_secs'      : phone_summary['total_secs'],
            'phone_events'          : phone_summary['events'],

            'attention_ts'          : self._attention_ts,
        }

    # ----------------------------------------------------------------------- #
    def _publish(self, state):
        """Write state dict to outputs/state.json for dashboard polling (atomic write)."""
        try:
            # Atomic write: write to temp file, then rename
            state_dir = os.path.dirname(STATE_FILE)
            _ensure_dir(state_dir)
            
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=state_dir, 
                delete=False,
                suffix='.json'
            ) as tmp:
                json.dump(state, tmp)
                tmp_path = tmp.name
            
            # Atomic rename
            shutil.move(tmp_path, STATE_FILE)
        except Exception as e:
            print("[StudentTracker] State publish failed: {}".format(e))
