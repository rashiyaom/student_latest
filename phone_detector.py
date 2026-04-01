# =============================================================================
# phone_detector.py
# EvilEye — Protect your sessions from evil doing | Phone Usage Detector + Screenshot Capture
#
# DESIGN: Re-uses the SAME YOLOv8n TensorRT inference already running.
# -----------------------------------------------------------------------
# YOLOv8n is trained on COCO-80.  COCO class 67 = "cell phone".
# The TRTYoloDetector is extended to return ALL detections (not just person).
# PhoneDetector receives raw detections, filters for class 67, and:
#
#   1. Confirms the phone is near the student's person box (overlap check)
#   2. Records:
#       - phone_detected   : bool  (this frame)
#       - phone_events     : list of {start_time, end_time, duration_sec,
#                                     screenshot_b64, thumbnail_b64}
#       - total_phone_secs : int   cumulative seconds of phone usage
#       - current_streak   : int   current continuous phone seconds
#   3. Takes a screenshot (saves to outputs/ folder + stores as JPEG bytes)
#      the FIRST frame a new phone event starts
#   4. Sustained trigger: phone must be detected for MIN_PHONE_FRAMES
#      consecutive frames before an event is registered (avoids false positives
#      from brief reflective surfaces, book covers, etc.)
#   5. Phone event ends after MAX_MISS_FRAMES consecutive frames without phone
#
# COCO class IDs referenced:
#   0  → person
#   67 → cell phone
#
# Compatible with: Python 3.6 / OpenCV / NumPy / no extra installs
# =============================================================================

import cv2
import os
import time
import base64
import numpy as np
from typing import Optional

# ============================================================================ #
#   CONFIGURABLE
# ============================================================================ #

PHONE_CLASS_ID      = 67     # COCO class index for 'cell phone'
PHONE_CONF_THRESH   = 0.35   # confidence threshold specific to phone class
MIN_PHONE_FRAMES    = 6      # consecutive frames before event is confirmed
MAX_MISS_FRAMES     = 10     # consecutive missed frames to end a phone event

SCREENSHOT_DIR      = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'outputs', 'phone_screenshots')

SCREENSHOT_QUALITY  = 70     # JPEG quality 0-100
THUMBNAIL_SIZE      = (80, 60)   # thumbnail stored in JSON payload

# ============================================================================ #

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _iou(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter + 1e-6
    return inter / union


def _box_center_in(cx, cy, box):
    """True if point (cx,cy) is inside box [x1,y1,x2,y2]."""
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


class PhoneEvent(object):
    """One confirmed phone-usage episode."""
    __slots__ = ['start_time', 'end_time', 'duration_sec',
                 'screenshot_path', 'screenshot_b64', 'thumbnail_b64',
                 'frame_count']

    def __init__(self, start_time):
        self.start_time      = start_time      # float unix timestamp
        self.end_time        = None            # float unix timestamp or None
        self.duration_sec    = 0
        self.screenshot_path = None
        self.screenshot_b64  = None
        self.thumbnail_b64   = None
        self.frame_count     = 0              # frames phone was visible this event


class PhoneDetector(object):
    """
    Receives all YOLO detections per frame, tracks phone usage for ONE student.

    Usage:
        detector = PhoneDetector()

        # Every frame — pass all raw detections AND the current frame:
        phone_now, event = detector.update(
            all_detections,   # list of [x1,y1,x2,y2,conf,class_id]
            student_box,      # [x1,y1,x2,y2] of the person (or None)
            frame             # BGR numpy array (for screenshot)
        )
        # phone_now : bool — phone visible this frame
        # event     : PhoneEvent or None — newly confirmed event (for logging)
    """

    def __init__(self):
        _ensure_dir(SCREENSHOT_DIR)

        self._detect_count  = 0    # consecutive frames with phone detected
        self._miss_count    = 0    # consecutive frames WITHOUT phone

        self._event_active  = False
        self._current_event = None   # type: Optional[PhoneEvent]

        self.events             = []     # list of completed PhoneEvent
        self.total_phone_secs   = 0.0
        self.phone_detected     = False  # current frame
        self._last_phone_time   = None

        print("[PhoneDetector] Initialised. Screenshots → {}".format(SCREENSHOT_DIR))

    # ----------------------------------------------------------------------- #
    def update(self, all_detections, student_box, frame):
        """
        all_detections : list of [x1,y1,x2,y2,conf,class_id]
                         (requires trt_yolo to return class_id too — see note)
        student_box    : [x1,y1,x2,y2] or None
        frame          : BGR numpy array

        Returns (phone_visible_this_frame: bool, new_event: PhoneEvent or None)
        """
        now = time.time()
        new_event = None

        # ------------------------------------------------------------------ #
        # 1. Filter for phone detections near the student
        # ------------------------------------------------------------------ #
        phone_dets = [
            d for d in all_detections
            if len(d) >= 6 and int(d[5]) == PHONE_CLASS_ID and d[4] >= PHONE_CONF_THRESH
        ]

        phone_near = False
        for det in phone_dets:
            px1, py1, px2, py2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])

            if student_box is not None:
                # Phone must overlap or be inside/near the student bounding box
                # We expand the student box slightly (30px padding) to be generous
                sb = [student_box[0]-30, student_box[1]-30,
                      student_box[2]+30, student_box[3]+30]
                phone_cx = (px1 + px2) // 2
                phone_cy = (py1 + py2) // 2
                if _box_center_in(phone_cx, phone_cy, sb) or _iou([px1,py1,px2,py2], sb) > 0.05:
                    phone_near = True
                    break
            else:
                # No person box → accept any phone detection
                phone_near = True
                break

        # ------------------------------------------------------------------ #
        # 2. Sustained-trigger state machine
        # ------------------------------------------------------------------ #
        if phone_near:
            self._detect_count += 1
            self._miss_count    = 0
        else:
            self._miss_count   += 1
            self._detect_count  = 0

        phone_confirmed = (self._detect_count >= MIN_PHONE_FRAMES)
        event_ended     = (self._event_active and self._miss_count >= MAX_MISS_FRAMES)

        # ------------------------------------------------------------------ #
        # 3. Event lifecycle
        # ------------------------------------------------------------------ #
        if phone_confirmed and not self._event_active:
            # — NEW EVENT START —
            self._event_active  = True
            ev = PhoneEvent(now - MIN_PHONE_FRAMES / 12.0)  # backdate slightly
            self._current_event = ev
            # Take screenshot
            self._capture_screenshot(frame, phone_dets[0] if phone_dets else None, ev)
            print("[PhoneDetector] Phone event STARTED at {:.1f}".format(ev.start_time))

        if self._event_active and self._current_event is not None:
            self._current_event.frame_count += 1
            # Update running duration every frame
            self._current_event.duration_sec = now - self._current_event.start_time

        if event_ended and self._current_event is not None:
            # — EVENT END —
            ev = self._current_event
            ev.end_time      = now - MAX_MISS_FRAMES / 12.0  # backdate
            ev.duration_sec  = ev.end_time - ev.start_time
            self.total_phone_secs += max(0, ev.duration_sec)
            self.events.append(ev)
            new_event = ev
            self._event_active  = False
            self._current_event = None
            print("[PhoneDetector] Phone event ENDED. Duration: {:.1f}s".format(
                ev.duration_sec))

        self.phone_detected = self._event_active

        return self.phone_detected, new_event

    # ----------------------------------------------------------------------- #
    def _capture_screenshot(self, frame, phone_det, event):
        """Save a full-frame screenshot and store both full and thumbnail as base64."""
        if frame is None:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = "phone_{}.jpg".format(ts)
        fpath = os.path.join(SCREENSHOT_DIR, fname)

        # Annotate frame copy
        annotated = frame.copy()
        if phone_det is not None:
            x1,y1,x2,y2 = int(phone_det[0]),int(phone_det[1]),int(phone_det[2]),int(phone_det[3])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(annotated, 'PHONE DETECTED', (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(annotated, time.strftime("%H:%M:%S"), (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,128), 2, cv2.LINE_AA)

        try:
            cv2.imwrite(fpath, annotated, [cv2.IMWRITE_JPEG_QUALITY, SCREENSHOT_QUALITY])
            event.screenshot_path = fpath
        except Exception as e:
            print("[PhoneDetector] Screenshot save failed: {}".format(e))

        # Full-size screenshot as base64 for dashboard viewer
        try:
            _, buf_full = cv2.imencode('.jpg', annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, SCREENSHOT_QUALITY])
            event.screenshot_b64 = base64.b64encode(buf_full.tobytes()).decode('utf-8')
        except Exception as e:
            print("[PhoneDetector] Full screenshot encode failed: {}".format(e))
            event.screenshot_b64 = None

        # Thumbnail as base64 for dashboard JSON (smaller for quick loading)
        try:
            thumb = cv2.resize(annotated, THUMBNAIL_SIZE, interpolation=cv2.INTER_LINEAR)
            _, buf_thumb = cv2.imencode('.jpg', thumb,
                                        [cv2.IMWRITE_JPEG_QUALITY, 25])  # Reduced quality for smaller size
            event.thumbnail_b64 = base64.b64encode(buf_thumb.tobytes()).decode('utf-8')
        except Exception as e:
            print("[PhoneDetector] Thumbnail encode failed: {}".format(e))
            event.thumbnail_b64 = None  # Don't store if encoding fails

    # ----------------------------------------------------------------------- #
    def get_current_duration(self):
        """Return duration in seconds of the currently active phone event, or 0."""
        if self._event_active and self._current_event is not None:
            return time.time() - self._current_event.start_time
        return 0.0

    # ----------------------------------------------------------------------- #
    def get_summary(self):
        """Return a summary dict for the dashboard."""
        return {
            'total_events'      : len(self.events),
            'total_secs'        : round(self.total_phone_secs, 1),
            'current_active'    : self._event_active,
            'current_duration'  : round(self.get_current_duration(), 1),
            'events'            : [
                {
                    'start': time.strftime('%H:%M:%S',
                             time.localtime(ev.start_time)),
                    'duration_sec': round(ev.duration_sec, 1),
                    'screenshot_path': ev.screenshot_path,
                    'thumbnail_b64': ev.thumbnail_b64,
                    'screenshot_b64': ev.screenshot_b64,  # Full screenshot for viewer
                }
                for ev in self.events
            ],
        }

    # ----------------------------------------------------------------------- #
    def reset(self):
        """Reset all state for a new session."""
        self._detect_count  = 0
        self._miss_count    = 0
        self._event_active  = False
        self._current_event = None
        self.events         = []
        self.total_phone_secs = 0.0
        self.phone_detected = False
