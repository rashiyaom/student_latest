# =============================================================================
# single_student_main.py
# EvilEye — Protect your sessions from evil doing | Main Entry Point
#
# IMPORTANT: Run server.py in a separate terminal for dashboard to work!
#   Terminal 1: python3 single_student_main.py
#   Terminal 2: python3 server.py
#   Then open: http://localhost:5000 (or http://<jetson_ip>:5000 from another device)
#
# Full pipeline per frame:
#   1.  Capture webcam frame
#   2.  YOLOv8n TRT inference → ALL COCO detections (person + phone)
#   3.  Find the ONE student (person with highest confidence / largest box)
#   4.  Crop student ROI
#   5.  StudentTracker.update():
#         a. BehaviourAnalyzer  → yawn, head, eye, attention
#         b. HeadDownTracker    → cumulative head-down time
#         c. PhoneDetector      → phone events + screenshots
#   6.  Draw all overlays on frame
#   7.  Display live window
#   8.  Write state.json every second (server.py serves this to dashboard)
#
# Press 'q' to quit | 'r' to reset session | 's' to save screenshot
#
# Compatible with: Python 3.6+ / JetPack 4.6+ / Jetson Nano 2GB
# =============================================================================

import cv2
import time
import os
import numpy as np

from trt_yolo_multi   import TRTYoloDetectorMulti   # updated multi-class detector
from student_tracker  import StudentTracker
from single_utils     import draw_single_student_hud

# ============================================================================ #
#   USER-ADJUSTABLE SETTINGS
# ============================================================================ #

ENGINE_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "yolov8n.engine")
WEBCAM_INDEX   = 0
INPUT_SIZE     = 320          # must match engine build size
CONF_THRESH    = 0.35         # person confidence
IOU_THRESH     = 0.45         # NMS IoU
CAP_WIDTH      = 640
CAP_HEIGHT     = 480
STUDENT_NAME   = "Student"    # change to actual student name
LOG_INTERVAL   = 30           # terminal log every N frames

# ============================================================================ #

PERSON_CLASS = 0
PHONE_CLASS  = 67


def open_webcam(idx, w, h):
    """Open webcam with retry logic for Jetson camera initialization."""
    retries = 3
    initial_delay = 0.5
    
    for attempt in range(retries):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                return cap
            cap.release()
        except Exception as e:
            print("[Webcam] Attempt {}/{} failed: {}".format(attempt+1, retries, e))
        
        if attempt < retries - 1:
            delay = initial_delay * (2 ** attempt)
            print("[Webcam] Retrying in {:.1f}s...".format(delay))
            time.sleep(delay)
    
    raise RuntimeError("Cannot open webcam {} after {} attempts".format(idx, retries))


def best_person_box(detections):
    """
    From a list of [x1,y1,x2,y2,conf,class_id] detections,
    return the single best person detection as [x1,y1,x2,y2,conf] or None.
    "Best" = highest confidence person detection.
    """
    persons = [d for d in detections
               if len(d) >= 6 and int(d[5]) == PERSON_CLASS and d[4] >= CONF_THRESH]
    if not persons:
        return None
    # Sort by confidence descending; take the first
    persons.sort(key=lambda d: d[4], reverse=True)
    return persons[0][:5]   # [x1,y1,x2,y2,conf]


def crop_roi(frame, x1, y1, x2, y2):
    fh, fw = frame.shape[:2]
    x1c = max(0, int(x1)); y1c = max(0, int(y1))
    x2c = min(fw, int(x2)); y2c = min(fh, int(y2))
    if x2c - x1c < 10 or y2c - y1c < 10:
        return None
    return frame[y1c:y2c, x1c:x2c]


def run():
    print("[SingleStudent] Loading TRT engine: {}".format(ENGINE_PATH))
    
    # Validate engine file exists
    if not os.path.isfile(ENGINE_PATH):
        print("[ERROR] TensorRT engine not found: {}".format(ENGINE_PATH))
        print("[ERROR] Please ensure yolov8n.engine is in the project directory.")
        print("[ERROR] Build with: python3 -m tensorrt.build_engine --model yolov8n.pt")
        return
    
    # Create outputs directory for screenshots and state file
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'phone_screenshots'), exist_ok=True)
    
    detector = TRTYoloDetectorMulti(ENGINE_PATH, input_size=INPUT_SIZE)

    tracker  = StudentTracker(student_name=STUDENT_NAME)
    tracker.start_session()

    print("[SingleStudent] Opening webcam {}...".format(WEBCAM_INDEX))
    cap = open_webcam(WEBCAM_INDEX, CAP_WIDTH, CAP_HEIGHT)
    print("[SingleStudent] Running. Press 'q'=quit, 'r'=reset, 's'=screenshot")

    fps = 0.0
    frame_count = 0
    t_fps = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[SingleStudent] Frame read failed — retrying.")
            continue

        frame_count += 1

        # ------------------------------------------------------------------ #
        # 1. Multi-class YOLO inference (person + phone)
        # ------------------------------------------------------------------ #
        all_dets = detector.detect_all(frame,
                                       conf_thresh=CONF_THRESH,
                                       iou_thresh=IOU_THRESH)
        # all_dets: list of [x1,y1,x2,y2,conf,class_id]

        # ------------------------------------------------------------------ #
        # 2. Find single student
        # ------------------------------------------------------------------ #
        person_det = best_person_box(all_dets)
        person_roi = None
        person_box = None

        if person_det is not None:
            x1,y1,x2,y2,conf = person_det
            person_box = [int(x1),int(y1),int(x2),int(y2)]
            person_roi = crop_roi(frame, x1, y1, x2, y2)

        # ------------------------------------------------------------------ #
        # 3. StudentTracker update
        # ------------------------------------------------------------------ #
        state = tracker.update(
            person_roi_bgr  = person_roi,
            person_box      = person_box,
            all_detections  = all_dets,
            frame_bgr       = frame
        )

        # ------------------------------------------------------------------ #
        # 4. Draw HUD
        # ------------------------------------------------------------------ #
        draw_single_student_hud(frame, state, person_box, all_dets, fps)

        # ------------------------------------------------------------------ #
        # 5. Display
        # ------------------------------------------------------------------ #
        cv2.imshow("EvilEye — Protect your sessions from evil doing", frame)

        # ------------------------------------------------------------------ #
        # 6. Terminal log (enhanced with yawn diagnostics)
        # ------------------------------------------------------------------ #
        if frame_count % LOG_INTERVAL == 0:
            yawn_state_indicator = "🥱" if state.get('yawn_state') == 'YAWNING' else "  "
            sleep_indicator = "😴" if state.get('sleep_state') == 'SLEEPING' else "  "
            print(
                "[CP] #{:5d} | {} | ATT:{:.0f}% | EYE:{} | HEAD:{} | SLP:{} | "
                "YAWNS:{} {} | PHONE:{} | FPS:{:.1f}".format(
                    frame_count,
                    state.get('attention_label','?')[:3],
                    state.get('attention_pct', 0),
                    state.get('eye_state','?')[:3],
                    state.get('head_state','?')[:3],
                    state.get('sleep_state','?')[:3],
                    state.get('yawn_count', 0),
                    yawn_state_indicator,
                    'ON' if state.get('phone_active') else 'off',
                    fps
                )
            )

        # ------------------------------------------------------------------ #
        # 7. FPS
        # ------------------------------------------------------------------ #
        if frame_count % 30 == 0:
            elapsed = time.time() - t_fps
            fps = 30.0 / elapsed if elapsed > 0 else 0.0
            t_fps = time.time()

        # ------------------------------------------------------------------ #
        # 8. Keys
        # ------------------------------------------------------------------ #
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[SingleStudent] Quit.")
            break
        elif key == ord('r'):
            tracker.end_session()
            tracker.start_session()
            print("[SingleStudent] Session reset.")
        elif key == ord('s'):
            ts  = time.strftime("%Y%m%d_%H%M%S")
            fn  = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'outputs', 'snapshot_{}.jpg'.format(ts))
            cv2.imwrite(fn, frame)
            print("[SingleStudent] Snapshot saved: {}".format(fn))

    # Cleanup
    tracker.end_session()
    cap.release()
    cv2.destroyAllWindows()
    print("[SingleStudent] Done.")


if __name__ == "__main__":
    run()
