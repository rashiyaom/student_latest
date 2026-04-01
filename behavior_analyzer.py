# =============================================================================
# behavior_analyzer.py
# EvilEye — Protect your sessions from evil doing | Real-Time Participant Behaviour Analyzer
#
# DESIGN: No MediaPipe. No dlib. No PyTorch. No ONNX Runtime.
# ---------------------------------------------------------------
# Everything runs on:
#   - OpenCV (already on Jetson)
#   - NumPy  (already on Jetson)
#   - The SAME TensorRT YOLOv8n engine already loaded for person detection
#     (re-used to detect 'cell phone' — COCO class 67)
#
# What this module detects from a FACE ROI:
#   1. Yawn detection  — MAR via OpenCV face + mouth detection cascade
#   2. Head-down       — Face region position in frame (y-centroid ratio)
#   3. Eye state       — EAR via eye sub-cascade on face ROI
#   4. Engagement score — Composite of all signals
#
# HOW FACE/EYE/MOUTH IS FOUND (without MediaPipe / dlib):
#   OpenCV ships Haar cascade XML files that detect faces, eyes, and
#   mouth regions.  On Jetson JetPack 4.6, OpenCV is built with these
#   cascades at:
#       /usr/share/opencv4/haarcascades/
#   or the Python package ships them via cv2.data.haarcascades.
#   We use three cascades:
#       haarcascade_frontalface_default.xml
#       haarcascade_eye.xml
#       haarcascade_mcs_mouth.xml
#   All are CPU-only, extremely lightweight, and zero extra installs.
#
# YAWN DETECTION STRATEGY (smart, not naive MAR):
#   -----------------------------------------------
#   Naive MAR from 2 pixel rows is unreliable on low-res crops.
#   Instead we use a FOUR-STAGE confirmation system:
#
#   Stage 1: Mouth region detected by cascade → compute vertical fill ratio
#            (how much of the bounding box is "open dark area")
#   Stage 2: Darkness ratio — inside the mouth box, count dark pixels
#            (open mouth = darker interior than closed)
#   Stage 3: Aspect ratio check — open mouth is taller relative to width
#   Stage 4: Temporal smoothing — must persist for MIN_YAWN_FRAMES consecutive
#            frames before being counted as ONE yawn event
#            (prevents double-counting a single yawn)
#   Stage 5: Cooldown — after a yawn is confirmed, ignore next YAWN_COOLDOWN_FRAMES
#            frames to avoid re-triggering on the same event
#
# HEAD-DOWN STRATEGY:
#   -----------------------------------------------
#   If a face IS detected in the frame:
#       - Measure how low the face centroid is in the person ROI
#         (y_center / roi_height). If > HEAD_FACE_Y_THRESH → looking down.
#       - Additionally: if the face box is much wider than tall → tilted down.
#   If NO face is detected in the person ROI:
#       - This is a strong signal for head-down (face not visible to camera).
#       - We count this as HEAD_DOWN after NO_FACE_THRESH frames.
#
# EAR STRATEGY (without landmarks):
#   -----------------------------------------------
#   On the eye sub-region detected by the eye cascade:
#       - Convert to grayscale, threshold to binary
#       - Compute horizontal extent of dark region (pupil/iris shadow)
#       - Compare dark region height vs total eye box height
#       - Low ratio → partially closed → drowsy
#       - This is a proxy EAR, not a true geometric EAR, but reliable enough
#         for drowsy/awake distinction on a Jetson crop
#
# Compatible with: Python 3.6 / OpenCV 4.x / NumPy / Jetson Nano 2GB
# =============================================================================

import cv2
import numpy as np
import os
import time
from collections import deque

# ============================================================================ #
#   CONFIGURABLE THRESHOLDS
# ============================================================================ #

# --- Cascade paths ---
# These paths work on Jetson JetPack 4.6 / OpenCV installed via apt or pip.
# NOTE: Some OpenCV builds (esp. on embedded systems) don't expose cv2.data
def _find_cascade(name):
    """Find a Haar cascade XML by name. Returns full path or None."""
    candidates = []
    
    # Try cv2.data.haarcascades if available (some builds expose it)
    try:
        cv2_data = getattr(cv2, 'data', None)
        if cv2_data is not None:
            haar_dir = getattr(cv2_data, 'haarcascades', None)
            if haar_dir:
                candidates.append(os.path.join(haar_dir, name))
    except (AttributeError, Exception):
        pass
    
    # Common system install paths
    candidates += [
        '/usr/share/opencv4/haarcascades/' + name,
        '/usr/share/opencv/haarcascades/'  + name,
        '/usr/local/share/opencv4/haarcascades/' + name,
        '/usr/local/share/opencv/haarcascades/'  + name,
    ]
    
    # Project-local fallbacks (if cascades are vendored with the repo)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, name),
        os.path.join(here, 'haarcascades', name),
        os.path.join(here, 'cascades', name),
    ]
    
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

FACE_CASCADE_NAME  = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_NAME   = 'haarcascade_eye.xml'
MOUTH_CASCADE_NAME = 'haarcascade_mcs_mouth.xml'

# --- ROI sizing ---
ROI_W = 160   # face analysis ROI width  (person crop resized to this)
ROI_H = 200   # face analysis ROI height

# --- Yawn thresholds (QUICK DETECTION - fast response to yawn motion) ---
MOUTH_OPEN_RATIO       = 0.30   # mouth box height/width ratio → open if above (more sensitive for quick detection)
MOUTH_DARK_RATIO       = 0.20   # fraction of dark pixels inside mouth → open (lower for faster detection)
MIN_YAWN_FRAMES        = 12     # consecutive frames mouth must be open = 1 yawn (12 frames @ 12fps = 1 second FAST)
YAWN_COOLDOWN_FRAMES   = 60     # frames to ignore after a yawn is confirmed (~5s, shorter cooldown for quick re-detection)
YAWN_MIN_SIZE          = 12     # minimum mouth width/height in pixels (lower for quicker sensitivity)
MOUTH_THRESHOLD_VALUE  = 50     # manual threshold for mouth darkness (lower = catches more open mouths quickly)

# --- Head-down thresholds ---
HEAD_FACE_Y_THRESH     = 0.55   # face centroid y / roi_height > this → head down
NO_FACE_THRESH         = 10     # consecutive frames with no face → head down

# --- Eye / drowsy thresholds (OPTIMIZED for glasses-wearing users) ---
EYE_OPEN_RATIO         = 0.35   # dark pixel ratio in eye → above = open (increased for glasses reflections)
EYE_CLOSED_RATIO       = 0.15   # dark pixel ratio → below = closed (increased to account for glasses frame reflections)
EYE_DARK_THRESHOLD     = 85     # absolute pixel value threshold for detecting iris/pupil (helps with glasses)
EAR_SMOOTH_FRAMES      = 12     # temporal smoother for eye state (more stable for glasses)
SLEEP_CONFIRM_FRAMES   = 120    # ~10s @ 12 FPS: eyes closed for this many frames → sleeping

# --- Sleep detection with motion analysis (to distinguish sleep from note-taking) ---
MOTION_THRESHOLD       = 2.5    # pixels/frame movement threshold for stillness
MOTION_HISTORY_FRAMES  = 8      # smoothing window for motion (memory efficient)
SLEEP_MOTION_RATIO     = 0.3    # motion must be <30% of threshold for sleep confirmation

# --- Attention score weights ---
W_EYE     = 0.35
W_HEAD    = 0.30
W_YAWN    = 0.20
W_PRESENT = 0.15

# ============================================================================ #
#   RESULT CONTAINER
# ============================================================================ #

class BehaviourResult(object):
    """
    All behaviour signals for the single tracked participant, one frame.

    head_state      : "UP" | "DOWN" | "UNKNOWN"
    eye_state       : "OPEN" | "DROWSY" | "CLOSED" | "UNKNOWN"
    sleep_state     : "SLEEPING" | "AWAKE" | "UNKNOWN"  (eyes closed > 8s or head_down + eyes_closed)
    yawn_state      : "YAWNING" | "NO_YAWN" | "UNKNOWN"
    engagement      : "ENGAGED" | "DISTRACTED" | "DROWSY" | "SLEEPING" | "UNKNOWN"
    engagement_score : float 0.0–1.0
    face_found      : bool
    mouth_open_raw  : float  — raw mouth openness score (0–1)
    eye_open_raw    : float  — proxy eye openness (0–1, higher=more open)
    eyes_closed_frames : int  — consecutive frames eyes have been closed
    """
    __slots__ = [
        'head_state', 'eye_state', 'sleep_state', 'yawn_state',
        'engagement', 'engagement_score',
        'face_found', 'mouth_open_raw', 'eye_open_raw',
        'eyes_closed_frames', 'yawn_duration_sec',
    ]

    def __init__(self):
        self.head_state      = "UNKNOWN"
        self.eye_state       = "UNKNOWN"
        self.sleep_state     = "UNKNOWN"
        self.yawn_state      = "NO_YAWN"
        self.engagement      = "UNKNOWN"
        self.engagement_score = 0.0
        self.face_found      = False
        self.mouth_open_raw  = 0.0
        self.eye_open_raw    = 0.5
        self.eyes_closed_frames = 0
        self.yawn_duration_sec = 0.0

    def __repr__(self):
        return (
            "BehaviourResult(face={}, head={}, eye={}, yawn={}, "
            "engagement={} [{:.2f}])"
        ).format(self.face_found, self.head_state, self.eye_state,
                 self.yawn_state, self.engagement, self.engagement_score)


# ============================================================================ #
#   SCALAR SMOOTHER
# ============================================================================ #

class _Smoother(object):
    def __init__(self, n):
        self._buf = deque(maxlen=n)
    def push(self, v):
        self._buf.append(v)
    def mean(self):
        return float(sum(self._buf)) / len(self._buf) if self._buf else 0.5
    def reset(self):
        self._buf.clear()


# ============================================================================ #
#   MAIN ANALYSER
# ============================================================================ #

class BehaviourAnalyzer(object):
    """
    Single-participant behaviour analyzer using OpenCV cascades only.

    Usage:
        analyzer = BehaviourAnalyzer()
        result   = analyzer.analyse(person_roi_bgr, person_present=True)
    """

    def __init__(self):
        # Load cascades
        self._face_cas  = self._load_cascade(FACE_CASCADE_NAME,  'face')
        self._eye_cas   = self._load_cascade(EYE_CASCADE_NAME,   'eye')
        self._mouth_cas = self._load_cascade(MOUTH_CASCADE_NAME, 'mouth')

        # --- Yawn state machine ---
        self._mouth_open_count  = 0    # consecutive frames mouth is open
        self._yawn_cooldown     = 0    # frames remaining in cooldown (prevents duplicate yawns)
        self._yawn_in_progress  = False
        self._yawn_start_time   = None # timestamp when yawn started (for duration tracking)

        # --- Head-down state (with hysteresis to prevent glitching) ---
        self._no_face_count = 0
        self._head_down_count = 0      # consecutive frames head is down
        self._head_down_start_time = None  # timestamp when head went down
        self._head_is_down = False     # stable state (prevents rapid flipping)

        # --- Sleep tracking (eyes closed > SLEEP_CONFIRM_FRAMES) ---
        self._eyes_closed_frames = 0
        
        # --- Motion-based sleep detection (to distinguish from note-taking) ---
        self._prev_face_center = None  # Previous face center (x, y)
        self._face_motion_buffer = deque(maxlen=MOTION_HISTORY_FRAMES)  # Lightweight motion history

        # --- Temporal smoothers ---
        self._eye_smoother  = _Smoother(EAR_SMOOTH_FRAMES)
        self._mouth_smoother= _Smoother(4)

        print("[BehaviourAnalyzer] Initialised (OpenCV cascades only).")

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _load_cascade(name, label):
        path = _find_cascade(name)
        if path is None:
            if label == 'face':
                print("[ERROR] CRITICAL: Face cascade not found!")
                print("[ERROR] Please install OpenCV cascades:")
                print("[ERROR]   sudo apt-get install libopencv-dev")
                print("[ERROR] Or copy cascades to: /usr/share/opencv4/haarcascades/")
                raise RuntimeError("Face cascade required for system to function")
            else:
                print("[BehaviourAnalyzer] WARNING: {} cascade not found ({})."
                      " {} detection disabled.".format(label, name, label.capitalize()))
            return None
        try:
            cas = cv2.CascadeClassifier(path)
            if cas.empty():
                print("[BehaviourAnalyzer] WARNING: {} cascade failed to load from: {}".format(label, path))
                return None
            print("[BehaviourAnalyzer] Loaded {} cascade: {}".format(label, path))
            return cas
        except Exception as e:
            print("[BehaviourAnalyzer] ERROR loading {} cascade: {}".format(label, e))
            return None

    # ----------------------------------------------------------------------- #
    def _calculate_face_motion(self, face_box):
        """
        Calculate pixel displacement of face center (lightweight for Jetson).
        Uses Manhattan distance (faster than Euclidean on embedded systems).
        """
        if face_box is None:
            self._prev_face_center = None
            return 0.0
        
        fx, fy, fw, fh = face_box
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        
        if self._prev_face_center is None:
            self._prev_face_center = (cx, cy)
            return 0.0
        
        # Manhattan distance (no sqrt = faster)
        dx = abs(cx - self._prev_face_center[0])
        dy = abs(cy - self._prev_face_center[1])
        displacement = dx + dy
        
        self._prev_face_center = (cx, cy)
        self._face_motion_buffer.append(displacement)
        
        return displacement

    # ----------------------------------------------------------------------- #
    def _get_avg_face_motion(self):
        """Get averaged face motion (lightweight deque operation)."""
        if not self._face_motion_buffer:
            return 0.0
        return float(sum(self._face_motion_buffer)) / len(self._face_motion_buffer)

    # ----------------------------------------------------------------------- #
    def analyse(self, person_roi_bgr, person_present=True):
        """
        Analyse a single frame's person crop.

        person_roi_bgr : BGR numpy array (the YOLO person bounding box crop)
                         Can be any size — will be resized internally.
        person_present : bool — if False, return absent result immediately.

        Returns BehaviourResult.
        """
        result = BehaviourResult()

        if not person_present or person_roi_bgr is None or person_roi_bgr.size == 0:
            self._no_face_count += 1
            return result

        # ------------------------------------------------------------------ #
        # 1. Resize to standard ROI
        # ------------------------------------------------------------------ #
        h0, w0 = person_roi_bgr.shape[:2]
        if h0 < 8 or w0 < 8:
            return result

        roi = cv2.resize(person_roi_bgr, (ROI_W, ROI_H),
                         interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Histogram equalise → better cascade performance under varying light
        gray = cv2.equalizeHist(gray)

        # ------------------------------------------------------------------ #
        # 2. Face detection in ROI
        # ------------------------------------------------------------------ #
        face_found = False
        face_box   = None

        if self._face_cas is not None:
            faces = self._face_cas.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                # Pick the largest face
                faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
                face_box   = faces[0]   # (x, y, w, h)
                face_found = True

        result.face_found = face_found

        # ------------------------------------------------------------------ #
        # 3. Head-down detection (IMPROVED with hysteresis to prevent glitching)
        # ================================================================== #
        # Strategy: Use TWO thresholds to create hysteresis (prevents flickering)
        #   - HEAD_DOWN_THRESHOLD = 0.60 to START marking head as down
        #   - HEAD_UP_THRESHOLD = 0.50 to STOP marking head as down (hysteresis zone)
        #   - Require MIN_HEAD_DOWN_FRAMES (12 frames) to confirm state change
        # This prevents rapid flipping between UP/DOWN states
        # ================================================================== #
        
        if face_found:
            self._no_face_count = 0
            fx, fy, fw, fh = face_box
            
            # y centroid of face relative to ROI height
            face_cy_ratio = (fy + fh * 0.5) / ROI_H
            
            # Aspect ratio (width/height) - wide faces indicate head-down angle
            aspect_ratio = fw / float(fh) if fh > 0 else 1.0
            
            # STRICT CHECK: Both conditions should align for reliable head-down detection
            # Head down if: face centroid is LOW (face is in lower part of ROI)
            #           OR: face is unusually WIDE (head tilted down away from camera)
            is_head_down_signal = (face_cy_ratio > 0.60 or aspect_ratio > 1.50)
            
            # Accumulate / decay counter for hysteresis
            if is_head_down_signal:
                self._head_down_count += 1
            else:
                # Slow decay (prevents jitter)
                self._head_down_count = max(0, self._head_down_count - 1)
            
            # Use hysteresis: threshold to enter vs threshold to exit head-down state
            MIN_HEAD_DOWN_FRAMES = 8  # ~0.67 seconds @ 12fps
            
            if not self._head_is_down and self._head_down_count >= MIN_HEAD_DOWN_FRAMES:
                # TRANSITION: UP → DOWN
                self._head_is_down = True
                self._head_down_start_time = time.time()
                result.head_state = "DOWN"
            elif self._head_is_down and self._head_down_count <= 2:
                # TRANSITION: DOWN → UP (requires low count to exit)
                self._head_is_down = False
                self._head_down_start_time = None
                result.head_state = "UP"
            else:
                # STABLE STATE
                result.head_state = "DOWN" if self._head_is_down else "UP"
        
        else:
            # No face detected
            self._no_face_count += 1
            self._head_down_count = 0  # reset counter
            
            # If face disappears during active yawn, force end the yawn
            if self._yawn_in_progress:
                result.yawn_state = "NO_YAWN"
                self._yawn_in_progress = False
                self._yawn_start_time = None
                self._mouth_open_count = 0
                self._yawn_cooldown = YAWN_COOLDOWN_FRAMES
            
            # If no face for 10+ frames, assume head is down (face turned away)
            if self._no_face_count >= NO_FACE_THRESH:
                if not self._head_is_down:
                    self._head_is_down = True
                    self._head_down_start_time = time.time()
                result.head_state = "DOWN"
            else:
                result.head_state = "UNKNOWN"

        # ------------------------------------------------------------------ #
        # 4. Eye state (proxy EAR via eye sub-cascade) — IMPROVED for glasses
        # ------------------------------------------------------------------ #
        eye_open_score = 0.5   # default neutral

        if face_found and self._eye_cas is not None:
            fx, fy, fw, fh = face_box
            # Look for eyes in TOP HALF of the face only
            face_gray = gray[fy: fy + fh//2, fx: fx + fw]
            if face_gray.size > 0:
                # Apply CLAHE to eye region for better contrast (especially with glasses glare)
                try:
                    clahe_eye = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
                    face_gray_enhanced = clahe_eye.apply(face_gray)
                except Exception:
                    face_gray_enhanced = face_gray
                
                eyes = self._eye_cas.detectMultiScale(
                    face_gray_enhanced,
                    scaleFactor=1.05,  # More aggressive search for glasses
                    minNeighbors=4,     # Lower threshold for glasses-occluded eyes
                    minSize=(6, 6)      # Detect smaller regions (glasses reduce visible area)
                )
                if len(eyes) > 0:
                    # Compute proxy openness: dark pixel ratio inside each eye box
                    # Optimized for glasses: detects iris/pupil through lens
                    scores = []
                    for (ex, ey, ew, eh) in eyes[:2]:
                        eye_patch = face_gray_enhanced[ey:ey+eh, ex:ex+ew]
                        if eye_patch.size == 0:
                            continue
                        
                        # Multi-threshold approach for glasses robustness
                        # 1. Threshold at iris/pupil darkness level (50)
                        _, thresh1 = cv2.threshold(
                            eye_patch, 50, 255, cv2.THRESH_BINARY_INV)
                        
                        # 2. Absolute dark pixel detection (iris typically < 85 gray value)
                        _, thresh2 = cv2.threshold(
                            eye_patch, EYE_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                        
                        # 3. Adaptive local threshold (handles glasses glare & shadows)
                        if eh > 5 and ew > 5:
                            try:
                                thresh3 = cv2.adaptiveThreshold(
                                    eye_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 3, 2)
                            except:
                                thresh3 = np.zeros_like(eye_patch)
                        else:
                            thresh3 = np.zeros_like(eye_patch)
                        
                        # Combine thresholds (OR operation) - more lenient for glasses detection
                        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
                        combined_thresh = cv2.bitwise_or(combined_thresh, thresh3)
                        
                        # Count dark pixels (iris/pupil)
                        dark_pixels = np.count_nonzero(combined_thresh)
                        total_pixels = eye_patch.size
                        dark_frac = float(dark_pixels) / total_pixels if total_pixels > 0 else 0.0
                        
                        # Clamp to valid range [0, 1]
                        dark_frac = max(0.0, min(1.0, dark_frac))
                        scores.append(dark_frac)
                    
                    if scores:
                        eye_open_score = float(np.mean(scores))
                    else:
                        eye_open_score = 0.5  # if only detection artifacts
                else:
                    # Eyes not detected by cascade - adaptive fallback for glasses
                    # Analyze larger region for dark iris/pupil areas that cascade missed
                    face_region_larger = gray[fy: fy + fh//2, fx: fx + fw]
                    
                    try:
                        clahe_fallback = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                        face_region_enh = clahe_fallback.apply(face_region_larger)
                    except:
                        face_region_enh = face_region_larger
                    
                    # Look for dark regions that represent iris (visible through glasses)
                    _, dark_mask = cv2.threshold(face_region_enh, EYE_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                    dark_ratio = float(np.count_nonzero(dark_mask)) / face_region_larger.size
                    
                    # Conservative estimate: if significant dark regions present → eyes partially open
                    # Use a lower threshold since we're looking at larger region
                    if dark_ratio > 0.08:
                        eye_open_score = 0.25 + (dark_ratio * 0.5)  # scale to 0.25-0.75 range
                    else:
                        eye_open_score = 0.10  # Very low when few dark regions (eyes likely closed)

        self._eye_smoother.push(eye_open_score)
        smooth_eye = self._eye_smoother.mean()
        result.eye_open_raw = smooth_eye

        # Three-state eye detection with hysteresis: OPEN, DROWSY (partial), CLOSED
        if smooth_eye >= EYE_OPEN_RATIO:
            result.eye_state = "OPEN"
            self._eyes_closed_frames = 0  # reset sleep counter
        elif smooth_eye >= EYE_CLOSED_RATIO:
            result.eye_state = "DROWSY"
            self._eyes_closed_frames += 1
        else:
            result.eye_state = "CLOSED"
            self._eyes_closed_frames += 1

        result.eyes_closed_frames = self._eyes_closed_frames

        # ------------------------------------------------------------------ #
        # 5. Yawn detection (REFINED & OPTIMIZED for Jetson Nano)
        # ================================================================== #
        # Strategy: Detect clear open mouth persistence (3+ seconds minimum)
        # 1. Extract mouth region (lower 45% of face)
        # 2. Apply CLAHE + morphological operations for noise reduction
        # 3. Use combined Otsu + manual threshold for robust detection
        # 4. Compute aspect ratio and darkness ratio for confirmation
        # 5. Require BOTH metrics + MIN_YAWN_FRAMES frames for confirmation
        # ================================================================== #
        # ================================================================== #
        # 5. YAWN DETECTION - SIMPLIFIED & RELIABLE
        # ================================================================== #
        # Simple strategy: If mouth is open > 3 seconds → it's a yawn
        # No complex thresholds, just detect open mouth reliably
        # ================================================================== #
        mouth_open_score = 0.0
        yawning_raw      = False

        if face_found:
            fx, fy, fw, fh = face_box
            
            # Extract mouth region from BOTTOM 50% of face
            mouth_y_start = fy + int(fh * 0.50)
            mouth_y_end   = fy + fh
            mouth_x_start = fx
            mouth_x_end   = fx + fw
            
            # Clamp to ROI bounds
            mouth_y_start = max(0, mouth_y_start)
            mouth_y_end   = min(ROI_H, mouth_y_end)
            mouth_x_start = max(0, mouth_x_start)
            mouth_x_end   = min(ROI_W, mouth_x_end)
            
            mouth_h = mouth_y_end - mouth_y_start
            mouth_w = mouth_x_end - mouth_x_start
            
            if mouth_h > YAWN_MIN_SIZE and mouth_w > YAWN_MIN_SIZE:
                mouth_region = gray[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
                
                if mouth_region.size > 0:
                    try:
                        # CLAHE for better contrast
                        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
                        mouth_region_enhanced = clahe.apply(mouth_region)
                    except Exception:
                        mouth_region_enhanced = mouth_region
                    
                    mouth_region_enhanced = np.clip(mouth_region_enhanced, 0, 255).astype(np.uint8)
                    
                    # Simple threshold - just find dark regions (mouth interior)
                    _, binary = cv2.threshold(mouth_region_enhanced, MOUTH_THRESHOLD_VALUE, 255, 
                                             cv2.THRESH_BINARY_INV)
                    
                    # Clean up with morphology
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                    
                    # Count dark pixels
                    dark_pixels = np.count_nonzero(binary)
                    total_pixels = mouth_region.size
                    dark_frac = float(dark_pixels) / total_pixels if total_pixels > 0 else 0.0
                    
                    # Compute aspect ratio (height/width)
                    aspect_ratio = float(mouth_h) / (mouth_w + 1e-6)
                    
                    # SIMPLE CHECK: Is mouth open?
                    # Just need EITHER high darkness OR tall shape
                    is_dark_enough = dark_frac >= MOUTH_DARK_RATIO
                    is_tall_enough = aspect_ratio >= MOUTH_OPEN_RATIO
                    
                    # Mouth is open if either condition met (OR logic - more lenient)
                    yawning_raw = is_dark_enough or is_tall_enough
                    
                    # Score for visualization
                    mouth_open_score = min(1.0, max(dark_frac, aspect_ratio / 0.5))

        self._mouth_smoother.push(mouth_open_score)
        result.mouth_open_raw = self._mouth_smoother.mean()

        # ================================================================== #
        # Stage 5: Simple frame counting - mouth open for 3+ seconds = YAWN
        # ================================================================== #
        # Decrement cooldown each frame
        if self._yawn_cooldown > 0:
            self._yawn_cooldown -= 1
        
        if yawning_raw:
            self._mouth_open_count += 1
        else:
            # Quick decay - when mouth closes, fast transition
            self._mouth_open_count = max(0, self._mouth_open_count - 2)

        # Check if we've reached 3 seconds (36 frames @ 12fps) AND not in cooldown
        if self._mouth_open_count >= MIN_YAWN_FRAMES and self._yawn_cooldown == 0:
            # Mouth has been open for 3+ seconds → YAWN DETECTED
            if not self._yawn_in_progress:
                # NEW YAWN STARTED
                self._yawn_in_progress = True
                self._yawn_start_time = time.time()
                result.yawn_state = "YAWNING"
            else:
                # YAWN CONTINUING
                result.yawn_state = "YAWNING"
        else:
            result.yawn_state = "NO_YAWN"

        # End yawn when mouth closes after being open OR timeout occurs
        if self._yawn_in_progress:
            # Calculate current yawn duration
            current_duration = time.time() - self._yawn_start_time if self._yawn_start_time else 0.0
            
            # End yawn if:
            # (A) Mouth closes (count drops below 50% threshold), OR
            # (B) Yawn exceeded maximum duration (person stuck with mouth open)
            should_end_yawn = (
                self._mouth_open_count < MIN_YAWN_FRAMES * 0.5 or
                current_duration > 10.0  # Force end after 10 seconds max
            )
            
            if should_end_yawn:
                # Record yawn duration (clamp to realistic range)
                result.yawn_duration_sec = max(0.5, min(15.0, current_duration))
                
                # Reset state and start cooldown
                self._yawn_in_progress = False
                self._yawn_start_time = None
                self._mouth_open_count = 0
                self._yawn_cooldown = YAWN_COOLDOWN_FRAMES  # Prevent duplicate detection
                result.yawn_state = "NO_YAWN"

        # ================================================================== #
        # 6. Sleep detection with motion analysis (OPTIMIZED for Jetson)
        # ================================================================== #
        # CRITICAL FIX: Only detect sleeping if HEAD is DOWN for 10+ seconds
        #              while eyes are closed. This prevents false positives
        #              from note-taking (head down, eyes looking at paper).
        #
        # Strategy: Sleeping = head down 10+ seconds + eyes closed + minimal motion
        #           Note-taking = head down + eyes open OR visible motion
        #           Drowsy = head down + eyes closed < 10 seconds
        # ================================================================== #
        
        # Calculate face motion (lightweight operation - no sqrt)
        if face_found:
            self._calculate_face_motion(face_box)
        else:
            # No face = no motion data
            self._prev_face_center = None
        
        avg_face_motion = self._get_avg_face_motion()
        motion_buffer_ready = len(self._face_motion_buffer) >= 3
        
        # Calculate head-down duration (only if head is currently down)
        head_down_duration_secs = 0.0
        if self._head_is_down and self._head_down_start_time is not None:
            head_down_duration_secs = time.time() - self._head_down_start_time
        
        # SLEEP DETECTION LOGIC (STRICT & RELIABLE)
        # Sleep requires: Head DOWN for 25+ seconds + Eyes CLOSED + Minimal motion
        if result.head_state == "DOWN":
            # Head is DOWN - check for sleeping conditions
            
            if head_down_duration_secs >= 25.0:
                # Head has been down for 25+ seconds - now check eyes for sleep confirmation
                if result.eye_state == "CLOSED":
                    # Eyes closed + head down 25+ seconds
                    # Even without motion data, if head is down + eyes closed for 25+ sec → SLEEPING
                    # (motion buffer might not be ready on first frames, so be lenient)
                    if not motion_buffer_ready or avg_face_motion < MOTION_THRESHOLD * SLEEP_MOTION_RATIO:
                        # Very still + eyes closed for 25+ seconds → Sleeping
                        result.sleep_state = "SLEEPING"
                    else:
                        # Some motion despite head down → Drowsy but not fully asleep
                        result.sleep_state = "DROWSY"
                elif result.eye_state == "DROWSY":
                    # Eyes partially closed + head down 25+ seconds
                    if not motion_buffer_ready or avg_face_motion < MOTION_THRESHOLD * SLEEP_MOTION_RATIO:
                        # Partially closed eyes + head down 25+ sec → SLEEPING (early stage)
                        result.sleep_state = "SLEEPING"
                    else:
                        # Active motion = likely reading/writing
                        result.sleep_state = "DROWSY"
                else:
                    # Eyes open + head down = note-taking or looking at material
                    result.sleep_state = "AWAKE"
            
            else:
                # Head down but < 25 seconds - can't be sleeping yet
                if result.eye_state == "CLOSED" or result.eye_state == "DROWSY":
                    result.sleep_state = "DROWSY"
                else:
                    result.sleep_state = "AWAKE"
        
        else:
            # Head is UP - only sleep if eyes closed 10+ seconds regardless
            if self._eyes_closed_frames >= SLEEP_CONFIRM_FRAMES:
                # Eyes closed for 10+ seconds with head up = unusual, likely sleeping
                result.sleep_state = "SLEEPING"
            else:
                result.sleep_state = "AWAKE"

        # ------------------------------------------------------------------ #
        # 7. Engagement score (composite)
        # ------------------------------------------------------------------ #
        score = 0.0
        if result.eye_state == "OPEN":
            score += W_EYE
        elif result.eye_state == "DROWSY":
            score += W_EYE * 0.4
        # CLOSED eyes = 0 score for this component

        if result.head_state == "UP":
            score += W_HEAD
        elif result.head_state == "UNKNOWN":
            score += W_HEAD * 0.5  # partial credit when uncertain

        # Penalize for yawning (sign of drowsiness)
        if result.yawn_state == "YAWNING":
            score -= W_YAWN * 0.6  # Reduce score when yawning
        elif result.yawn_state == "NO_YAWN":
            score += W_YAWN  # Full credit when no yawning

        if face_found:
            score += W_PRESENT

        result.engagement_score = round(max(0.0, min(score, 1.0)), 3)

        # Determine engagement label
        if result.sleep_state == "SLEEPING":
            result.engagement = "SLEEPING"
        elif result.engagement_score >= 0.65:
            result.engagement = "ENGAGED"
        elif result.eye_state == "CLOSED" or result.head_state == "DOWN":
            result.engagement = "DROWSY"
        else:
            result.engagement = "DISTRACTED"

        return result

    # ----------------------------------------------------------------------- #
    def reset(self):
        """Reset all state (call when participant leaves / session restarts)."""
        self._mouth_open_count = 0
        self._yawn_cooldown    = 0
        self._yawn_in_progress = False
        self._yawn_start_time  = None
        self._no_face_count    = 0
        self._head_down_count  = 0
        self._head_down_start_time = None
        self._head_is_down     = False
        self._eyes_closed_frames = 0
        self._prev_face_center = None
        self._face_motion_buffer.clear()
        self._eye_smoother.reset()
        self._mouth_smoother.reset()
