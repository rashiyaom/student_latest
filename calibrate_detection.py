#!/usr/bin/env python3
# =============================================================================
# calibrate_detection.py
# EvilEye — Protect your sessions from evil doing | Detection Calibration & Debugging Tool
#
# Use this script to fine-tune yawn and eye detection parameters.
# Especially useful when wearing glasses or under different lighting.
#
# Usage:
#   python3 calibrate_detection.py
#
# Controls:
#   q = quit
#   r = reset counters
#   y = force trigger yawn for testing
#   d = dump current frame analysis to debug.txt
#   SPACE = capture frame for analysis
#
# Output: Detailed frame-by-frame analysis with raw metric values
#
# Compatible with: Python 3.6+ / Jetson Nano
# =============================================================================

import cv2
import numpy as np
import time
import os
from behavior_analyzer import BehaviourAnalyzer

class YawnCalibrator:
    def __init__(self):
        print("[Calibrator] Initializing behavior analyzer...")
        self.analyzer = BehaviourAnalyzer()
        
        print("[Calibrator] Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[Calibrator] ERROR: Cannot open webcam!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_count = 0
        self.yawn_count = 0
        self.last_yawn_state = "NO_YAWN"
        self.debug_output = []
        
        print("[Calibrator] Ready!")
        print("Controls:")
        print("  q = quit")
        print("  r = reset")
        print("  y = force yawn trigger (test)")
        print("  d = dump debug info")
        print("  SPACE = capture frame analysis")
        print()
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Calibrator] Frame read failed")
                continue
            
            self.frame_count += 1
            h, w = frame.shape[:2]
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Display frame info
            info_text = f"Frame: {self.frame_count} | Yawns: {self.yawn_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            if len(faces) > 0:
                # Analyze largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w_face, h_face = faces[0]
                
                person_roi = frame[y:y+h_face, x:x+w_face]
                result = self.analyzer.analyse(person_roi, person_present=True)
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
                
                # Display all metrics
                metrics_text = (
                    f"Eye: {result.eye_state} ({result.eye_open_raw:.3f}) | "
                    f"Head: {result.head_state} | Yawn: {result.yawn_state} | "
                    f"Mouth: {result.mouth_open_raw:.3f}"
                )
                cv2.putText(frame, metrics_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 0), 1)
                
                # Additional debug info
                debug_text = f"Attention: {result.attention} [{result.attention_score:.2f}]"
                cv2.putText(frame, debug_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (200, 200, 0), 1)
                
                # Count yawn transitions
                if result.yawn_state == "YAWNING" and self.last_yawn_state == "NO_YAWN":
                    self.yawn_count += 1
                    ts = time.strftime("%H:%M:%S")
                    print(f"[Calibrator] ✓ YAWN #{self.yawn_count} at {ts} "
                          f"(duration: {result.yawn_duration_sec:.2f}s)")
                
                self.last_yawn_state = result.yawn_state
                
                # Draw mouth analysis region
                mouth_y_start = y + int(h_face * 0.55)
                mouth_y_end = y + h_face
                cv2.rectangle(frame, (x, mouth_y_start), (x+w_face, mouth_y_end), 
                             (255, 0, 0), 1)
                
                # Store debug data
                debug_info = {
                    'frame': self.frame_count,
                    'eye_state': result.eye_state,
                    'eye_raw': result.eye_open_raw,
                    'head_state': result.head_state,
                    'yawn_state': result.yawn_state,
                    'mouth_raw': result.mouth_open_raw,
                    'attention': result.attention,
                    'attention_score': result.attention_score,
                }
                self.debug_output.append(debug_info)
                
            else:
                cv2.putText(frame, "No face detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display info panel
            cv2.putText(frame, "Press SPACE to analyze, 'd' for debug, 'q' to quit",
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Yawn & Eye Detection Calibrator", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Calibrator] Quit.")
                break
            elif key == ord('r'):
                self.yawn_count = 0
                self.last_yawn_state = "NO_YAWN"
                self.analyzer.reset()
                self.debug_output = []
                print("[Calibrator] Reset counters.")
            elif key == ord('y'):
                print("[Calibrator] Forcing yawn trigger (for testing)...")
                self.yawn_count += 1
            elif key == ord('d'):
                self.dump_debug()
            elif key == 32:  # SPACE
                self.analyze_current_frame()
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def dump_debug(self):
        """Save debug output to file"""
        filename = f"debug_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write("=== YAWN & EYE DETECTION DEBUG LOG ===\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames: {self.frame_count}\n")
            f.write(f"Yawns detected: {self.yawn_count}\n")
            f.write("\n")
            
            f.write("Frame-by-frame analysis:\n")
            f.write("-" * 80 + "\n")
            for info in self.debug_output:
                f.write(f"Frame {info['frame']}: ")
                f.write(f"Eye={info['eye_state']}({info['eye_raw']:.3f}) ")
                f.write(f"Head={info['head_state']} ")
                f.write(f"Yawn={info['yawn_state']} ")
                f.write(f"Mouth={info['mouth_raw']:.3f} ")
                f.write(f"Attention={info['attention']}({info['attention_score']:.2f})\n")
        
        print(f"[Calibrator] Debug log saved: {filename}")
    
    def analyze_current_frame(self):
        """Detailed analysis of a single captured frame"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Analyze
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("[Calibrator] No face in captured frame")
            return
        
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w_face, h_face = faces[0]
        person_roi = frame[y:y+h_face, x:x+w_face]
        result = self.analyzer.analyse(person_roi, person_present=True)
        
        print("\n" + "="*70)
        print("[Calibrator] FRAME ANALYSIS SNAPSHOT")
        print("="*70)
        print(f"Face detected: YES (size: {w_face}x{h_face})")
        print(f"\nEYE DETECTION:")
        print(f"  State: {result.eye_state}")
        print(f"  Raw score: {result.eye_open_raw:.4f}")
        print(f"  Eyes closed frames: {result.eyes_closed_frames}")
        print(f"  Recommendation: {'✓ Working well' if result.eye_state != 'UNKNOWN' else '✗ Check cascade'}")
        
        print(f"\nMOUTH / YAWN DETECTION:")
        print(f"  State: {result.yawn_state}")
        print(f"  Mouth raw: {result.mouth_open_raw:.4f}")
        print(f"  Recommendation: {'✓ Working well' if result.mouth_open_raw > 0.1 or result.yawn_state == 'YAWNING' else '○ Neutral'}")
        
        print(f"\nHEAD POSITION:")
        print(f"  State: {result.head_state}")
        
        print(f"\nOVERALL ATTENTION:")
        print(f"  Score: {result.attention_score:.3f}")
        print(f"  Label: {result.attention}")
        print(f"  Sleep state: {result.sleep_state}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    calibrator = YawnCalibrator()
    calibrator.run()
