# =============================================================================
# test_yawn_detection.py
# ClassPulse AI — Yawn Detection Tester
#
# Run this standalone to debug yawn detection without the full pipeline.
# Useful for Jetson Nano testing and calibration.
#
# Usage:
#   python3 test_yawn_detection.py
#
# Controls:
#   q = quit
#   r = reset counters
#   s = save frame for analysis
#
# Compatible with: Python 3.6 / OpenCV 4.x / Jetson Nano 2GB
# =============================================================================

import cv2
import numpy as np
import time
import os
from behavior_analyzer import BehaviourAnalyzer

def main():
    print("[YawnTest] Initializing behavior analyzer...")
    analyzer = BehaviourAnalyzer()
    
    print("[YawnTest] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[YawnTest] ERROR: Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[YawnTest] Ready. Press 'q' to quit, 'r' to reset, 's' to save frame")
    print("[YawnTest] Please yawn naturally in front of camera to test detection")
    
    yawn_count = 0
    frame_count = 0
    yawn_events = []
    last_yawn_state = "NO_YAWN"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[YawnTest] Frame read failed")
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Detect face in frame (simple detection using cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Get face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        info_text = f"Frame: {frame_count} | Yawns: {yawn_count}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        if len(faces) > 0:
            # Take largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w_face, h_face = faces[0]
            
            # Crop and analyze
            person_roi = frame[y:y+h_face, x:x+w_face]
            
            # Run analyzer
            result = analyzer.analyse(person_roi, person_present=True)
            
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
            
            # Display results
            result_text = (
                f"Eye: {result.eye_state} | Head: {result.head_state} | "
                f"Yawn: {result.yawn_state} | Mouth: {result.mouth_open_raw:.2f}"
            )
            cv2.putText(frame, result_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 0), 1)
            
            # Count yawn transitions
            if result.yawn_state == "YAWNING" and last_yawn_state == "NO_YAWN":
                yawn_count += 1
                ts = time.strftime("%H:%M:%S")
                yawn_events.append(ts)
                print(f"[YawnTest] ✓ YAWN #{yawn_count} detected at {ts}")
            
            last_yawn_state = result.yawn_state
            
            # Draw mouth analysis box for debug
            mouth_y_start = y + int(h_face * 0.55)
            mouth_y_end = y + h_face
            cv2.rectangle(frame, (x, mouth_y_start), (x+w_face, mouth_y_end), 
                         (255, 0, 0), 1)
            cv2.putText(frame, "Mouth Region", (x, mouth_y_start - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            cv2.putText(frame, "No face detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Yawn Detection Test", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[YawnTest] Quit.")
            break
        elif key == ord('r'):
            yawn_count = 0
            yawn_events = []
            last_yawn_state = "NO_YAWN"
            analyzer.reset()
            print("[YawnTest] Reset counters.")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"yawn_test_{ts}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[YawnTest] Saved: {fn}")
    
    # Summary
    print("\n" + "="*60)
    print("[YawnTest] SESSION SUMMARY")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Yawns detected: {yawn_count}")
    if yawn_events:
        print("Yawn timestamps:")
        for i, ts in enumerate(yawn_events, 1):
            print(f"  {i}. {ts}")
    print("="*60)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
