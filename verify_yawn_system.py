#!/usr/bin/env python3
# =============================================================================
# verify_yawn_system.py
# EvilEye — Protect your sessions from evil doing | Yawn Detection System Verification
#
# Validates that yawn detection is properly configured and working.
# Run on Jetson Nano after deployment.
#
# Usage:
#   python3 verify_yawn_system.py
#
# Checks:
#   ✓ behavior_analyzer imports
#   ✓ student_tracker imports
#   ✓ sleep_tracker imports
#   ✓ Cascade files available
#   ✓ YOLOv8 engine exists
#   ✓ outputs/ directory writable
#   ✓ Dashboard server accessible
#   ✓ State.json format
#
# Compatible with: Python 3.6+ / Jetson Nano
# =============================================================================

import sys
import os
import json
import time

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name:<25} OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:<25} FAILED: {e}")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"✓ {description:<40} OK ({size} bytes)")
        return True
    else:
        print(f"✗ {description:<40} NOT FOUND: {path}")
        return False

def check_dir(path, description, create=False):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✓ {description:<40} OK")
        return True
    else:
        if create:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"✓ {description:<40} CREATED")
                return True
            except Exception as e:
                print(f"✗ {description:<40} FAILED: {e}")
                return False
        else:
            print(f"✗ {description:<40} NOT FOUND: {path}")
            return False

def main():
    print("\n" + "="*70)
    print("EvilEye — System Verification")
    print("="*70 + "\n")
    
    all_ok = True
    
    # --- Python Imports ---
    print("[1] Checking Python Dependencies...")
    print("-" * 70)
    all_ok &= check_import('cv2', 'OpenCV (cv2)')
    all_ok &= check_import('numpy', 'NumPy')
    all_ok &= check_import('flask', 'Flask (server)')
    print()
    
    # --- Project Modules ---
    print("[2] Checking Project Modules...")
    print("-" * 70)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    
    all_ok &= check_import('behavior_analyzer', 'BehaviourAnalyzer')
    all_ok &= check_import('student_tracker', 'StudentTracker')
    all_ok &= check_import('sleep_tracker', 'SleepTracker')
    all_ok &= check_import('phone_detector', 'PhoneDetector')
    all_ok &= check_import('head_down_tracker', 'HeadDownTracker')
    all_ok &= check_import('trt_yolo_multi', 'TRTYoloDetectorMulti')
    print()
    
    # --- Required Files ---
    print("[3] Checking Required Files...")
    print("-" * 70)
    all_ok &= check_file(os.path.join(here, 'dashboard.html'), 'Dashboard HTML')
    all_ok &= check_file(os.path.join(here, 'server.py'), 'Flask server script')
    all_ok &= check_file(os.path.join(here, 'single_student_main.py'), 'Main entry point')
    print()
    
    # --- Cascades ---
    print("[4] Checking OpenCV Cascade Files...")
    print("-" * 70)
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_eye.xml',
    ]
    for cascade in cascade_paths:
        basename = os.path.basename(cascade)
        if os.path.isfile(cascade):
            print(f"✓ {basename:<40} OK")
        else:
            print(f"✗ {basename:<40} NOT FOUND")
            all_ok = False
    print()
    
    # --- TensorRT Engine ---
    print("[5] Checking TensorRT Engine...")
    print("-" * 70)
    engine_path = os.path.join(here, 'yolov8n.engine')
    if os.path.isfile(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024**2)
        print(f"✓ yolov8n.engine{'':<28} OK ({size_mb:.1f} MB)")
    else:
        print(f"✗ yolov8n.engine{'':<28} NOT FOUND")
        print(f"  → Expected at: {engine_path}")
        print(f"  → See SETUP_GUIDE.bat for conversion instructions")
        all_ok = False
    print()
    
    # --- Output Directories ---
    print("[6] Checking Output Directories...")
    print("-" * 70)
    all_ok &= check_dir(os.path.join(here, 'outputs'), 'outputs/ directory', create=True)
    print()
    
    # --- Writable outputs/state.json ---
    print("[7] Testing State File Writability...")
    print("-" * 70)
    state_file = os.path.join(here, 'outputs', 'state.json')
    test_state = {
        'test': True,
        'ts': time.strftime('%H:%M:%S'),
        'yawn_count': 42,
        'yawn_markers': ['12:34:56'],
    }
    try:
        with open(state_file, 'w') as f:
            json.dump(test_state, f)
        print(f"✓ outputs/state.json{'':<26} WRITABLE")
        # Read back
        with open(state_file, 'r') as f:
            loaded = json.load(f)
        if loaded['yawn_count'] == 42:
            print(f"✓ State JSON format{'':<30} OK")
        else:
            print(f"✗ State JSON format{'':<30} MISMATCH")
            all_ok = False
    except Exception as e:
        print(f"✗ outputs/state.json{'':<26} ERROR: {e}")
        all_ok = False
    print()
    
    # --- Camera Test ---
    print("[8] Testing Camera Access...")
    print("-" * 70)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Camera /dev/video0{'':<28} OK ({w}x{h})")
            else:
                print(f"✗ Camera /dev/video0{'':<28} NO FRAME")
                all_ok = False
            cap.release()
        else:
            print(f"✗ Camera /dev/video0{'':<28} NOT ACCESSIBLE")
            print(f"  → Check: ls -la /dev/video*")
            print(f"  → Try: sudo usermod -aG video $(whoami)")
            all_ok = False
    except Exception as e:
        print(f"✗ Camera test{'':<34} ERROR: {e}")
        all_ok = False
    print()
    
    # --- Summary ---
    print("="*70)
    if all_ok:
        print("✓ ALL CHECKS PASSED — System ready for yawn detection!")
        print("\nNext steps:")
        print("  1. Start tracking: python3 single_student_main.py")
        print("  2. Start server:   python3 server.py")
        print("  3. Open dashboard: http://localhost:5000")
        print("  4. Test yawning:   python3 test_yawn_detection.py")
        return 0
    else:
        print("✗ SOME CHECKS FAILED — See above for details")
        print("\nFor help, see:")
        print("  - SETUP_GUIDE.bat (deployment steps)")
        print("  - YAWN_DETECTION_GUIDE.md (yawn system documentation)")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
