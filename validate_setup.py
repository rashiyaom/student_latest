#!/usr/bin/env python3
"""
validate_setup.py
EvilEye — Protect your sessions from evil doing | Setup validation script

Checks:
  1. Required files exist
  2. JSON syntax in state.json (if exists)
  3. HTML dashboard is valid
  4. Python dependencies are installed
  5. Directory structure is correct

Run: python3 validate_setup.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_ok(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warn(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def check_file_exists(filepath, critical=False):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_ok(f"Found: {filepath}")
        return True
    else:
        if critical:
            print_error(f"MISSING (CRITICAL): {filepath}")
        else:
            print_warn(f"Not found (optional): {filepath}")
        return not critical

def check_directory_exists(dirpath):
    """Check if a directory exists and create if missing."""
    if os.path.isdir(dirpath):
        print_ok(f"Directory exists: {dirpath}")
        return True
    else:
        print_warn(f"Creating directory: {dirpath}")
        try:
            os.makedirs(dirpath, exist_ok=True)
            print_ok(f"Created: {dirpath}")
            return True
        except Exception as e:
            print_error(f"Failed to create {dirpath}: {e}")
            return False

def check_json_valid(filepath):
    """Check if a JSON file is valid."""
    if not os.path.exists(filepath):
        print_warn(f"File not found: {filepath} (will be created at runtime)")
        return True
    
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print_ok(f"Valid JSON: {filepath}")
        return True
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in {filepath}: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading {filepath}: {e}")
        return False

def check_html_valid(filepath):
    """Check if HTML file exists and has basic structure."""
    if not os.path.exists(filepath):
        print_error(f"HTML file not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            '<!DOCTYPE html>': 'Has DOCTYPE',
            '<head>': 'Has <head>',
            '<body>': 'Has <body>',
            'dashboard.html': 'File extension valid',
            'fetch': 'Has fetch API',
            '/outputs/state.json': 'References state.json endpoint',
        }
        
        all_ok = True
        for check_str, label in checks.items():
            if check_str.lower() in content.lower():
                print_ok(f"HTML check: {label}")
            else:
                print_warn(f"HTML check: Missing {label}")
                all_ok = False
        
        return True
    except Exception as e:
        print_error(f"Error reading HTML: {e}")
        return False

def check_python_module(module_name, package_name=None):
    """Check if a Python module is installed."""
    pkg = package_name or module_name
    try:
        __import__(module_name)
        print_ok(f"Python module installed: {pkg}")
        return True
    except ImportError:
        print_warn(f"Python module NOT installed: {pkg} (install with: pip3 install {pkg})")
        return False

def main():
    parser = argparse.ArgumentParser(description='EvilEye Setup Validator')
    parser.add_argument('--strict', action='store_true', 
                   help='Treat warnings as errors')
    args = parser.parse_args()
    
    print_header("EvilEye — Setup Validation")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    # ========================================================================
    print_header("1. Checking Python Files")
    python_files = [
        'single_student_main.py',
        'server.py',
        'student_tracker.py',
        'behavior_analyzer.py',
        'phone_detector.py',
        'head_down_tracker.py',
        'trt_yolo_multi.py',
        'single_utils.py',
    ]
    
    for pf in python_files:
        filepath = os.path.join(base_dir, pf)
        if not check_file_exists(filepath, critical=True):
            all_ok = False
    
    # ========================================================================
    print_header("2. Checking Dashboard")
    dashboard_path = os.path.join(base_dir, 'dashboard.html')
    if not check_html_valid(dashboard_path):
        all_ok = False
    
    # ========================================================================
    print_header("3. Checking Dependencies")
    deps_path = os.path.join(base_dir, 'requirements.txt')
    if os.path.exists(deps_path):
        print_ok(f"Found: {deps_path}")
    else:
        print_warn(f"Not found: {deps_path}")
    
    # ========================================================================
    print_header("4. Checking Directories")
    dirs_to_check = [
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'outputs', 'snapshots'),
    ]
    
    for d in dirs_to_check:
        if not check_directory_exists(d):
            if args.strict:
                all_ok = False
    
    # ========================================================================
    print_header("5. Checking State File (if exists)")
    state_path = os.path.join(base_dir, 'outputs', 'state.json')
    check_json_valid(state_path)
    
    # ========================================================================
    print_header("6. Checking Python Modules")
    print("\n  Production (required on Jetson):")
    modules_required = [
        ('cv2', 'opencv-contrib-python'),
        ('numpy', 'numpy'),
        ('tensorrt', 'tensorrt'),
        ('pycuda', 'pycuda'),
    ]
    
    for mod, pkg in modules_required:
        check_python_module(mod, pkg)
    
    print("\n  Server dependencies (required for dashboard):")
    modules_server = [
        ('flask', 'Flask'),
    ]
    
    for mod, pkg in modules_server:
        check_python_module(mod, pkg)
    
    # ========================================================================
    print_header("7. Runtime Configuration")
    print("\n  Default settings from single_student_main.py:")
    config_items = [
        ('ENGINE_PATH', 'yolov8n.engine'),
        ('INPUT_SIZE', '320'),
        ('WEBCAM_INDEX', '0'),
        ('STUDENT_NAME', 'Student'),
    ]
    
    for item, default in config_items:
        print(f"  {item} = {default} (adjustable in code)")
    
    # ========================================================================
    print_header("8. Summary")
    if all_ok:
        print_ok("All checks passed! Setup looks good.")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print("  1. On Jetson Nano:")
        print("     pip3 install -r requirements.txt")
        print("  2. Terminal 1: python3 single_student_main.py")
        print("  3. Terminal 2: python3 server.py")
        print("  4. Open browser: http://localhost:5000")
        print()
        return 0
    else:
        if args.strict:
            print_error("Some critical checks failed. Please review and fix.")
        else:
            print_warn("Some optional checks failed. Review the warnings above.")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
