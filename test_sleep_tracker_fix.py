#!/usr/bin/env python3
"""Quick verification that sleep_tracker.py is fixed"""

import sys
import time

# Test SleepTracker import and basic functionality
try:
    from sleep_tracker import SleepTracker, MIN_SLEEP_FRAMES, MIN_AWAKE_FRAMES
    print("✓ sleep_tracker.py imports successfully")
    
    # Create instance
    tracker = SleepTracker()
    print("✓ SleepTracker() instantiates correctly")
    
    # Test update method
    tracker.update("AWAKE")
    print("✓ tracker.update('AWAKE') works")
    
    tracker.update("SLEEPING")
    print("✓ tracker.update('SLEEPING') works")
    
    # Test get_summary
    summary = tracker.get_summary()
    print(f"✓ tracker.get_summary() works: {summary}")
    
    print("\n✅ All sleep_tracker tests PASSED!")
    
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
