#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_sleep_fix.py
Quick test to verify sleep detection is working after fix
"""

import time
from sleep_tracker import SleepTracker

print("=" * 70)
print("Sleep Tracker Verification Test")
print("=" * 70)

# Create tracker
tracker = SleepTracker()

print("\n1️⃣  Testing: Simulate 12+ frames of SLEEPING")
print("-" * 70)
for i in range(15):
    is_active = tracker.update("SLEEPING")
    if i == 14:
        print(f"   Frame {i+1}: update('SLEEPING') → active={is_active}")
        summary = tracker.get_summary()
        print(f"   Event count: {summary['event_count']}")
        print(f"   Current active: {summary['current_active']}")
        if summary['event_count'] > 0:
            print("   ✓ PASS: Sleep event counted!")
        else:
            print("   ✗ FAIL: Sleep event not counted")

print("\n2️⃣  Testing: Continue sleeping for 30 frames total")
print("-" * 70)
for i in range(15, 30):
    is_active = tracker.update("SLEEPING")

summary = tracker.get_summary()
print(f"   Total frames of SLEEPING: 30 (~2.5s @ 12fps)")
print(f"   Event count: {summary['event_count']}")
print(f"   Current streak: {summary['current_streak_sec']:.1f}s")

print("\n3️⃣  Testing: Wake up (12+ frames of AWAKE)")
print("-" * 70)
for i in range(12):
    is_active = tracker.update("AWAKE")
    if i == 11:
        print(f"   Frame {30+i}: update('AWAKE') → active={is_active}")
        
summary = tracker.get_summary()
print(f"   Event count after wake: {summary['event_count']}")
if summary['event_count'] >= 1 and not summary['current_active']:
    print("   ✓ PASS: Sleep event ended and counted!")
else:
    print("   ✗ FAIL: Sleep event not properly ended")

print(f"\n   Event details:")
for i, ev in enumerate(summary['events']):
    print(f"     Event #{i+1}: {ev['start']} | Duration: {ev['duration_sec']:.1f}s")

print("\n4️⃣  Testing: Multiple sleep events")
print("-" * 70)
tracker.reset()
print("   Reset tracker")

# First sleep event
print("   Simulating 1st sleep event...")
for _ in range(15):
    tracker.update("SLEEPING")
for _ in range(15):
    tracker.update("AWAKE")

# Second sleep event
print("   Simulating 2nd sleep event...")
for _ in range(15):
    tracker.update("SLEEPING")
for _ in range(15):
    tracker.update("AWAKE")

summary = tracker.get_summary()
print(f"   Total events: {summary['event_count']}")
if summary['event_count'] >= 2:
    print("   ✓ PASS: Multiple sleep events counted!")
else:
    print("   ✗ FAIL: Multiple sleep events not counted")

print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)
print("\nKey metrics:")
print(f"  • MIN_SLEEP_FRAMES = 12 (about 1 second @ 12fps)")
print(f"  • MIN_AWAKE_FRAMES = 12 (about 1 second @ 12fps)")
print(f"  • Event threshold in behavior_analyzer = 25 seconds head-down")
print("\nAll tests should show ✓ PASS for proper functionality")
