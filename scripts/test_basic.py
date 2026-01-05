#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')

print("?? Testing Basic System Components...")
print("="*60)

# Test 1: Import logger
print("\n1?? Testing logger...")
try:
    from utils.logger import get_logger
    logger = get_logger("BasicTest")
    logger.info("Logger initialized successfully")
    print("? Logger works!")
except Exception as e:
    print(f"? Logger failed: {e}")
    sys.exit(1)

# Test 2: Import Board
print("\n2?? Testing Board connection...")
try:
    from ros_robot_controller_sdk import Board
    board = Board(device="/dev/ttyAMA0", baudrate=1000000, timeout=5)
    print("? Board connection works!")
    print(f"   Device: /dev/ttyAMA0")
except Exception as e:
    print(f"? Board connection failed: {e}")
    print("   This is OK if robot is not powered on")

# Test 3: Check action files
print("\n3?? Checking action files...")
import os
action_path = "/home/ubuntu/software/ainex_controller/ActionGroups"
if os.path.exists(action_path):
    actions = [f[:-4] for f in os.listdir(action_path) if f.endswith('.d6a')]
    print(f"? Found {len(actions)} actions")
    print(f"   Examples: {', '.join(actions[:5])}")
else:
    print(f"? Action path not found: {action_path}")

# Test 4: Check ROS
print("\n4?? Testing ROS...")
try:
    import rospy
    print("? ROS Python library available")
except Exception as e:
    print(f"? ROS import failed: {e}")

print("\n" + "="*60)
print("? Basic system test complete!")
print("="*60)
