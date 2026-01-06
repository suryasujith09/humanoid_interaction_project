#!/usr/bin/env python3
"""
Ultra-Fast Humanoid Mimic System
Watches human skeleton → Matches pose → Triggers robot action
"""

import cv2
import mediapipe as mp
import numpy as np
import serial
import sqlite3
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# ===================== CONFIG =====================
CAMERA_PATH = "/dev/usb_cam"   # ❗ NOT CHANGED (as requested)
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_TIME = 2.0  # seconds

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM STARTING")
print("=" * 60)

# ===================== MEDIAPIPE =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ===================== POSE SIGNATURE =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable

def check_hands_up(lm):
    left_up = lm[15].y < lm[0].y - 0.1
    right_up = lm[16].y < lm[0].y - 0.1
    close = abs(lm[15].x - lm[16].x) < 0.4
    return left_up and right_up and close

def check_hands_straight(lm):
    shoulder_y = (lm[11].y + lm[12].y) / 2
    left = abs(lm[15].y - shoulder_y) < 0.15 and lm[15].x < lm[11].x - 0.2
    right = abs(lm[16].y - shoulder_y) < 0.15 and lm[16].x > lm[12].x + 0.2
    return left and right

def check_wave(lm):
    return lm[16].y < lm[12].y - 0.2 and lm[16].x > lm[0].x

def check_hands_down(lm):
    hip_y = (lm[23].y + lm[24].y) / 2
    return lm[15].y > hip_y - 0.1 and lm[16].y > hip_y - 0.1

def check_place_block(lm):
    shoulder_y = (lm[11].y + lm[12].y) / 2
    chest = abs(lm[15].y - shoulder_y) < 0.2 and abs(lm[16].y - shoulder_y) < 0.2
    forward = lm[15].z < -0.1 and lm[16].z < -0.1
    return chest and forward

POSES = [
    PoseSignature("HANDS UP", "hands_up.d6a", check_hands_up),
    PoseSignature("T-POSE", "hands_straight.d6a", check_hands_straight),
    PoseSignature("WAVE", "wave.d6a", check_wave),
    PoseSignature("PLACE BLOCK", "placeblock.d6a", check_place_block),
    PoseSignature("HANDS DOWN", "greet.d6a", check_hands_down),
]

print(f"📋 Loaded {len(POSES)} poses")
for p in POSES:
    print(f" - {p.name} → {p.action_file}")

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port, baud):
        try:
            print(f"\n🔌 Connecting to {port} @ {baud}")
            self.serial = serial.Serial(port, baud, timeout=0.1)
            time.sleep(0.5)
            print("✅ Robot connected")
        except Exception as e:
            print(f"⚠️ Serial unavailable: {e}")
            print("⚠️ DEMO MODE enabled")
            self.serial = None

    def play_action(self, action_file, pose_name):
        print("\n" + "=" * 50)
        print(f"🎬 ACTION: {pose_name}")
        path = Path(ACTIONS_DIR) / action_file

        if not path.exists():
            print(f"❌ Missing file: {path}")
            return

        if not self.serial:
            print("🤖 DEMO MODE (no servo output)")
            time.sleep(0.5)
            return

        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        frames = cur.execute(
            "SELECT * FROM frames ORDER BY frame_num"
        ).fetchall()
        conn.close()

        print(f"📊 Frames: {len(frames)}")
        for i, frame in enumerate(frames):
            positions = frame[1:25]
            cmd = "#000P{:04d}".format(positions[0])
            for p in positions[1:]:
                cmd += "P{:04d}".format(p)
            cmd += "T0100\r\n"
            self.serial.write(cmd.encode())
            time.sleep(0.02)

# ===================== INIT =====================
robot = RobotController(SERIAL_PORT, BAUD_RATE)

cap = cv2.VideoCapture(CAMERA_PATH)
if not cap.isOpened():
    print("❌ Camera not accessible at", CAMERA_PATH)
    sys.exit(1)

last_trigger = {}

print("\n🎥 Camera started")
print("⎋ Press ESC to exit\n")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame read failed")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        for p in POSES:
            if p.check_func(lm):
                now = time.time()
                if now - last_trigger.get(p.name, 0) > COOLDOWN_TIME:
                    robot.play_action(p.action_file, p.name)
                    last_trigger[p.name] = now

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Humanoid Mimic System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")
