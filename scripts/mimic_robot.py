#!/usr/bin/env python3
"""
Ultra-Fast Humanoid Mimic System
Watches human skeleton → Matches pose → Triggers robot action
"""

import cv2
import mediapipe as mp
import serial
import sqlite3
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# ===================== CONFIG =====================
CAMERA_PATH = "/dev/usb_cam"        # ❗ KEPT AS REQUESTED
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
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
    return (
        lm[15].y < lm[0].y - 0.1 and
        lm[16].y < lm[0].y - 0.1 and
        abs(lm[15].x - lm[16].x) < 0.4
    )

def check_hands_straight(lm):
    shoulder_y = (lm[11].y + lm[12].y) / 2
    return (
        abs(lm[15].y - shoulder_y) < 0.15 and
        abs(lm[16].y - shoulder_y) < 0.15 and
        lm[15].x < lm[11].x - 0.2 and
        lm[16].x > lm[12].x + 0.2
    )

def check_wave(lm):
    return lm[16].y < lm[12].y - 0.2 and lm[16].x > lm[0].x

def check_hands_down(lm):
    hip_y = (lm[23].y + lm[24].y) / 2
    return lm[15].y > hip_y - 0.1 and lm[16].y > hip_y - 0.1

def check_place_block(lm):
    shoulder_y = (lm[11].y + lm[12].y) / 2
    return (
        abs(lm[15].y - shoulder_y) < 0.2 and
        abs(lm[16].y - shoulder_y) < 0.2 and
        lm[15].z < -0.1 and
        lm[16].z < -0.1
    )

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

        # 🔍 Detect correct table name automatically
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cur.fetchall()]

        if "frames" in tables:
            table = "frames"
            start_idx = 1
        elif "ActionGroup" in tables:
            table = "ActionGroup"
            start_idx = 2
        else:
            print(f"❌ Unsupported .d6a format. Tables: {tables}")
            conn.close()
            return

        frames = cur.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        conn.close()

        print(f"📊 Executing {len(frames)} frames")

        for frame in frames:
            positions = frame[start_idx:start_idx + 24]

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
print("⎋ Press ESC to exit")

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

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")
