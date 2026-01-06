#!/usr/bin/env python3
"""
🔥 Ultra-Stable Humanoid Mimic System
Human Pose → Action → REAL Servo Motion
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
CAMERA_INDEX = 0                 # ✅ reliable
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 3.0              # ✅ longer cooldown
SERVO_COUNT = 24
FRAME_DELAY = 0.03               # ✅ servo safe

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
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ===================== POSE SIGNATURE =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable

def check_hands_up(lm):
    return lm[15].y < lm[0].y and lm[16].y < lm[0].y

def check_hands_straight(lm):
    sy = (lm[11].y + lm[12].y) / 2
    return abs(lm[15].y - sy) < 0.15 and abs(lm[16].y - sy) < 0.15

def check_wave(lm):
    return lm[16].y < lm[12].y - 0.2

def check_hands_down(lm):
    hy = (lm[23].y + lm[24].y) / 2
    return lm[15].y > hy and lm[16].y > hy

def check_place_block(lm):
    return lm[15].z < -0.15 and lm[16].z < -0.15

POSES = [
    PoseSignature("HANDS UP", "hands_up.d6a", check_hands_up),
    PoseSignature("T-POSE", "hands_straight.d6a", check_hands_straight),
    PoseSignature("WAVE", "wave.d6a", check_wave),
    PoseSignature("PLACE BLOCK", "placeblock.d6a", check_place_block),
    PoseSignature("HANDS DOWN", "greet.d6a", check_hands_down),
]

print(f"📋 Loaded {len(POSES)} poses")

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port, baud):
        try:
            print(f"\n🔌 Connecting to {port}")
            self.serial = serial.Serial(
                port, baud,
                timeout=0.1,
                write_timeout=0.1,
                rtscts=False,
                dsrdtr=False
            )
            time.sleep(1)
            self.busy = False
            print("✅ Robot connected")
        except Exception as e:
            print(f"⚠️ Serial failed: {e}")
            self.serial = None
            self.busy = False

    def play_action(self, action_file, pose_name):
        if self.busy:
            return

        path = Path(ACTIONS_DIR) / action_file
        if not path.exists():
            print(f"❌ Missing {path}")
            return

        print("\n" + "=" * 50)
        print(f"🎬 ACTION → {pose_name}")

        if not self.serial:
            print("🤖 DEMO MODE")
            time.sleep(1)
            return

        self.busy = True

        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cur.fetchall()]

        table = "frames" if "frames" in tables else "ActionGroup"
        start_idx = 1 if table == "frames" else 2

        frames = cur.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        conn.close()

        print(f"📊 Frames: {len(frames)}")

        for frame in frames:
            positions = frame[start_idx:start_idx + SERVO_COUNT]

            if len(positions) < SERVO_COUNT:
                continue

            cmd = ""
            for i, p in enumerate(positions):
                if p < 500 or p > 2500:
                    p = 1500
                cmd += f"#{i:03d}P{int(p):04d}"
            cmd += "T100\r\n"

            self.serial.write(cmd.encode())
            time.sleep(FRAME_DELAY)

        self.busy = False

# ===================== INIT =====================
robot = RobotController(SERIAL_PORT, BAUD_RATE)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Camera failed")
    sys.exit(1)

last_trigger = {}

print("\n🎥 Camera started | ESC to exit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
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
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Humanoid Mimic System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")
