#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import sqlite3
import threading
import argparse
import signal

import rospy
from std_msgs.msg import String

import mediapipe as mp
from pathlib import Path

# ─────────────────────────────────────────────
# SDK PATHS (DO NOT CHANGE)
# ─────────────────────────────────────────────
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

try:
    from ainex_sdk import Board
except ImportError:
    print("❌ AiNex SDK not found. Run on robot.")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CUSTOM_ACTIONS = "/home/ubuntu/humanoid_interaction_project/actions/custom"
SYSTEM_ACTIONS = "/home/ubuntu/software/ainex_controller/ActionGroups"
COOLDOWN = 2.5

POSE_ACTION_MAP = {
    "hands_up": "hands_up",
    "hands_straight": "hands_straight",
    "wave": "wave",
    "stand": "stand"
}

# ─────────────────────────────────────────────
# ROBOT CONTROLLER
# ─────────────────────────────────────────────
class RobotController:
    def __init__(self):
        rospy.loginfo("🔌 Connecting to robot...")
        try:
            try:
                self.board = Board.Board()
            except AttributeError:
                self.board = Board()
            rospy.loginfo("✅ Robot connected")
        except Exception as e:
            rospy.logerr(f"Hardware error: {e}")
            sys.exit(1)

        self.lock = threading.Lock()

    def play_action(self, action_name):
        if not self.lock.acquire(False):
            return

        try:
            filename = action_name + ".d6a"
            path = Path(CUSTOM_ACTIONS) / filename
            if not path.exists():
                path = Path(SYSTEM_ACTIONS) / filename
                if not path.exists():
                    rospy.logwarn(f"Action missing: {filename}")
                    return

            rospy.loginfo(f"🎬 Playing action: {action_name}")

            conn = sqlite3.connect(str(path))
            cur = conn.cursor()

            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cur.fetchall()]
            table = "ActionGroup" if "ActionGroup" in tables else "frames"

            frames = cur.execute(f"SELECT * FROM {table} ORDER BY [Index]").fetchall()
            conn.close()

            for f in frames:
                duration = f[1]
                servos = f[2:]

                for i, pos in enumerate(servos):
                    sid = i + 1
                    if sid > 22:
                        break
                    self.board.setBusServoPulse(sid, pos, duration)

                time.sleep(duration / 1000.0)

        finally:
            self.lock.release()

# ─────────────────────────────────────────────
# VISION SYSTEM
# ─────────────────────────────────────────────
class VisionSystem:
    def __init__(self, camera, robot):
        self.camera = camera
        self.robot = robot
        self.last = {}

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.draw = mp.solutions.drawing_utils

    def detect_pose(self, lm):
        if lm[15].y < lm[0].y and lm[16].y < lm[0].y:
            return "hands_up"

        sy = (lm[11].y + lm[12].y) / 2
        if abs(lm[15].y - sy) < 0.2 and abs(lm[16].y - sy) < 0.2:
            return "hands_straight"

        if lm[16].y < lm[12].y - 0.2:
            return "wave"

        hy = (lm[23].y + lm[24].y) / 2
        if lm[15].y > hy and lm[16].y > hy:
            return "stand"

        return None

    def run(self):
        cap = cv2.VideoCapture(int(self.camera)) if self.camera.isdigit() else cv2.VideoCapture(self.camera)

        if not cap.isOpened():
            rospy.logerr("Camera failed")
            return

        rospy.loginfo("📷 Vision started")

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if res.pose_landmarks:
                self.draw.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                pose = self.detect_pose(res.pose_landmarks.landmark)

                if pose:
                    now = time.time()
                    if now - self.last.get(pose, 0) > COOLDOWN:
                        action = POSE_ACTION_MAP.get(pose)
                        threading.Thread(target=self.robot.play_action, args=(action,)).start()
                        self.last[pose] = now

            cv2.imshow("Humanoid Mimic", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# ROS TRIGGER
# ─────────────────────────────────────────────
def ros_callback(msg):
    action = msg.data.strip()
    rospy.loginfo(f"📡 ROS trigger: {action}")
    threading.Thread(target=robot.play_action, args=(action,)).start()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    args = parser.parse_args()

    rospy.init_node("ultimate_humanoid_mimic", anonymous=False)

    global robot
    robot = RobotController()

    rospy.Subscriber("/humanoid/action", String, ros_callback)

    vision = VisionSystem(args.camera, robot)

    signal.signal(signal.SIGINT, lambda s, f: rospy.signal_shutdown("exit"))

    vision.run()

if __name__ == "__main__":
    main()
