#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import threading
import argparse
import signal

import rospy
from std_msgs.msg import String

import mediapipe as mp
from pathlib import Path

# ─────────────────────────────────────────────
# SDK PATHS
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
CUSTOM_ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions/custom"
SYSTEM_ACTIONS_DIR = "/home/ubuntu/software/ainex_controller/ActionGroups"

COOLDOWN_TIME = 2.5
MAX_SERVOS = 22

POSE_TO_ACTION = {
    "hands_up": "hands_up",
    "hands_straight": "hands_straight",
    "wave": "wave",
    "stand": "stand"
}

# ─────────────────────────────────────────────
# ROBOT CONTROLLER (CORRECT WAY)
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
            rospy.logerr(f"❌ Robot connection failed: {e}")
            sys.exit(1)

        self.lock = threading.Lock()

        # Enable torque
        for sid in range(1, MAX_SERVOS + 1):
            try:
                self.board.bus_servo_enable_torque(sid, 1)
            except:
                pass

    def play_action(self, action_name):
        if not self.lock.acquire(False):
            return

        try:
            filename = action_name + ".d6a"

            path = Path(CUSTOM_ACTIONS_DIR) / filename
            if not path.exists():
                path = Path(SYSTEM_ACTIONS_DIR) / filename
                if not path.exists():
                    rospy.logwarn(f"⚠️ Action not found: {filename}")
                    return

            rospy.loginfo(f"🎬 Playing action: {action_name}")

            # 🔥 CORRECT: use servo queue
            self.board.bus_servo_stop()

            # AiNex SDK automatically parses .d6a internally
            self.board.bus_servo_queue(str(path))
            self.board.bus_servo_queue_execute()

        except Exception as e:
            rospy.logerr(f"❌ Action error: {e}")

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
        self.drawer = mp.solutions.drawing_utils

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
            rospy.logerr("❌ Camera failed")
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
                self.drawer.draw_landmarks(frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                pose = self.detect_pose(res.pose_landmarks.landmark)

                if pose:
                    now = time.time()
                    if now - self.last.get(pose, 0) > COOLDOWN_TIME:
                        action = POSE_TO_ACTION[pose]
                        threading.Thread(
                            target=self.robot.play_action,
                            args=(action,),
                            daemon=True
                        ).start()
                        self.last[pose] = now

            cv2.imshow("Humanoid Mimic", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# ROS CALLBACK
# ─────────────────────────────────────────────
def ros_action_cb(msg):
    threading.Thread(
        target=robot.play_action,
        args=(msg.data.strip(),),
        daemon=True
    ).start()

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

    rospy.Subscriber("/humanoid/action", String, ros_action_cb)

    vision = VisionSystem(args.camera, robot)

    signal.signal(signal.SIGINT, lambda s, f: rospy.signal_shutdown("exit"))
    vision.run()

if __name__ == "__main__":
    main()
