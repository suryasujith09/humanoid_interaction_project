#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import cv2
import sqlite3
import threading
import argparse
import signal
from pathlib import Path

import rospy
from std_msgs.msg import String
import mediapipe as mp

# ─────────────────────────────────────────────
# SDK PATHS
# ─────────────────────────────────────────────
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

from ainex_sdk import Board

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CUSTOM_ACTIONS = "/home/ubuntu/humanoid_interaction_project/actions/custom"
SYSTEM_ACTIONS = "/home/ubuntu/software/ainex_controller/ActionGroups"

POSE_TO_ACTION = {
    "hands_up": "hands_up",
    "hands_straight": "hands_straight",
    "wave": "wave",
    "stand": "stand"
}

COOLDOWN = 2.5

# ─────────────────────────────────────────────
# ROBOT CONTROLLER (FINAL)
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
            rospy.logerr(f"Robot init failed: {e}")
            sys.exit(1)

        self.lock = threading.Lock()

        # Enable torque safely
        for i in range(1, 23):
            try:
                self.board.bus_servo_enable_torque(i, 1)
            except:
                pass

    def play_action(self, action_name):
        if not self.lock.acquire(False):
            return

        try:
            filename = action_name + ".d6a"

            path = Path(CUSTOM_ACTIONS) / filename
            if not path.exists():
                path = Path(SYSTEM_ACTIONS) / filename
                if not path.exists():
                    rospy.logwarn(f"Action not found: {filename}")
                    return

            rospy.loginfo(f"🎬 Playing action: {action_name}")

            conn = sqlite3.connect(str(path))
            cur = conn.cursor()

            # THIS TABLE IS SERVO-BY-SERVO (confirmed)
            rows = cur.execute(
                "SELECT * FROM frames ORDER BY [Index]"
            ).fetchall()

            conn.close()

            for row in rows:
                # Expected format:
                # [Index, Time(ms), ServoID, Position]
                if len(row) != 4:
                    continue

                _, duration, servo_id, position = row

                self.board.bus_servo_set_position(
                    int(servo_id),
                    int(position)
                )

                time.sleep(duration / 1000.0)

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
        cap = cv2.VideoCapture(int(self.camera))
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
                self.draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                pose = self.detect_pose(res.pose_landmarks.landmark)
                if pose:
                    now = time.time()
                    if now - self.last.get(pose, 0) > COOLDOWN:
                        threading.Thread(
                            target=self.robot.play_action,
                            args=(POSE_TO_ACTION[pose],),
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
def ros_cb(msg):
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

    rospy.Subscriber("/humanoid/action", String, ros_cb)

    VisionSystem(args.camera, robot).run()

if __name__ == "__main__":
    main()
