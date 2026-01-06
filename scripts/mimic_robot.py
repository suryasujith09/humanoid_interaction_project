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
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# ===================== CONFIG =====================
CAMERA_PATH = "/dev/usb_cam"
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "~/humanoid_interaction_project/actions"
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_TIME = 2.0  # Seconds between same action triggers

# ===================== POSE SIGNATURES =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: callable

def check_hands_up(lm):
    """Both hands raised above head"""
    return (lm[15].y < lm[0].y - 0.1 and  # Left wrist above nose
            lm[16].y < lm[0].y - 0.1 and  # Right wrist above nose
            abs(lm[15].x - lm[16].x) < 0.4)  # Hands not too far apart

def check_hands_straight(lm):
    """Arms extended sideways (T-pose)"""
    shoulder_y = (lm[11].y + lm[12].y) / 2
    return (abs(lm[15].y - shoulder_y) < 0.15 and  # Left wrist at shoulder height
            abs(lm[16].y - shoulder_y) < 0.15 and  # Right wrist at shoulder height
            lm[15].x < lm[11].x - 0.2 and  # Left hand extended left
            lm[16].x > lm[12].x + 0.2)     # Right hand extended right

def check_wave(lm):
    """Right hand raised and moving"""
    return (lm[16].y < lm[12].y - 0.2 and  # Right wrist above shoulder
            lm[16].x > lm[0].x)             # Hand to the right of nose

def check_hands_down(lm):
    """Both hands at sides"""
    hip_y = (lm[23].y + lm[24].y) / 2
    return (lm[15].y > hip_y - 0.1 and  # Left wrist near hip
            lm[16].y > hip_y - 0.1)     # Right wrist near hip

def check_place_block(lm):
    """Hands forward at chest level (placing motion)"""
    shoulder_y = (lm[11].y + lm[12].y) / 2
    return (abs(lm[15].y - shoulder_y) < 0.2 and  # Left wrist at chest
            abs(lm[16].y - shoulder_y) < 0.2 and  # Right wrist at chest
            lm[15].z < -0.1 and lm[16].z < -0.1)  # Hands forward

# Register all poses
POSES = [
    PoseSignature("hands_up", "hands_up.d6a", check_hands_up),
    PoseSignature("hands_straight", "hands_straight.d6a", check_hands_straight),
    PoseSignature("wave", "wave.d6a", check_wave),
    PoseSignature("place_block", "placeblock.d6a", check_place_block),
    PoseSignature("hands_down", "greet.d6a", check_hands_down),
]

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port: str, baud: int):
        try:
            self.serial = serial.Serial(port, baud, timeout=0.1)
            time.sleep(0.5)
            print(f"✅ Serial connected: {port}")
        except Exception as e:
            print(f"⚠️ Serial unavailable: {e}")
            self.serial = None
    
    def play_action(self, action_file: str) -> bool:
        """Load and execute action from .d6a file"""
        if not self.serial:
            print(f"🤖 [DEMO] Playing: {action_file}")
            return True
        
        try:
            db_path = Path(ACTIONS_DIR).expanduser() / action_file
            if not db_path.exists():
                print(f"❌ Action file not found: {action_file}")
                return False
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            frames = cursor.execute("SELECT * FROM frames ORDER BY frame_num").fetchall()
            conn.close()
            
            for frame in frames:
                positions = frame[1:25]  # 24 servo positions
                cmd = "#000P{:04d}".format(positions[0])
                for pos in positions[1:]:
                    cmd += "P{:04d}".format(pos)
                cmd += "T0100\r\n"
                self.serial.write(cmd.encode())
                time.sleep(0.02)
            
            print(f"✅ Executed: {action_file}")
            return True
        except Exception as e:
            print(f"❌ Action failed: {e}")
            return False

# ===================== POSE DETECTOR =====================
class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0  # Fastest model
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect(self, frame):
        """Returns landmarks if person detected"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results.pose_landmarks.landmark if results.pose_landmarks else None
    
    def draw(self, frame, landmarks):
        """Draw skeleton on frame"""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                type('obj', (), {'landmark': landmarks})(),
                self.mp_pose.POSE_CONNECTIONS
            )

# ===================== MAIN MIMIC ENGINE =====================
class MimicEngine:
    def __init__(self, camera_path: str):
        self.detector = PoseDetector()
        self.robot = RobotController(SERIAL_PORT, BAUD_RATE)
        self.cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.last_action = None
        self.last_trigger_time = 0
        
        print("🚀 Mimic Engine Ready")
    
    def match_pose(self, landmarks) -> Optional[str]:
        """Find matching action for current pose"""
        for pose in POSES:
            try:
                if pose.check_func(landmarks):
                    return pose.action_file
            except Exception:
                continue
        return None
    
    def run(self):
        """Main loop: Capture → Detect → Match → Execute"""
        print("👁️ Watching for poses... (Press 'q' to quit)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect human skeleton
            landmarks = self.detector.detect(frame)
            
            if landmarks:
                # Match pose to action
                action = self.match_pose(landmarks)
                current_time = time.time()
                
                # Trigger robot if new pose detected
                if action and action != self.last_action:
                    if current_time - self.last_trigger_time > COOLDOWN_TIME:
                        print(f"🎯 Detected: {action}")
                        self.robot.play_action(action)
                        self.last_action = action
                        self.last_trigger_time = current_time
                
                # Visual feedback
                self.detector.draw(frame, landmarks)
                status = f"Action: {action or 'None'}"
                cv2.putText(frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Humanoid Mimic", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.robot.serial:
            self.robot.serial.close()
        print("👋 Shutdown complete")

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Humanoid Pose Mimic System")
    parser.add_argument("--camera", default=CAMERA_PATH, help="Camera device path")
    args = parser.parse_args()
    
    engine = MimicEngine(args.camera)
    engine.run()
