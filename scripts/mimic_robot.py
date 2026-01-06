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
from typing import Optional

# ===================== CONFIG =====================
CAMERA_PATH = "/dev/usb_cam"
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "~/humanoid_interaction_project/actions"
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_TIME = 2.0  # Seconds between same action triggers

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM STARTING")
print("=" * 60)

# ===================== POSE SIGNATURES =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: callable

def check_hands_up(lm):
    """Both hands raised above head"""
    left_up = lm[15].y < lm[0].y - 0.1  # Left wrist above nose
    right_up = lm[16].y < lm[0].y - 0.1  # Right wrist above nose
    close_together = abs(lm[15].x - lm[16].x) < 0.4
    return left_up and right_up and close_together

def check_hands_straight(lm):
    """Arms extended sideways (T-pose)"""
    shoulder_y = (lm[11].y + lm[12].y) / 2
    left_level = abs(lm[15].y - shoulder_y) < 0.15
    right_level = abs(lm[16].y - shoulder_y) < 0.15
    left_extended = lm[15].x < lm[11].x - 0.2
    right_extended = lm[16].x > lm[12].x + 0.2
    return left_level and right_level and left_extended and right_extended

def check_wave(lm):
    """Right hand raised and moving"""
    hand_up = lm[16].y < lm[12].y - 0.2  # Right wrist above shoulder
    hand_side = lm[16].x > lm[0].x  # Hand to the right
    return hand_up and hand_side

def check_hands_down(lm):
    """Both hands at sides"""
    hip_y = (lm[23].y + lm[24].y) / 2
    left_down = lm[15].y > hip_y - 0.1
    right_down = lm[16].y > hip_y - 0.1
    return left_down and right_down

def check_place_block(lm):
    """Hands forward at chest level (placing motion)"""
    shoulder_y = (lm[11].y + lm[12].y) / 2
    left_chest = abs(lm[15].y - shoulder_y) < 0.2
    right_chest = abs(lm[16].y - shoulder_y) < 0.2
    hands_forward = lm[15].z < -0.1 and lm[16].z < -0.1
    return left_chest and right_chest and hands_forward

# Register all poses
POSES = [
    PoseSignature("HANDS UP", "hands_up.d6a", check_hands_up),
    PoseSignature("T-POSE", "hands_straight.d6a", check_hands_straight),
    PoseSignature("WAVE", "wave.d6a", check_wave),
    PoseSignature("PLACE BLOCK", "placeblock.d6a", check_place_block),
    PoseSignature("HANDS DOWN", "greet.d6a", check_hands_down),
]

print(f"📋 Loaded {len(POSES)} pose signatures:")
for p in POSES:
    print(f"   - {p.name} → {p.action_file}")

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port: str, baud: int):
        print(f"\n🔌 Connecting to robot at {port} ({baud} baud)...")
        try:
            self.serial = serial.Serial(port, baud, timeout=0.1)
            time.sleep(0.5)
            print(f"✅ Serial connected successfully!")
        except Exception as e:
            print(f"⚠️  Serial unavailable: {e}")
            print(f"⚠️  Running in DEMO mode (no robot commands sent)")
            self.serial = None
    
    def play_action(self, action_file: str, pose_name: str) -> bool:
        """Load and execute action from .d6a file"""
        print(f"\n{'='*50}")
        print(f"🎬 TRIGGERING: {pose_name}")
        print(f"📂 Loading: {action_file}")
        
        if not self.serial:
            print(f"🤖 [DEMO MODE] Simulating action playback...")
            time.sleep(0.5)
            print(f"✅ Action complete (demo)")
            return True
        
        try:
            db_path = Path(ACTIONS_DIR).expanduser() / action_file
            if not db_path.exists():
                print(f"❌ ERROR: Action file not found at {db_path}")
                return False
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            frames = cursor.execute("SELECT * FROM frames ORDER BY frame_num").fetchall()
            conn.close()
            
            print(f"📊 Executing {len(frames)} frames...")
            for i, frame in enumerate(frames):
                positions = frame[1:25]  # 24 servo positions
                cmd = "#000P{:04d}".format(positions[0])
                for pos in positions[1:]:
                    cmd += "P{:04d}".format(pos)
                cmd += "T0100\r\n"
                self.serial.write(cmd.encode())
                time.sleep(0.02)
                
                if i % 10 == 0:
                    print(f"   Frame {i+1}/{len(frames)}", end='\r')
            
            print(f"\n✅ Action complete!                    ")
            return True
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return False

# ===================== POSE DETECTOR =====================
class PoseDetector:
    def __init__(self):
        print(f"\n🧠 Initializing MediaPipe Pose detector...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0  # Fastest model
        )
        self.mp_draw = mp.solutions.drawing_utils
        print(f"✅ Pose detector ready!")
    
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
        print(f"\n📹 Opening camera: {camera_path}")
        
        # Try multiple backends
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            print(f"   Trying backend: {backend}...")
            self.cap = cv2.VideoCapture(camera_path, backend)
            if self.cap.isOpened():
                print(f"   ✅ Camera opened with backend {backend}")
                break
        else:
            print(f"❌ FATAL: Cannot open camera at {camera_path}")
            print(f"   Trying /dev/video0 as fallback...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print(f"❌ FATAL: No camera available!")
                sys.exit(1)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"📐 Camera resolution: {width}x{height} @ {fps}fps")
        
        self.detector = PoseDetector()
        self.robot = RobotController(SERIAL_PORT, BAUD_RATE)
        
        self.last_action = None
        self.last_trigger_time = 0
        self.frame_count = 0
        self.detection_count = 0
        
        print("\n" + "="*60)
        print("🚀 MIMIC ENGINE READY!")
        print("="*60)
    
    def match_pose(self, landmarks) -> Optional[tuple]:
        """Find matching action for current pose"""
        for pose in POSES:
            try:
                if pose.check_func(landmarks):
                    return (pose.name, pose.action_file)
            except Exception as e:
                continue
        return None
    
    def run(self):
        """Main loop: Capture → Detect → Match → Execute"""
        print("\n👁️  WATCHING FOR POSES...")
        print("   • Stand in front of camera")
        print("   • Try: Hands Up, T-Pose, Wave")
        print("   • Press 'q' to quit\n")
        
        last_status_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️  WARNING: Failed to read frame!")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # Detect human skeleton
            landmarks = self.detector.detect(frame)
            
            # Status text color
            status_color = (0, 255, 0)  # Green
            status_text = "No person detected"
            
            if landmarks:
                self.detection_count += 1
                
                # Match pose to action
                match = self.match_pose(landmarks)
                current_time = time.time()
                
                if match:
                    pose_name, action_file = match
                    status_text = f"DETECTED: {pose_name}"
                    status_color = (0, 255, 255)  # Yellow
                    
                    # Trigger robot if new pose detected
                    if action_file != self.last_action:
                        if current_time - self.last_trigger_time > COOLDOWN_TIME:
                            self.robot.play_action(action_file, pose_name)
                            self.last_action = action_file
                            self.last_trigger_time = current_time
                    else:
                        cooldown_remaining = COOLDOWN_TIME - (current_time - self.last_trigger_time)
                        if cooldown_remaining > 0:
                            status_text += f" [Cooldown: {cooldown_remaining:.1f}s]"
                else:
                    status_text = "Person detected - Make a pose!"
                    status_color = (255, 255, 0)  # Cyan
                
                # Draw skeleton
                self.detector.draw(frame, landmarks)
            
            # Add status overlay
            cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # FPS counter
            if self.frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - last_status_time)
                last_status_time = current_time
                print(f"📊 FPS: {fps:.1f} | Frames: {self.frame_count} | Detections: {self.detection_count}")
            
            # Show frame
            cv2.imshow("🤖 Humanoid Mimic System", frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Quit signal received...")
                break
        
        self.cleanup()
    
    def cleanup(self):
        print("\n🧹 Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.robot.serial:
            self.robot.serial.close()
        print("👋 Shutdown complete")
        print(f"📊 Final stats: {self.frame_count} frames, {self.detection_count} detections")

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Humanoid Pose Mimic System")
    parser.add_argument("--camera", default=CAMERA_PATH, help="Camera device path")
    args = parser.parse_args()
    
    try:
        engine = MimicEngine(args.camera)
        engine.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("Program terminated")
        print("="*60)
