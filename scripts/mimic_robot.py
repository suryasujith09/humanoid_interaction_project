#!/usr/bin/env python3
"""
🔥 Humanoid Mimic System - WORKING VERSION
Direct serial control - proven to work!
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
CAMERA_INDEX = 0
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 4.0
SERVO_COUNT = 24
FRAME_TIME = 100

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM - WORKING VERSION")
print("=" * 60)

# ===================== MEDIAPIPE =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ===================== POSE DEFINITIONS =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable
    description: str
    color: tuple

def check_hands_up(lm):
    """Both hands raised above head"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    nose = lm[0]
    return (left_wrist.y < nose.y - 0.1 and 
            right_wrist.y < nose.y - 0.1)

def check_t_pose(lm):
    """Arms extended horizontally"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    left_shoulder = lm[11]
    right_shoulder = lm[12]
    
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    left_straight = abs(left_wrist.y - shoulder_y) < 0.12
    right_straight = abs(right_wrist.y - shoulder_y) < 0.12
    arms_spread = abs(left_wrist.x - right_wrist.x) > 0.5
    
    return left_straight and right_straight and arms_spread

def check_wave(lm):
    """Right hand raised for waving"""
    right_wrist = lm[16]
    right_shoulder = lm[12]
    return right_wrist.y < right_shoulder.y - 0.15

def check_hands_down(lm):
    """Hands down by hips (greeting pose)"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    left_hip = lm[23]
    right_hip = lm[24]
    
    hip_y = (left_hip.y + right_hip.y) / 2
    return (left_wrist.y > hip_y - 0.15 and 
            right_wrist.y > hip_y - 0.15)

def check_place_block(lm):
    """Hands forward in placing motion"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    
    hands_forward = left_wrist.z < -0.1 and right_wrist.z < -0.1
    hands_close = abs(left_wrist.x - right_wrist.x) < 0.25
    
    return hands_forward and hands_close

# ===================== POSE REGISTRY =====================
POSES = [
    PoseSignature(
        name="HANDS UP",
        action_file="hands_up.d6a",
        check_func=check_hands_up,
        description="Raise both hands above head",
        color=(0, 255, 255)
    ),
    PoseSignature(
        name="T-POSE",
        action_file="hands_straight.d6a",
        check_func=check_t_pose,
        description="Extend arms horizontally",
        color=(255, 0, 255)
    ),
    PoseSignature(
        name="WAVE",
        action_file="wave.d6a",
        check_func=check_wave,
        description="Raise right hand to wave",
        color=(255, 165, 0)
    ),
    PoseSignature(
        name="GREET",
        action_file="greet.d6a",
        check_func=check_hands_down,
        description="Lower hands to greet",
        color=(0, 255, 0)
    ),
    PoseSignature(
        name="PLACE BLOCK",
        action_file="placeblock.d6a",
        check_func=check_place_block,
        description="Reach forward to place",
        color=(255, 100, 100)
    ),
]

print(f"\n📋 Loaded {len(POSES)} pose signatures:")
for p in POSES:
    action_path = Path(ACTIONS_DIR) / p.action_file
    status = "✅" if action_path.exists() else "❌"
    print(f"   {status} {p.name}: {p.description}")

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port, baud):
        self.busy = False
        self.serial = None
        
        try:
            print(f"\n🔌 Connecting to {port} @ {baud} baud...")
            self.serial = serial.Serial(
                port, baud,
                timeout=0.5,
                write_timeout=0.5
            )
            time.sleep(2)
            print("✅ Robot connected")
        except Exception as e:
            print(f"⚠️ Serial failed: {e}")
            print("📝 Running in DEMO mode")
    
    def send_frame(self, positions):
        """Send servo positions for one frame"""
        if not self.serial:
            return
        
        cmd = ""
        for i, pos in enumerate(positions):
            pos = max(500, min(2500, int(pos)))
            cmd += f"#{i:02d}P{pos:04d}"
        
        cmd += f"T{FRAME_TIME}\r\n"
        
        try:
            self.serial.write(cmd.encode('ascii'))
            self.serial.flush()
        except Exception as e:
            print(f"⚠️ Serial write error: {e}")
    
    def play_action(self, action_file, pose_name):
        """Execute an action from .d6a file"""
        if self.busy:
            return False
        
        path = Path(ACTIONS_DIR) / action_file
        if not path.exists():
            print(f"❌ File not found: {action_file}")
            return False
        
        print("\n" + "=" * 50)
        print(f"🎬 EXECUTING: {pose_name}")
        print(f"📂 File: {action_file}")
        
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            
            # Get table structure
            tables = [t[0] for t in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            
            # Find the right table
            table = None
            if "ActionGroup" in tables:
                table = "ActionGroup"
            elif "frames" in tables:
                table = "frames"
            else:
                table = tables[0] if tables else None
            
            if not table:
                print("⚠️ No valid table found")
                conn.close()
                return False
            
            # Read frames
            frames = cur.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            conn.close()
            
            if not frames:
                print("⚠️ No frames found")
                return False
            
            print(f"🎬 Playing {len(frames)} frames")
            
            if not self.serial:
                print("🎭 DEMO MODE - simulating")
                time.sleep(len(frames) * FRAME_TIME / 1000)
                return True
            
            self.busy = True
            
            # Determine start column
            start_col = 1
            if len(frames[0]) > SERVO_COUNT + 5:
                start_col = 2
            
            # Play each frame
            for frame in frames:
                end_col = start_col + SERVO_COUNT
                positions = frame[start_col:end_col]
                
                # Fill missing data
                if len(positions) < SERVO_COUNT:
                    positions = list(positions) + [1500] * (SERVO_COUNT - len(positions))
                
                self.send_frame(positions[:SERVO_COUNT])
                time.sleep(FRAME_TIME / 1000)
            
            print("✅ Action completed")
            self.busy = False
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.busy = False
            return False

# ===================== UI DRAWING =====================
def draw_ui_overlay(frame, detected_pose=None, fps=0, robot_status="Ready"):
    """Draw UI overlay"""
    h, w = frame.shape[:2]
    
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Title
    cv2.putText(frame, "HUMANOID MIMIC SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status
    status_color = (0, 255, 0) if robot_status == "Ready" else (0, 165, 255)
    cv2.putText(frame, f"Status: {robot_status}", (w - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Current pose badge
    if detected_pose:
        pose_obj = next((p for p in POSES if p.name == detected_pose), None)
        if pose_obj:
            badge_y = 100
            badge_h = 60
            cv2.rectangle(frame, (10, badge_y), (w - 10, badge_y + badge_h),
                         pose_obj.color, -1)
            cv2.rectangle(frame, (10, badge_y), (w - 10, badge_y + badge_h),
                         (255, 255, 255), 3)
            
            cv2.putText(frame, f">>> {detected_pose} <<<", (25, badge_y + 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Pose list panel
    panel_x = w - 320
    panel_y = 180
    panel_w = 310
    panel_h = len(POSES) * 50 + 50
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h),
                 (20, 20, 20), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + panel_w, panel_y + panel_h),
                 (100, 100, 100), 2)
    
    # Panel title
    cv2.putText(frame, "Available Poses", (panel_x + 10, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Pose list
    y_pos = panel_y + 60
    for pose in POSES:
        is_active = detected_pose == pose.name
        
        # Color indicator
        cv2.circle(frame, (panel_x + 15, y_pos - 5), 7, pose.color, -1)
        cv2.circle(frame, (panel_x + 15, y_pos - 5), 7, (255, 255, 255), 1)
        
        # Pose name
        color = (255, 255, 255) if is_active else (180, 180, 180)
        weight = 2 if is_active else 1
        cv2.putText(frame, pose.name, (panel_x + 30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
        
        y_pos += 50
    
    # Instructions
    cv2.putText(frame, "Press ESC or Q to exit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

# ===================== INITIALIZE =====================
print("\n🔧 Initializing system...")

robot = RobotController(SERIAL_PORT, BAUD_RATE)

print("🎥 Starting camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Camera failed")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera ready")
print("\n" + "=" * 60)
print("🟢 SYSTEM ACTIVE - Strike a pose!")
print("=" * 60)

# ===================== MAIN LOOP =====================
last_trigger = {}
frame_count = 0
start_time = time.time()
detected_pose_name = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        
        detected_pose_name = None
        robot_status = "Busy" if robot.busy else "Ready"
        
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            
            # Check poses (only if robot ready)
            if not robot.busy:
                for pose_sig in POSES:
                    try:
                        if pose_sig.check_func(lm):
                            detected_pose_name = pose_sig.name
                            now = time.time()
                            
                            # Check cooldown
                            last_time = last_trigger.get(pose_sig.name, 0)
                            if now - last_time > COOLDOWN_TIME:
                                # Execute action
                                if robot.play_action(pose_sig.action_file, pose_sig.name):
                                    last_trigger[pose_sig.name] = now
                            
                            break
                    except Exception as e:
                        print(f"⚠️ Error checking {pose_sig.name}: {e}")
            
            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 255), thickness=2
                )
            )
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw UI
        frame = draw_ui_overlay(frame, detected_pose_name, fps, robot_status)
        
        # Display
        cv2.imshow("Humanoid Mimic System", frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")
except Exception as e:
    print(f"\n\n❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n🛑 Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    if robot.serial:
        robot.serial.close()
    print("✅ Shutdown complete")
