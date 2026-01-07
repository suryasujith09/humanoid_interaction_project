#!/usr/bin/env python3
"""
🔥 Humanoid Mimic System - FINAL VERSION
Uses proper MotionManager API (Hiwonder/AiNex protocol)
"""
import cv2
import mediapipe as mp
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# Import the motion manager (Hiwonder/AiNex API)
try:
    from ainex_kinematics.motion_manager import MotionManager
    motion_manager = MotionManager('/home/ubuntu/software/ainex_controller/ActionGroups')
    MOTION_AVAILABLE = True
    print("✅ MotionManager imported successfully")
except ImportError as e:
    print(f"⚠️ MotionManager import failed: {e}")
    print("📝 Running in DEMO mode")
    motion_manager = None
    MOTION_AVAILABLE = False

# ===================== CONFIG =====================
CAMERA_INDEX = 0
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 4.0

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM - FINAL")
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
    action_name: str  # Name for motion_manager.runAction()
    check_func: Callable
    description: str
    color: tuple

def check_hands_up(lm):
    """Both hands raised above head"""
    return lm[15].y < lm[0].y - 0.1 and lm[16].y < lm[0].y - 0.1

def check_t_pose(lm):
    """Arms extended horizontally"""
    sy = (lm[11].y + lm[12].y) / 2
    return (abs(lm[15].y - sy) < 0.12 and 
            abs(lm[16].y - sy) < 0.12 and
            abs(lm[15].x - lm[16].x) > 0.5)

def check_wave(lm):
    """Right hand raised for waving"""
    return lm[16].y < lm[12].y - 0.15

def check_hands_down(lm):
    """Hands down by hips"""
    hy = (lm[23].y + lm[24].y) / 2
    return lm[15].y > hy - 0.15 and lm[16].y > hy - 0.15

def check_place_block(lm):
    """Hands forward in placing motion"""
    return (lm[15].z < -0.1 and lm[16].z < -0.1 and
            abs(lm[15].x - lm[16].x) < 0.25)

# ===================== POSE REGISTRY =====================
# Map to action names that MotionManager recognizes
POSES = [
    PoseSignature(
        name="HANDS UP",
        action_name="hands_up",  # This calls motion_manager.runAction('hands_up')
        check_func=check_hands_up,
        description="Raise both hands above head",
        color=(0, 255, 255)
    ),
    PoseSignature(
        name="T-POSE",
        action_name="hands_straight",
        check_func=check_t_pose,
        description="Extend arms horizontally",
        color=(255, 0, 255)
    ),
    PoseSignature(
        name="WAVE",
        action_name="wave",
        check_func=check_wave,
        description="Raise right hand to wave",
        color=(255, 165, 0)
    ),
    PoseSignature(
        name="GREET",
        action_name="greet",
        check_func=check_hands_down,
        description="Lower hands to greet",
        color=(0, 255, 0)
    ),
    PoseSignature(
        name="PLACE BLOCK",
        action_name="placeblock",
        check_func=check_place_block,
        description="Reach forward to place",
        color=(255, 100, 100)
    ),
]

print(f"\n📋 Loaded {len(POSES)} pose signatures:")
for p in POSES:
    print(f"   • {p.name} → motion_manager.runAction('{p.action_name}')")

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self):
        self.busy = False
        self.available = MOTION_AVAILABLE
        
        if self.available:
            print("\n✅ Robot controller ready (MotionManager)")
        else:
            print("\n⚠️ Robot controller in DEMO mode")
    
    def execute_action(self, action_name, pose_name):
        """Execute action using MotionManager"""
        if self.busy:
            print(f"⏳ Robot busy, skipping {pose_name}")
            return False
        
        print("\n" + "=" * 50)
        print(f"🎬 EXECUTING: {pose_name}")
        print(f"📞 Calling: motion_manager.runAction('{action_name}')")
        
        if not self.available:
            print("🎭 DEMO MODE - No actual robot motion")
            time.sleep(1.5)
            return True
        
        self.busy = True
        success = False
        
        try:
            # This is the key call - exactly like in your example!
            motion_manager.runAction(action_name)
            print("✅ Action completed successfully")
            success = True
        except Exception as e:
            print(f"❌ Action failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            time.sleep(0.5)
            self.busy = False
        
        return success

# ===================== UI DRAWING =====================
def draw_ui_overlay(frame, detected_pose=None, fps=0, robot_status="Ready"):
    """Draw beautiful UI overlay"""
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
print("\n🔧 Initializing components...")

robot = RobotController()

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
                                # Execute action using MotionManager
                                if robot.execute_action(pose_sig.action_name, pose_sig.name):
                                    last_trigger[pose_sig.name] = now
                            
                            break  # Only one pose at a time
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
        if key == 27 or key == ord('q'):  # ESC or Q
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
    pose.close()
    print("✅ Shutdown complete")
    print("\nThank you for using Humanoid Mimic System!")
