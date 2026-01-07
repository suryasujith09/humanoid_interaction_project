#!/usr/bin/env python3
"""
🔥 Humanoid Mimic System - Integrated Version
Uses your existing action_controller infrastructure
"""
import cv2
import mediapipe as mp
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# Add your project to path
sys.path.insert(0, '/home/ubuntu/humanoid_interaction_project/scripts')

try:
    from controllers.action_controller import ActionController
    CONTROLLER_AVAILABLE = True
    print("✅ ActionController imported successfully")
except ImportError as e:
    print(f"⚠️ ActionController not found: {e}")
    print("📝 Running in DEMO mode (visualization only)")
    CONTROLLER_AVAILABLE = False

# ===================== CONFIG =====================
CAMERA_INDEX = 0
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 4.0
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM v2.0")
print("=" * 60)

# ===================== MEDIAPIPE SETUP =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# ===================== POSE SIGNATURES =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable
    description: str
    color: tuple  # BGR color for UI

def check_hands_up(lm):
    """Both hands raised above head"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    nose = lm[0]
    
    # Both wrists above nose
    return (left_wrist.y < nose.y - 0.1 and 
            right_wrist.y < nose.y - 0.1)

def check_t_pose(lm):
    """Arms extended horizontally"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    left_shoulder = lm[11]
    right_shoulder = lm[12]
    
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Wrists at shoulder height and arms spread wide
    left_straight = abs(left_wrist.y - shoulder_y) < 0.12
    right_straight = abs(right_wrist.y - shoulder_y) < 0.12
    arms_spread = abs(left_wrist.x - right_wrist.x) > 0.5
    
    return left_straight and right_straight and arms_spread

def check_wave(lm):
    """Right hand raised for waving"""
    right_wrist = lm[16]
    right_shoulder = lm[12]
    
    # Right wrist above shoulder
    return right_wrist.y < right_shoulder.y - 0.15

def check_hands_down(lm):
    """Hands down by hips (greeting pose)"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    left_hip = lm[23]
    right_hip = lm[24]
    
    hip_y = (left_hip.y + right_hip.y) / 2
    
    # Both hands at or below hip level
    return (left_wrist.y > hip_y - 0.15 and 
            right_wrist.y > hip_y - 0.15)

def check_place_block(lm):
    """Hands forward in placing motion"""
    left_wrist = lm[15]
    right_wrist = lm[16]
    
    # Both hands reaching forward (negative z = toward camera)
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
        color=(0, 255, 255)  # Yellow
    ),
    PoseSignature(
        name="T-POSE",
        action_file="hands_straight.d6a",
        check_func=check_t_pose,
        description="Extend arms horizontally",
        color=(255, 0, 255)  # Magenta
    ),
    PoseSignature(
        name="WAVE",
        action_file="wave.d6a",
        check_func=check_wave,
        description="Raise right hand to wave",
        color=(255, 165, 0)  # Orange
    ),
    PoseSignature(
        name="GREET",
        action_file="greet.d6a",
        check_func=check_hands_down,
        description="Lower hands to greet",
        color=(0, 255, 0)  # Green
    ),
    PoseSignature(
        name="PLACE BLOCK",
        action_file="placeblock.d6a",
        check_func=check_place_block,
        description="Reach forward to place",
        color=(255, 100, 100)  # Light Blue
    ),
]

print(f"\n📋 Loaded {len(POSES)} pose signatures:")
for p in POSES:
    action_path = Path(ACTIONS_DIR) / p.action_file
    status = "✅" if action_path.exists() else "❌"
    print(f"   {status} {p.name}: {p.description}")

# ===================== ROBOT CONTROLLER =====================
class RobotMimicController:
    def __init__(self, actions_dir):
        self.actions_dir = Path(actions_dir)
        self.busy = False
        self.last_action_time = 0
        
        if CONTROLLER_AVAILABLE:
            try:
                self.controller = ActionController(str(self.actions_dir))
                print("\n✅ Robot controller initialized")
                self.available = True
            except Exception as e:
                print(f"\n⚠️ Controller init failed: {e}")
                self.controller = None
                self.available = False
        else:
            self.controller = None
            self.available = False
    
    def execute_pose(self, pose: PoseSignature):
        """Execute a pose action on the robot"""
        if self.busy:
            return False, "Robot busy"
        
        action_path = self.actions_dir / pose.action_file
        if not action_path.exists():
            return False, f"Action file not found: {pose.action_file}"
        
        print("\n" + "=" * 50)
        print(f"🎬 EXECUTING: {pose.name}")
        print(f"📂 File: {pose.action_file}")
        
        if not self.available:
            print("🎭 DEMO MODE - Simulating action")
            time.sleep(1.5)
            return True, "Demo mode"
        
        self.busy = True
        success = False
        message = ""
        
        try:
            # Use your ActionController to play the action
            self.controller.play_action(str(action_path))
            print("✅ Action completed")
            success = True
            message = "Success"
        except Exception as e:
            print(f"❌ Action failed: {e}")
            message = str(e)
        finally:
            time.sleep(0.3)  # Small delay
            self.busy = False
            self.last_action_time = time.time()
        
        return success, message

# ===================== UI DRAWING =====================
def draw_ui_overlay(frame, detected_pose=None, fps=0, robot_status="Ready"):
    """Draw beautiful UI overlay"""
    h, w = frame.shape[:2]
    
    # === TOP BAR ===
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Title
    cv2.putText(frame, "HUMANOID MIMIC SYSTEM", (20, 35),
                cv2.FONT_HERSHEY_BOLD, 1.0, (0, 255, 255), 2)
    
    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status
    status_color = (0, 255, 0) if robot_status == "Ready" else (0, 165, 255)
    cv2.putText(frame, f"Status: {robot_status}", (w - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # === CURRENT POSE (if detected) ===
    if detected_pose:
        # Find the pose object
        pose_obj = next((p for p in POSES if p.name == detected_pose), None)
        if pose_obj:
            # Big colored badge
            badge_y = 100
            badge_h = 60
            cv2.rectangle(frame, (10, badge_y), (w - 10, badge_y + badge_h),
                         pose_obj.color, -1)
            cv2.rectangle(frame, (10, badge_y), (w - 10, badge_y + badge_h),
                         (255, 255, 255), 2)
            
            # Text
            text = f"🎯 DETECTED: {detected_pose}"
            cv2.putText(frame, text, (20, badge_y + 40),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)
    
    # === POSE LIST (side panel) ===
    panel_x = w - 320
    panel_y = 180
    panel_w = 310
    panel_h = len(POSES) * 50 + 50
    
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h),
                 (20, 20, 20), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + panel_w, panel_y + panel_h),
                 (100, 100, 100), 2)
    
    # Title
    cv2.putText(frame, "Available Poses", (panel_x + 10, panel_y + 30),
                cv2.FONT_HERSHEY_BOLD, 0.6, (200, 200, 200), 2)
    
    # Pose list
    y_pos = panel_y + 60
    for pose in POSES:
        is_active = detected_pose == pose.name
        
        # Color indicator
        cv2.circle(frame, (panel_x + 15, y_pos - 5), 6, pose.color, -1)
        
        # Pose name
        color = (255, 255, 255) if is_active else (180, 180, 180)
        weight = 2 if is_active else 1
        cv2.putText(frame, pose.name, (panel_x + 30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
        
        y_pos += 50
    
    # === INSTRUCTIONS ===
    cv2.putText(frame, "Press ESC or Q to exit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

# ===================== INITIALIZE =====================
print("\n🔧 Initializing components...")

robot = RobotMimicController(ACTIONS_DIR)

print("🎥 Starting camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Failed to open camera")
    sys.exit(1)

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera ready")
print("\n" + "=" * 60)
print("🟢 SYSTEM ACTIVE - Strike a pose to control the robot!")
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
            
            # Check each pose (only if robot is ready)
            if not robot.busy:
                for pose_sig in POSES:
                    try:
                        if pose_sig.check_func(lm):
                            detected_pose_name = pose_sig.name
                            now = time.time()
                            
                            # Check cooldown
                            last_time = last_trigger.get(pose_sig.name, 0)
                            if now - last_time > COOLDOWN_TIME:
                                # Execute the action
                                success, msg = robot.execute_pose(pose_sig)
                                if success:
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
    print("\n\n⚠️ Interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\n\n❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    print("\n🛑 Shutting down system...")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("✅ Shutdown complete")
    print("\nThank you for using Humanoid Mimic System!")
