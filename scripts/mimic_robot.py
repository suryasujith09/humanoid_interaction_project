#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 Humanoid Mimic System - AiNex Integration
Based on ainex_example structure
"""
import sys
import cv2
import time
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# Add AiNex paths
sys.path.append('/home/ubuntu/ros_ws/src/ainex_driver/ainex_kinematics/src/')

# Import AiNex motion manager
try:
    from ainex_kinematics.motion_manager import MotionManager
    print("✅ MotionManager imported")
    MOTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Failed to import MotionManager: {e}")
    print("📝 Running in DEMO mode")
    MOTION_AVAILABLE = False

# ===================== CONFIG =====================
CAMERA_INDEX = 0
COOLDOWN_TIME = 4.0  # Seconds between same pose triggers
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

print("=" * 60)
print("🤖 HUMANOID MIMIC SYSTEM")
print("=" * 60)

# ===================== INITIALIZE MOTION MANAGER =====================
if MOTION_AVAILABLE:
    try:
        # Initialize MotionManager with ActionGroups directory
        motion_manager = MotionManager('/home/ubuntu/software/ainex_controller/ActionGroups/')
        print("✅ MotionManager initialized")
    except Exception as e:
        print(f"⚠️ MotionManager init failed: {e}")
        motion_manager = None
        MOTION_AVAILABLE = False
else:
    motion_manager = None

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

print("✅ MediaPipe initialized")

# ===================== POSE DEFINITIONS =====================
@dataclass
class PoseAction:
    """Defines a pose and its corresponding robot action"""
    name: str
    action_name: str
    check_func: Callable
    description: str
    color: tuple  # BGR color for visualization

def check_hands_up(landmarks):
    """Detect: Both hands raised above head"""
    left_wrist = landmarks[15]   # Left wrist
    right_wrist = landmarks[16]  # Right wrist
    nose = landmarks[0]          # Nose (head reference)
    
    # Both wrists should be above the nose
    return (left_wrist.y < nose.y - 0.1 and 
            right_wrist.y < nose.y - 0.1)

def check_t_pose(landmarks):
    """Detect: Arms extended horizontally (T-pose)"""
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    # Calculate average shoulder height
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Both wrists should be at shoulder height and arms spread wide
    left_level = abs(left_wrist.y - shoulder_y) < 0.12
    right_level = abs(right_wrist.y - shoulder_y) < 0.12
    arms_wide = abs(left_wrist.x - right_wrist.x) > 0.5
    
    return left_level and right_level and arms_wide

def check_wave(landmarks):
    """Detect: Right hand raised for waving"""
    right_wrist = landmarks[16]
    right_shoulder = landmarks[12]
    
    # Right wrist should be above shoulder
    return right_wrist.y < right_shoulder.y - 0.15

def check_greet(landmarks):
    """Detect: Hands down by hips (greeting pose)"""
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # Calculate average hip height
    hip_y = (left_hip.y + right_hip.y) / 2
    
    # Both hands should be at or below hip level
    return (left_wrist.y > hip_y - 0.15 and 
            right_wrist.y > hip_y - 0.15)

def check_place_block(landmarks):
    """Detect: Hands reaching forward (placing motion)"""
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    # Both hands should be reaching forward (toward camera)
    # and close together
    hands_forward = left_wrist.z < -0.1 and right_wrist.z < -0.1
    hands_together = abs(left_wrist.x - right_wrist.x) < 0.25
    
    return hands_forward and hands_together

# ===================== POSE REGISTRY =====================
# Define all poses that trigger robot actions
POSES = [
    PoseAction(
        name="HANDS UP",
        action_name="hands_up",
        check_func=check_hands_up,
        description="Raise both hands above your head",
        color=(0, 255, 255)  # Yellow
    ),
    PoseAction(
        name="T-POSE",
        action_name="hands_straight",
        check_func=check_t_pose,
        description="Extend arms horizontally to sides",
        color=(255, 0, 255)  # Magenta
    ),
    PoseAction(
        name="WAVE",
        action_name="wave",
        check_func=check_wave,
        description="Raise right hand to wave",
        color=(255, 165, 0)  # Orange
    ),
    PoseAction(
        name="GREET",
        action_name="greet",
        check_func=check_greet,
        description="Lower hands to greet",
        color=(0, 255, 0)  # Green
    ),
    PoseAction(
        name="PLACE BLOCK",
        action_name="placeblock",
        check_func=check_place_block,
        description="Reach forward with both hands",
        color=(100, 100, 255)  # Light red
    ),
]

print(f"\n📋 Loaded {len(POSES)} pose actions:")
for p in POSES:
    print(f"   • {p.name}: {p.description}")

# ===================== ROBOT CONTROLLER =====================
class HumanoidController:
    """Controls the humanoid robot using MotionManager"""
    
    def __init__(self):
        self.busy = False
        self.motion_manager = motion_manager
        self.available = MOTION_AVAILABLE
    
    def execute_action(self, action_name, pose_name):
        """
        Execute a robot action
        
        Args:
            action_name: Name of the action file (without .d6a)
            pose_name: Human-readable pose name for display
        
        Returns:
            bool: True if action executed successfully
        """
        if self.busy:
            print(f"⏳ Robot busy, skipping {pose_name}")
            return False
        
        print("\n" + "=" * 50)
        print(f"🎬 EXECUTING: {pose_name}")
        print(f"📞 Action: {action_name}")
        
        # Demo mode (no actual robot)
        if not self.available or not self.motion_manager:
            print("🎭 DEMO MODE - Simulating action")
            time.sleep(1.5)
            return True
        
        self.busy = True
        success = False
        
        try:
            # Execute the action using MotionManager
            # This is the proper way to control AiNex robots
            self.motion_manager.runAction(action_name)
            print("✅ Action completed")
            success = True
            
        except Exception as e:
            print(f"❌ Action failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Small delay before allowing next action
            time.sleep(0.3)
            self.busy = False
        
        return success

# ===================== UI FUNCTIONS =====================
def draw_ui(frame, detected_pose=None, fps=0, status="Ready"):
    """
    Draw user interface overlay on camera frame
    
    Args:
        frame: OpenCV frame (numpy array)
        detected_pose: Name of currently detected pose (or None)
        fps: Current frames per second
        status: Robot status ("Ready" or "Busy")
    
    Returns:
        frame: Frame with UI overlay drawn
    """
    h, w = frame.shape[:2]
    
    # === TOP BAR ===
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Title
    cv2.putText(frame, "HUMANOID MIMIC SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Robot status
    status_color = (0, 255, 0) if status == "Ready" else (0, 165, 255)
    cv2.putText(frame, f"Status: {status}", (w - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # === DETECTED POSE BANNER ===
    if detected_pose:
        pose_obj = next((p for p in POSES if p.name == detected_pose), None)
        if pose_obj:
            # Draw colored banner
            banner_y = 100
            banner_h = 60
            cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h),
                         pose_obj.color, -1)
            cv2.rectangle(frame, (10, banner_y), (w - 10, banner_y + banner_h),
                         (255, 255, 255), 3)
            
            # Draw text
            text = f">>> {detected_pose} <<<"
            cv2.putText(frame, text, (25, banner_y + 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # === POSE LIST PANEL ===
    panel_x = w - 320
    panel_y = 180
    panel_w = 310
    panel_h = len(POSES) * 50 + 50
    
    # Draw panel background
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
    
    # Draw each pose in the list
    y_pos = panel_y + 60
    for pose in POSES:
        is_active = detected_pose == pose.name
        
        # Color indicator circle
        cv2.circle(frame, (panel_x + 15, y_pos - 5), 7, pose.color, -1)
        cv2.circle(frame, (panel_x + 15, y_pos - 5), 7, (255, 255, 255), 1)
        
        # Pose name
        text_color = (255, 255, 255) if is_active else (180, 180, 180)
        thickness = 2 if is_active else 1
        cv2.putText(frame, pose.name, (panel_x + 30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)
        
        y_pos += 50
    
    # === INSTRUCTIONS ===
    cv2.putText(frame, "Press ESC or Q to exit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

# ===================== MAIN PROGRAM =====================
def main():
    """Main program loop"""
    
    print("\n🔧 Initializing system...")
    
    # Initialize robot controller
    robot = HumanoidController()
    
    # Initialize camera
    print("🎥 Starting camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera ready")
    print("\n" + "=" * 60)
    print("🟢 SYSTEM ACTIVE - Strike a pose!")
    print("=" * 60)
    
    # Main loop variables
    last_trigger = {}  # Track last trigger time for each pose
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            
            # Initialize detection
            detected_pose_name = None
            robot_status = "Busy" if robot.busy else "Ready"
            
            # Process pose landmarks
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                
                # Check each pose (only if robot is ready)
                if not robot.busy:
                    for pose_action in POSES:
                        try:
                            # Check if current body pose matches this action
                            if pose_action.check_func(landmarks):
                                detected_pose_name = pose_action.name
                                current_time = time.time()
                                
                                # Check cooldown period
                                last_time = last_trigger.get(pose_action.name, 0)
                                time_since_last = current_time - last_time
                                
                                if time_since_last > COOLDOWN_TIME:
                                    # Execute the robot action
                                    success = robot.execute_action(
                                        pose_action.action_name,
                                        pose_action.name
                                    )
                                    
                                    if success:
                                        last_trigger[pose_action.name] = current_time
                                
                                # Only trigger one pose at a time
                                break
                                
                        except Exception as e:
                            print(f"⚠️ Error checking pose {pose_action.name}: {e}")
                
                # Draw skeleton on frame
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
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw UI overlay
            frame = draw_ui(frame, detected_pose_name, fps, robot_status)
            
            # Display frame
            cv2.imshow("Humanoid Mimic System", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to exit
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

# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    main()
