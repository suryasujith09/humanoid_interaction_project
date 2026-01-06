import os
import sys
import time
import argparse
import signal
import sqlite3
import cv2
import mediapipe as mp
import threading
from pathlib import Path

# --- CRITICAL: IMPORT ROBOT SDK ---
# This is the bridge to the hardware. Without this, the robot stays silent.
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

try:
    from ainex_sdk import Board
except ImportError:
    print("❌ CRITICAL: AiNex SDK not found. Are you on the robot?")
    sys.exit(1)

# --- CONFIGURATION ---
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions/custom"
SYSTEM_ACTIONS_DIR = "/home/ubuntu/software/ainex_controller/ActionGroups"
COOLDOWN_TIME = 2.5  # Seconds between moves

# --- ROBOT CONTROLLER CLASS ---
class RobotController:
    def __init__(self):
        print("\n🔌 Connecting to Robot Hardware...")
        try:
            # Handle SDK import quirk
            try:
                self.board = Board.Board()
            except AttributeError:
                self.board = Board()
            print("✅ Robot Connected via Binary SDK")
        except Exception as e:
            print(f"❌ Hardware Connection Failed: {e}")
            print("   (Did you forget 'sudo systemctl stop ainex_controller'?)")
            sys.exit(1)
        
        self.lock = threading.Lock()

    def play_action(self, action_name):
        """
        Plays a .d6a file using the native SDK
        """
        # 1. Find the file
        filename = f"{action_name}.d6a" if not action_name.endswith(".d6a") else action_name
        path = Path(ACTIONS_DIR) / filename
        
        if not path.exists():
            # Try system folder
            path = Path(SYSTEM_ACTIONS_DIR) / filename
            if not path.exists():
                print(f"⚠️  Action file missing: {filename}")
                return

        # 2. Prevent overlapping actions
        if not self.lock.acquire(blocking=False):
            return # Busy
        
        try:
            print(f"🎬 ROBOT: Playing {action_name}...")
            
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            
            # Detect table name (ActionGroup vs frames)
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cur.fetchall()]
            table = "ActionGroup" if "ActionGroup" in tables else "frames"
            
            # Get frames
            frames = cur.execute(f"SELECT * FROM {table} ORDER BY [Index]").fetchall()
            conn.close()

            # Execute
            for frame in frames:
                duration_ms = frame[1]
                servo_data = frame[2:] # Servos 1-22+
                
                # Send to all servos
                for i, pos in enumerate(servo_data):
                    servo_id = i + 1
                    if servo_id > 22: break 
                    self.board.setBusServoPulse(servo_id, pos, duration_ms)
                
                time.sleep(duration_ms / 1000.0)
                
        except Exception as e:
            print(f"❌ Error playing action: {e}")
        finally:
            self.lock.release()

# --- VISION SYSTEM CLASS ---
class VisionSystem:
    def __init__(self, camera_path, robot):
        self.camera_path = camera_path
        self.robot = robot
        self.last_trigger = {}
        
        # Init MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

    def check_pose(self, lm):
        """Returns the name of the detected pose, or None"""
        
        # 1. HANDS UP (Wrists above nose)
        if (lm[15].y < lm[0].y and lm[16].y < lm[0].y):
            return "hands_up"

        # 2. T-POSE (Shoulders/Wrists aligned Y, spread X)
        sy = (lm[11].y + lm[12].y) / 2
        if (abs(lm[15].y - sy) < 0.2 and abs(lm[16].y - sy) < 0.2 and
            lm[15].x < lm[11].x - 0.1 and lm[16].x > lm[12].x + 0.1):
            return "hands_straight"

        # 3. WAVE (Right hand high and right)
        if (lm[16].y < lm[12].y - 0.2 and lm[16].x < lm[12].x):
            return "wave"
            
        # 4. STAND (Wrists below hips)
        hy = (lm[23].y + lm[24].y) / 2
        if (lm[15].y > hy and lm[16].y > hy):
            return "stand"
            
        return None

    def run(self):
        print(f"📷 Opening Camera: {self.camera_path}")
        
        # Handle Path vs Index
        if isinstance(self.camera_path, str) and not self.camera_path.isdigit():
             cap = cv2.VideoCapture(self.camera_path) # Path string
        else:
             cap = cv2.VideoCapture(int(self.camera_path), cv2.CAP_V4L2) # Index
             
        if not cap.isOpened():
            print("❌ Camera failed. Try --camera 0 or --camera /dev/usb_cam")
            return

        print("\n✅ SYSTEM ACTIVE - Stand back and Pose!")
        print("   (Hands Up, T-Pose, Wave, or Stand)")

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Process
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            
            status = "Scanning..."
            color = (255, 255, 0)

            if result.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Check Logic
                pose_name = self.check_pose(result.pose_landmarks.landmark)
                
                if pose_name:
                    now = time.time()
                    if now - self.last_trigger.get(pose_name, 0) > COOLDOWN_TIME:
                        # NEW TRIGGER
                        status = f"DETECTED: {pose_name.upper()}"
                        color = (0, 255, 0)
                        
                        # Trigger Robot in separate thread to not freeze video
                        threading.Thread(target=self.robot.play_action, args=(pose_name,)).start()
                        self.last_trigger[pose_name] = now
                    else:
                        status = f"Cooldown: {pose_name}"
                        color = (0, 0, 255)

            # UI
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Humanoid Mimic", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

# --- MAIN ENTRY POINT ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default="/dev/usb_cam", help="Camera path or ID")
    args = parser.parse_args()

    # 1. Cleanup
    print("🧹 Cleaning environment...")
    os.system("sudo systemctl stop ainex_controller > /dev/null 2>&1")
    os.system("sudo pkill -f python3 > /dev/null 2>&1")
    time.sleep(1)

    # 2. Init
    try:
        robot = RobotController()
        vision = VisionSystem(args.camera, robot)
        vision.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
