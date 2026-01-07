#!/usr/bin/env python3
"""
🔥 STANDALONE Humanoid Mimic System - FIXED
Directly reads .d6a files and controls servos via serial
"""
import cv2
import mediapipe as mp
import serial
import sqlite3
import time
import sys
from pathlib import Path
from dataclasses import dataclass

# ===================== CONFIG =====================
CAMERA_INDEX = 0
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 4.0
SERVO_COUNT = 24
FRAME_TIME = 100  # milliseconds per frame

print("=" * 60)
print("🤖 STANDALONE MIMIC SYSTEM v1.1")
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
class PoseAction:
    name: str
    file: str
    check_func: callable
    color: tuple

def check_hands_up(lm):
    return lm[15].y < lm[0].y - 0.1 and lm[16].y < lm[0].y - 0.1

def check_t_pose(lm):
    sy = (lm[11].y + lm[12].y) / 2
    return (abs(lm[15].y - sy) < 0.12 and 
            abs(lm[16].y - sy) < 0.12 and
            abs(lm[15].x - lm[16].x) > 0.5)

def check_wave(lm):
    return lm[16].y < lm[12].y - 0.15

def check_greet(lm):
    hy = (lm[23].y + lm[24].y) / 2
    return lm[15].y > hy - 0.15 and lm[16].y > hy - 0.15

def check_place(lm):
    return (lm[15].z < -0.1 and lm[16].z < -0.1 and
            abs(lm[15].x - lm[16].x) < 0.25)

POSES = [
    PoseAction("HANDS UP", "hands_up.d6a", check_hands_up, (0, 255, 255)),
    PoseAction("T-POSE", "hands_straight.d6a", check_t_pose, (255, 0, 255)),
    PoseAction("WAVE", "wave.d6a", check_wave, (255, 165, 0)),
    PoseAction("GREET", "greet.d6a", check_greet, (0, 255, 0)),
    PoseAction("PLACE", "placeblock.d6a", check_place, (255, 100, 100)),
]

print(f"\n📋 Checking {len(POSES)} actions:")
for p in POSES:
    path = Path(ACTIONS_DIR) / p.file
    status = "✅" if path.exists() else "❌"
    print(f"   {status} {p.name}: {p.file}")

# ===================== SERIAL SERVO CONTROLLER =====================
class ServoController:
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
            time.sleep(2)  # Let port stabilize
            print("✅ Serial connected")
        except Exception as e:
            print(f"⚠️ Serial failed: {e}")
            print("📝 Running in DEMO mode")
    
    def send_frame(self, positions):
        """Send servo positions for one frame"""
        if not self.serial:
            return
        
        cmd = ""
        for i, pos in enumerate(positions):
            # Clamp position to valid range
            pos = max(500, min(2500, int(pos)))
            cmd += f"#{i:02d}P{pos:04d}"
        
        cmd += f"T{FRAME_TIME}\r\n"
        
        try:
            self.serial.write(cmd.encode('ascii'))
            self.serial.flush()
        except Exception as e:
            print(f"⚠️ Serial write error: {e}")
    
    def play_d6a_file(self, filepath):
        """Read and play a .d6a action file"""
        if self.busy:
            return False
        
        path = Path(filepath)
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        print(f"\n📂 Loading: {path.name}")
        
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            
            # Get table structure
            tables = [t[0] for t in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            
            print(f"🔍 Tables found: {tables}")
            
            # Try common table names
            table = None
            if "ActionGroup" in tables:
                table = "ActionGroup"
            elif "frames" in tables:
                table = "frames"
            else:
                # Use first table
                table = tables[0] if tables else None
            
            if not table:
                print("⚠️ No valid table found")
                conn.close()
                return False
            
            # Get column info
            columns = [col[1] for col in cur.execute(f"PRAGMA table_info({table})").fetchall()]
            print(f"📊 Columns: {len(columns)} total")
            
            # Read all frames
            frames = cur.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            conn.close()
            
            if not frames:
                print("⚠️ No frames found")
                return False
            
            print(f"🎬 Playing {len(frames)} frames")
            print(f"📏 First frame has {len(frames[0])} columns")
            
            if not self.serial:
                print("🎭 DEMO MODE - simulating action")
                time.sleep(len(frames) * FRAME_TIME / 1000)
                return True
            
            self.busy = True
            
            # Determine start column (skip ID columns)
            # Most .d6a files have: [id, ...servo data...]
            # or [id, time, ...servo data...]
            start_col = 1  # Skip first ID column
            
            # If we have way more columns than expected, skip 2
            if len(frames[0]) > SERVO_COUNT + 5:
                start_col = 2
            
            print(f"📍 Reading servo data from column {start_col} onwards")
            
            # Play each frame
            played = 0
            for i, frame in enumerate(frames):
                # Extract servo positions
                end_col = start_col + SERVO_COUNT
                positions = frame[start_col:end_col]
                
                # Validate we have enough data
                if len(positions) < SERVO_COUNT:
                    # Try to fill missing servos with default position
                    positions = list(positions) + [1500] * (SERVO_COUNT - len(positions))
                
                self.send_frame(positions[:SERVO_COUNT])
                played += 1
                time.sleep(FRAME_TIME / 1000)
            
            print(f"✅ Played {played} frames successfully")
            self.busy = False
            return True
            
        except Exception as e:
            print(f"❌ Error playing action: {e}")
            import traceback
            traceback.print_exc()
            self.busy = False
            return False

# ===================== UI =====================
def draw_ui(frame, detected=None, fps=0, status="Ready"):
    h, w = frame.shape[:2]
    
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Title (using SIMPLEX which is always available)
    cv2.putText(frame, "HUMANOID MIMIC", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Status indicator
    status_color = (0, 255, 0) if status == "Ready" else (0, 165, 255)
    cv2.putText(frame, status, (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Detected pose banner
    if detected:
        pose_obj = next((p for p in POSES if p.name == detected), None)
        if pose_obj:
            cv2.rectangle(frame, (10, 90), (w - 10, 140), pose_obj.color, -1)
            cv2.rectangle(frame, (10, 90), (w - 10, 140), (255, 255, 255), 3)
            cv2.putText(frame, f">>> {detected} <<<", (25, 123),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Pose list panel
    panel_x = w - 250
    panel_y = 160
    panel_h = len(POSES) * 40 + 50
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + panel_h),
                 (20, 20, 20), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + panel_h),
                 (100, 100, 100), 2)
    
    cv2.putText(frame, "Available Poses:", (panel_x + 10, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    y = panel_y + 55
    for p in POSES:
        active = detected == p.name
        
        # Color dot
        cv2.circle(frame, (panel_x + 15, y - 5), 6, p.color, -1)
        cv2.circle(frame, (panel_x + 15, y - 5), 6, (255, 255, 255), 1)
        
        # Pose name
        text_color = (255, 255, 255) if active else (180, 180, 180)
        thickness = 2 if active else 1
        cv2.putText(frame, p.name, (panel_x + 30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)
        y += 40
    
    # Instructions
    cv2.putText(frame, "Press ESC or Q to exit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ===================== INIT =====================
controller = ServoController(SERIAL_PORT, BAUD_RATE)

print("\n🎥 Starting camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Camera failed")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Camera ready")
print("\n" + "=" * 60)
print("🟢 SYSTEM ACTIVE - Strike a pose!")
print("=" * 60)

# ===================== MAIN LOOP =====================
last_trigger = {}
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        
        detected = None
        status = "Busy" if controller.busy else "Ready"
        
        if result.pose_landmarks and not controller.busy:
            lm = result.pose_landmarks.landmark
            
            # Check each pose
            for p in POSES:
                try:
                    if p.check_func(lm):
                        detected = p.name
                        now = time.time()
                        
                        # Check cooldown
                        if now - last_trigger.get(p.name, 0) > COOLDOWN_TIME:
                            action_path = Path(ACTIONS_DIR) / p.file
                            if controller.play_d6a_file(action_path):
                                last_trigger[p.name] = now
                        break
                except Exception as e:
                    print(f"⚠️ Pose check error: {e}")
            
            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
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
        frame = draw_ui(frame, detected, fps, status)
        
        # Display
        cv2.imshow("Humanoid Mimic", frame)
        
        # Exit keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\n⚠️ Stopped by user")
except Exception as e:
    print(f"\n\n❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n🛑 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    if controller.serial:
        controller.serial.close()
    print("✅ Goodbye!")
