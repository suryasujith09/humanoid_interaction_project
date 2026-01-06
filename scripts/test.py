#!/usr/bin/env python3
"""
🔥 AiNex Humanoid Mimic - FIXED FOR HIWONDER PROTOCOL
Uses correct binary protocol for HiwonderSDK
"""
import cv2
import mediapipe as mp
import serial
import sqlite3
import time
import sys
import struct
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, List
from collections import deque

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
CAMERA_INDEX = 0
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000  # AiNex uses 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 3.0
SERVO_COUNT = 24
FRAME_DELAY = 0.02  # Faster for binary protocol
POSE_STABILITY_FRAMES = 5

logger.info("=" * 60)
logger.info("🤖 AINEX MIMIC - HIWONDER PROTOCOL")
logger.info("=" * 60)

# ===================== MEDIAPIPE =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ===================== POSE SIGNATURES =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable
    priority: int = 0

def check_hands_up(lm) -> float:
    left_up = max(0, (lm[0].y - lm[15].y) * 2)
    right_up = max(0, (lm[0].y - lm[16].y) * 2)
    return min(1.0, (left_up + right_up) / 2)

def check_hands_straight(lm) -> float:
    sy = (lm[11].y + lm[12].y) / 2
    left_diff = abs(lm[15].y - sy)
    right_diff = abs(lm[16].y - sy)
    return max(0, 1.0 - (left_diff + right_diff) * 3)

def check_wave(lm) -> float:
    hand_up = max(0, (lm[12].y - lm[16].y - 0.2) * 2)
    return min(1.0, hand_up)

def check_hands_down(lm) -> float:
    hy = (lm[23].y + lm[24].y) / 2
    left_down = max(0, (lm[15].y - hy) * 2)
    right_down = max(0, (lm[16].y - hy) * 2)
    return min(1.0, (left_down + right_down) / 2)

def check_place_block(lm) -> float:
    forward = max(0, (-lm[15].z - 0.15) * 3)
    return min(1.0, forward)

POSES = [
    PoseSignature("PLACE BLOCK", "placeblock.d6a", check_place_block, priority=5),
    PoseSignature("WAVE", "wave.d6a", check_wave, priority=4),
    PoseSignature("HANDS UP", "hands_up.d6a", check_hands_up, priority=3),
    PoseSignature("T-POSE", "hands_straight.d6a", check_hands_straight, priority=2),
    PoseSignature("HANDS DOWN", "greet.d6a", check_hands_down, priority=1),
]
POSES.sort(key=lambda p: p.priority, reverse=True)

# ===================== HIWONDER PROTOCOL =====================
class HiwonderServo:
    """Hiwonder Serial Bus Servo Protocol"""
    
    # Commands
    CMD_SERVO_MOVE = 0x03
    CMD_SERVO_MOVE_TIME_WRITE = 0x01
    CMD_SERVO_MOVE_TIME_READ = 0x02
    CMD_SERVO_MOVE_START = 0x0B
    CMD_SERVO_MOVE_STOP = 0x0C
    CMD_SERVO_ID_WRITE = 0x0D
    CMD_SERVO_ID_READ = 0x0E
    CMD_SERVO_ANGLE_OFFSET_ADJUST = 0x11
    CMD_SERVO_ANGLE_OFFSET_WRITE = 0x12
    CMD_SERVO_ANGLE_OFFSET_READ = 0x13
    CMD_SERVO_ANGLE_LIMIT_WRITE = 0x14
    CMD_SERVO_ANGLE_LIMIT_READ = 0x15
    CMD_SERVO_VIN_LIMIT_WRITE = 0x16
    CMD_SERVO_VIN_LIMIT_READ = 0x17
    CMD_SERVO_TEMP_MAX_LIMIT_WRITE = 0x18
    CMD_SERVO_TEMP_MAX_LIMIT_READ = 0x19
    CMD_SERVO_TEMP_READ = 0x1A
    CMD_SERVO_VIN_READ = 0x1B
    CMD_SERVO_POS_READ = 0x1C
    CMD_SERVO_OR_MOTOR_MODE_WRITE = 0x1D
    CMD_SERVO_OR_MOTOR_MODE_READ = 0x1E
    CMD_SERVO_LOAD_OR_UNLOAD_WRITE = 0x1F
    CMD_SERVO_LOAD_OR_UNLOAD_READ = 0x20
    CMD_SERVO_LED_CTRL_WRITE = 0x21
    CMD_SERVO_LED_CTRL_READ = 0x22
    CMD_SERVO_LED_ERROR_WRITE = 0x23
    CMD_SERVO_LED_ERROR_READ = 0x24
    
    @staticmethod
    def checksum(data: List[int]) -> int:
        """Calculate checksum"""
        return (~sum(data) & 0xFF)
    
    @staticmethod
    def move_time_write(servo_id: int, position: int, time_ms: int) -> bytes:
        """
        Command to move servo to position in given time
        Position: 0-1000 (0° to 240°)
        Time: milliseconds
        """
        pos_l = position & 0xFF
        pos_h = (position >> 8) & 0xFF
        time_l = time_ms & 0xFF
        time_h = (time_ms >> 8) & 0xFF
        
        data = [servo_id, 7, HiwonderServo.CMD_SERVO_MOVE_TIME_WRITE, 
                pos_l, pos_h, time_l, time_h]
        
        cmd = [0x55, 0x55] + data + [HiwonderServo.checksum(data)]
        return bytes(cmd)
    
    @staticmethod
    def move_servo(servo_id: int, position: int, time_ms: int = 100) -> bytes:
        """Simpler move command"""
        return HiwonderServo.move_time_write(servo_id, position, time_ms)

# ===================== POSE TRACKER =====================
class PoseTracker:
    def __init__(self, stability_frames=POSE_STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self.history = {p.name: deque(maxlen=stability_frames) for p in POSES}
        self.last_trigger = {}
    
    def update(self, lm) -> Optional[PoseSignature]:
        scores = {}
        for pose in POSES:
            try:
                score = pose.check_func(lm)
                scores[pose.name] = score
                self.history[pose.name].append(score > 0.5)
            except Exception as e:
                logger.warning(f"Pose check failed for {pose.name}: {e}")
                scores[pose.name] = 0
        
        for pose in POSES:
            hist = self.history[pose.name]
            if len(hist) >= self.stability_frames and all(hist):
                now = time.time()
                if now - self.last_trigger.get(pose.name, 0) > COOLDOWN_TIME:
                    self.last_trigger[pose.name] = now
                    self.history[pose.name].clear()
                    return pose
        
        return None

# ===================== ROBOT CONTROLLER =====================
class AiNexController:
    """AiNex Robot Controller with Hiwonder Protocol"""
    
    def __init__(self, port, baud):
        self.serial = None
        self.busy = False
        self.demo_mode = False
        
        try:
            logger.info(f"🔌 Connecting to {port} @ {baud}")
            self.serial = serial.Serial(
                port, baud,
                timeout=0.1,
                write_timeout=0.1
            )
            time.sleep(0.5)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            logger.info("✅ Robot connected (Hiwonder Protocol)")
        except Exception as e:
            logger.warning(f"⚠️ Serial failed: {e}")
            logger.info("Running in DEMO mode")
            self.demo_mode = True
    
    def send_servo_command(self, servo_id: int, position: int, time_ms: int = 100) -> bool:
        """Send single servo command using Hiwonder protocol"""
        if self.demo_mode:
            return True
        
        try:
            # Clamp position to valid range
            position = max(0, min(1000, position))
            
            cmd = HiwonderServo.move_servo(servo_id, position, time_ms)
            self.serial.write(cmd)
            
            # Small delay for servo processing
            time.sleep(0.001)
            
            # Clear any response
            if self.serial.in_waiting > 0:
                self.serial.read(self.serial.in_waiting)
            
            return True
        except Exception as e:
            logger.error(f"Servo command error: {e}")
            return False
    
    def play_action(self, action_file: str, pose_name: str) -> bool:
        """Play action file using Hiwonder protocol"""
        if self.busy:
            logger.debug(f"Busy - skipping {pose_name}")
            return False
        
        path = Path(ACTIONS_DIR) / action_file
        if not path.exists():
            logger.error(f"❌ Missing {path}")
            return False
        
        logger.info("=" * 50)
        logger.info(f"🎬 ACTION → {pose_name}")
        
        if self.demo_mode:
            logger.info("🤖 DEMO MODE")
            time.sleep(1)
            return True
        
        self.busy = True
        success = True
        
        try:
            # Load action frames from database
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cur.fetchall()]
            table = "frames" if "frames" in tables else "ActionGroup"
            start_idx = 1 if table == "frames" else 2
            
            frames = cur.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            conn.close()
            
            logger.info(f"📊 Frames: {len(frames)}")
            
            # Execute frames
            for i, frame in enumerate(frames):
                positions = frame[start_idx:start_idx + SERVO_COUNT]
                
                if len(positions) < SERVO_COUNT:
                    continue
                
                # Send all servo commands for this frame
                for servo_id, pos in enumerate(positions):
                    # Convert from pulse width (500-2500) to position (0-1000)
                    # AiNex uses 500-2500 pulse width = 0-1000 position
                    pos = int(pos)
                    if pos < 500 or pos > 2500:
                        pos = 1500  # Default center
                    
                    # Convert: 500-2500 → 0-1000
                    position = int((pos - 500) / 2)
                    
                    if not self.send_servo_command(servo_id + 1, position, 100):
                        logger.warning(f"Failed servo {servo_id} in frame {i}")
                
                time.sleep(FRAME_DELAY)
            
            logger.info("✅ Action complete")
            
        except Exception as e:
            logger.error(f"Action error: {e}")
            success = False
        finally:
            self.busy = False
        
        return success
    
    def cleanup(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Serial port closed")

# ===================== INIT =====================
robot = AiNexController(SERIAL_PORT, BAUD_RATE)
tracker = PoseTracker()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    logger.error("❌ Camera failed")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

logger.info("🎥 Camera started | ESC to exit")

# ===================== MAIN LOOP =====================
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_counter += 1
        
        if frame_counter % 2 == 0:  # Process every 2nd frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                
                detected_pose = tracker.update(lm)
                if detected_pose and not robot.busy:
                    robot.play_action(detected_pose.action_file, detected_pose.name)
                
                mp_draw.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
        
        status = "BUSY" if robot.busy else "READY"
        color = (0, 165, 255) if robot.busy else (0, 255, 0)
        cv2.putText(frame, status, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("AiNex Mimic (Hiwonder Protocol)", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    logger.info("Interrupted by user")
except Exception as e:
    logger.error(f"Fatal error: {e}", exc_info=True)
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    robot.cleanup()
    logger.info("🛑 System stopped")
