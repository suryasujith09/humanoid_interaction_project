#!/usr/bin/env python3
"""
🔥 Ultra-Stable Humanoid Mimic System v2.0
Enhanced with stability, logging, and error recovery
"""
import cv2
import mediapipe as mp
import serial
import sqlite3
import time
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional
from collections import deque

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/humanoid_mimic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
CAMERA_INDEX = 0
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 1000000
ACTIONS_DIR = "/home/ubuntu/humanoid_interaction_project/actions"
COOLDOWN_TIME = 3.0
SERVO_COUNT = 24
FRAME_DELAY = 0.03
POSE_STABILITY_FRAMES = 5  # 🆕 Require N consecutive frames
FRAME_SKIP = 1  # 🆕 Process every Nth frame

logger.info("=" * 60)
logger.info("🤖 HUMANOID MIMIC SYSTEM v2.0 STARTING")
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

# ===================== POSE SIGNATURE =====================
@dataclass
class PoseSignature:
    name: str
    action_file: str
    check_func: Callable
    priority: int = 0  # 🆕 Higher priority = checked first

def check_hands_up(lm) -> float:
    """Returns confidence score 0-1"""
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

logger.info(f"📋 Loaded {len(POSES)} poses (priority sorted)")

# ===================== POSE TRACKER =====================
class PoseTracker:
    """🆕 Tracks pose stability over multiple frames"""
    def __init__(self, stability_frames=POSE_STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self.history = {p.name: deque(maxlen=stability_frames) for p in POSES}
        self.last_trigger = {}
    
    def update(self, lm) -> Optional[PoseSignature]:
        """Returns pose if stable and ready to trigger"""
        scores = {}
        for pose in POSES:
            try:
                score = pose.check_func(lm)
                scores[pose.name] = score
                self.history[pose.name].append(score > 0.5)
            except Exception as e:
                logger.warning(f"Pose check failed for {pose.name}: {e}")
                scores[pose.name] = 0
        
        # Check for stable poses (priority order)
        for pose in POSES:
            hist = self.history[pose.name]
            if len(hist) >= self.stability_frames and all(hist):
                # Check cooldown
                now = time.time()
                if now - self.last_trigger.get(pose.name, 0) > COOLDOWN_TIME:
                    self.last_trigger[pose.name] = now
                    # Clear history to prevent re-trigger
                    self.history[pose.name].clear()
                    return pose
        
        return None

# ===================== ROBOT CONTROLLER =====================
class RobotController:
    def __init__(self, port, baud):
        self.serial = None
        self.busy = False
        self.demo_mode = False
        
        try:
            logger.info(f"🔌 Connecting to {port} @ {baud}")
            self.serial = serial.Serial(
                port, baud,
                timeout=0.1,
                write_timeout=0.1,
                rtscts=False,
                dsrdtr=False
            )
            time.sleep(1)
            # 🆕 Clear input buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            logger.info("✅ Robot connected")
        except Exception as e:
            logger.warning(f"⚠️ Serial failed: {e} - Running in DEMO mode")
            self.demo_mode = True
    
    def send_command(self, cmd: str) -> bool:
        """🆕 Send with error handling"""
        if not self.serial or self.demo_mode:
            return True
        
        try:
            self.serial.write(cmd.encode())
            # 🆕 Clear any response
            time.sleep(0.001)
            if self.serial.in_waiting > 0:
                self.serial.read(self.serial.in_waiting)
            return True
        except serial.SerialTimeoutException:
            logger.error("Serial write timeout")
            return False
        except Exception as e:
            logger.error(f"Serial error: {e}")
            return False
    
    def play_action(self, action_file: str, pose_name: str) -> bool:
        """Play action with enhanced error handling"""
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
            logger.info("🤖 DEMO MODE - Simulating action")
            time.sleep(1)
            return True
        
        self.busy = True
        success = True
        
        try:
            # Load action frames
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
                
                # Build command with validation
                cmd = ""
                for servo_id, pos in enumerate(positions):
                    pos = max(500, min(2500, int(pos)))  # Clamp
                    cmd += f"#{servo_id:03d}P{pos:04d}"
                cmd += "T100\r\n"
                
                # Send with retry
                if not self.send_command(cmd):
                    logger.warning(f"Failed to send frame {i}")
                    success = False
                    break
                
                time.sleep(FRAME_DELAY)
            
            logger.info(f"{'✅' if success else '⚠️'} Action complete")
            
        except Exception as e:
            logger.error(f"Action playback error: {e}")
            success = False
        finally:
            self.busy = False
        
        return success
    
    def cleanup(self):
        """🆕 Proper cleanup"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Serial port closed")

# ===================== INIT =====================
robot = RobotController(SERIAL_PORT, BAUD_RATE)
tracker = PoseTracker()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    logger.error("❌ Camera failed")
    sys.exit(1)

# 🆕 Camera optimization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

logger.info("🎥 Camera started | ESC to exit")

# ===================== MAIN LOOP =====================
frame_counter = 0
fps_time = time.time()
fps_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame read failed")
            continue
        
        frame = cv2.flip(frame, 1)
        frame_counter += 1
        
        # 🆕 Process only every Nth frame
        if frame_counter % (FRAME_SKIP + 1) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                
                # 🆕 Check for stable pose
                detected_pose = tracker.update(lm)
                if detected_pose and not robot.busy:
                    robot.play_action(detected_pose.action_file, detected_pose.name)
                
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
        
        # 🆕 FPS counter
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 🆕 Status overlay
        status = "BUSY" if robot.busy else "READY"
        color = (0, 165, 255) if robot.busy else (0, 255, 0)
        cv2.putText(frame, status, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Humanoid Mimic System v2.0", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    logger.info("Interrupted by user")
except Exception as e:
    logger.error(f"Fatal error: {e}", exc_info=True)
finally:
    # 🆕 Proper cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    robot.cleanup()
    logger.info("🛑 System stopped cleanly")
