#!/usr/bin/python3
import sys
import os
import time
import sqlite3
import tty
import termios

# Add robot SDK paths
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

try:
    from ainex_sdk import Board
    import ros_robot_controller_sdk as robot_sdk
except ImportError:
    print("❌ Critical: Could not import AiNex SDK. Are paths correct?")
    sys.exit(1)

# Configuration
SAVE_DIR = os.path.expanduser("~/humanoid_interaction_project/actions/custom")
TOTAL_SERVOS = 22  # AiNex usually has 22 servos (IDs 1-22)

def getch():
    """Read single character without pressing enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class ActionRecorder:
    def __init__(self):
        print("🤖 Initializing Action Recorder...")
        try:
            self.board = Board.Board()
            print("✅ Board Connected")
        except Exception as e:
            print(f"❌ Failed to connect to board: {e}")
            sys.exit(1)
            
        self.frames = []
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def relax_arms(self):
        """Turn off torque for arm servos so you can move them"""
        print("\n🔓 Relaxing Arms (IDs 13-16, 23-26)...")
        # Left Arm & Right Arm IDs common on AiNex
        arm_ids = [13, 14, 15, 23, 24, 25] 
        for servo_id in arm_ids:
            self.board.set_motor_servo_torque(servo_id, 0) # 0 = Torque Off

    def lock_servos(self):
        """Re-enable torque"""
        print("\n🔒 Locking all servos...")
        for i in range(1, TOTAL_SERVOS + 1):
            self.board.set_motor_servo_torque(i, 1)

    def read_frame(self, duration_ms=1000):
        """Capture current position of all servos"""
        frame_data = {'time': duration_ms, 'servos': {}}
        
        print("   Reading servo positions...", end="", flush=True)
        for i in range(1, TOTAL_SERVOS + 1):
            # Read position (returns tuple, index 0 is pos)
            pos = self.board.get_motor_servo_position(i)
            # Handle reading errors or timeouts
            if pos is None or pos == -1:
                pos = 500 # Default safe center if read fails
            
            frame_data['servos'][i] = pos
            
        print(" Done!")
        return frame_data

    def save_to_d6a(self, filename):
        """Save captured frames to SQLite .d6a file"""
        full_path = os.path.join(SAVE_DIR, f"{filename}.d6a")
        
        # Remove if exists
        if os.path.exists(full_path):
            os.remove(full_path)
            
        conn = sqlite3.connect(full_path)
        c = conn.cursor()
        
        # Create table structure matching AiNex format
        # Columns: Index, Time, Servo1, Servo2 ... Servo24
        cols = ["Servo{}".format(i) for i in range(1, 25)]
        col_str = ", ".join(cols)
        c.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INT, {col_str})")
        
        for idx, frame in enumerate(self.frames):
            # Prepare row data
            time_val = frame['time']
            # Get servo values 1-24 (default 500 if missing)
            servo_vals = [frame['servos'].get(i, 500) for i in range(1, 25)]
            
            # Insert
            placeholders = "?, " * 25
            query = f"INSERT INTO ActionGroup VALUES ({idx}, {time_val}, {placeholders[:-2]})"
            c.execute(query, servo_vals)
            
        conn.commit()
        conn.close()
        print(f"\n💾 Saved {len(self.frames)} frames to: {full_path}")

def main():
    recorder = ActionRecorder()
    
    print("\n" + "="*50)
    print("🎬  HUMANOID ACTION STUDIO")
    print("="*50)
    print("1. Relax arms (Torque Off)")
    print("2. Move robot to desired pose")
    print("3. Press [SPACE] to capture frame")
    print("4. Press [S] to Save and Exit")
    print("5. Press [Q] to Quit without saving")
    print("="*50)

    recorder.relax_arms()
    
    while True:
        print(f"\nFrames captured: {len(recorder.frames)}")
        print("Waiting for command (Space/S/Q)...")
        key = getch()
        
        if key == ' ':
            # Capture
            frame = recorder.read_frame(duration_ms=1000) # 1s transition time
            recorder.frames.append(frame)
            print("📸 Frame Captured!")
            
        elif key.lower() == 's':
            if len(recorder.frames) == 0:
                print("⚠️ No frames to save!")
                continue
                
            name = input("\n📝 Enter action name (e.g., 'hands_up'): ").strip()
            if not name: name = "new_action"
            recorder.save_to_d6a(name)
            break
            
        elif key.lower() == 'q':
            print("Exiting...")
            break

    recorder.lock_servos()

if __name__ == "__main__":
    main()
