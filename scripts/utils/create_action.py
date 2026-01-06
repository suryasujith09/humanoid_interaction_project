#!/usr/bin/python3
import sys
import os
import time
import sqlite3
import tty
import termios

# --- SETUP PATHS ---
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

try:
    from ainex_sdk import Board
except ImportError:
    print("❌ Critical: Could not import AiNex SDK. Are paths correct?")
    sys.exit(1)

# --- CONFIG ---
SAVE_DIR = os.path.expanduser("~/humanoid_interaction_project/actions/custom")
TOTAL_SERVOS = 22

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
            # Connect to Board (Handle Module vs Class issue)
            try:
                self.board = Board.Board()
            except AttributeError:
                self.board = Board()
            print("✅ Board Connected")
        except Exception as e:
            print(f"❌ Failed to connect to board: {e}")
            sys.exit(1)
            
        self.frames = []
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def set_torque(self, servo_id, state):
        """
        Robust function to set torque using whatever method exists on this robot.
        state: 1 = ON (Locked), 0 = OFF (Relaxed)
        """
        # List of all possible function names found in different Hiwonder SDK versions
        possible_methods = [
            ('setBusServoTorque', [servo_id, state]),
            ('set_motor_servo_torque', [servo_id, state]),
            ('set_torque', [servo_id, state]),
            ('enable_servo_torque', [servo_id, bool(state)]),
            ('unload', [servo_id]) if state == 0 else ('load', [servo_id]),
        ]
        
        success = False
        for method_name, args in possible_methods:
            if hasattr(self.board, method_name):
                try:
                    func = getattr(self.board, method_name)
                    func(*args)
                    success = True
                    break
                except Exception:
                    continue
        
        if not success:
            # If all else fails, printing this helps debug
            print(f"⚠️ Warning: Could not find torque function for ID {servo_id}")

    def relax_arms(self):
        """Turn off torque for arm servos"""
        print("\n🔓 Relaxing Arms (IDs 13-16, 23-26)...")
        arm_ids = [13, 14, 15, 23, 24, 25] 
        for servo_id in arm_ids:
            self.set_torque(servo_id, 0)

    def lock_servos(self):
        """Re-enable torque"""
        print("\n🔒 Locking all servos...")
        for i in range(1, TOTAL_SERVOS + 1):
            self.set_torque(i, 1)

    def read_frame(self, duration_ms=1000):
        """Capture current position of all servos"""
        frame_data = {'time': duration_ms, 'servos': {}}
        
        print("   Reading positions...", end="", flush=True)
        for i in range(1, TOTAL_SERVOS + 1):
            # Try different position reading methods
            pos = None
            if hasattr(self.board, 'get_motor_servo_position'):
                pos = self.board.get_motor_servo_position(i)
            elif hasattr(self.board, 'getBusServoPosition'):
                pos = self.board.getBusServoPosition(i)
                
            # Handle list/tuple return types
            if isinstance(pos, (list, tuple)):
                pos = pos[0]
            
            # Default if failed
            if pos is None or pos == -1:
                pos = 500
            
            frame_data['servos'][i] = int(pos)
            
        print(" Done!")
        return frame_data

    def save_to_d6a(self, filename):
        full_path = os.path.join(SAVE_DIR, f"{filename}.d6a")
        if os.path.exists(full_path):
            os.remove(full_path)
            
        conn = sqlite3.connect(full_path)
        c = conn.cursor()
        
        cols = ["Servo{}".format(i) for i in range(1, 25)]
        col_str = ", ".join(cols)
        c.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INT, {col_str})")
        
        for idx, frame in enumerate(self.frames):
            time_val = frame['time']
            servo_vals = [frame['servos'].get(i, 500) for i in range(1, 25)]
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
    print("1. Arms will relax automatically")
    print("2. Move robot to desired pose")
    print("3. Press [SPACE] to capture frame")
    print("4. Press [S] to Save and Exit")
    print("5. Press [Q] to Quit")
    print("="*50)

    recorder.relax_arms()
    
    while True:
        print(f"\nFrames captured: {len(recorder.frames)}")
        print("Waiting for command (Space/S/Q)...")
        key = getch()
        
        if key == ' ':
            frame = recorder.read_frame(duration_ms=1000)
            recorder.frames.append(frame)
            print("📸 Frame Captured!")
            
        elif key.lower() == 's':
            if not recorder.frames:
                print("⚠️ No frames!")
                continue
            name = input("\n📝 Action name (e.g. 'hands_up'): ").strip()
            if not name: name = "custom_action"
            recorder.save_to_d6a(name)
            break
            
        elif key.lower() == 'q':
            break

    recorder.lock_servos()

if __name__ == "__main__":
    main()
