#!/usr/bin/python3
import sys
import os
import time
import sqlite3

# --- SETUP PATHS ---
sys.path.insert(0, '/home/ubuntu/software/ainex_controller')
sys.path.insert(0, '/home/ubuntu/ros_ws/src/ainex_driver/ainex_sdk/src')

try:
    from ainex_sdk import Board
except ImportError:
    print("? Critical: Could not import AiNex SDK.")
    sys.exit(1)

# --- CONFIG ---
SAVE_DIR = os.path.expanduser("~/humanoid_interaction_project/actions/custom")

# --- SERVO MAP (AiNex V2.0 Standard) ---
JOINTS = {
    # RIGHT ARM
    23: "R_Shoulder_Up", 24: "R_Shoulder_Out", 25: "R_Elbow", 26: "R_Hand",
    # LEFT ARM
    13: "L_Shoulder_Up", 14: "L_Shoulder_Out", 15: "L_Elbow", 16: "L_Hand",
    # HEAD
    1: "Head_Pan", 2: "Head_Tilt"
}

class PoseEditor:
    def __init__(self):
        try:
            self.board = Board.Board()
        except AttributeError:
            self.board = Board()
        
        self.current_pos = {}
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def move(self, servo_id, angle):
        """Move servo safely (0-1000)"""
        if angle < 0 or angle > 1000:
            print("??  Angle must be 0-1000")
            return
            
        # Standard Hiwonder Move Command (Time=500ms)
        try:
            # Try Method A (Most common)
            self.board.setBusServoPulse(servo_id, angle, 500)
        except AttributeError:
            try:
                # Try Method B
                self.board.bus_servo_set_position(0.5, [[servo_id, angle]])
            except AttributeError:
                 # Try Method C
                self.board.set_motor_servo_position(0.5, servo_id, angle)

        self.current_pos[servo_id] = angle
        print(f"? Moved ID {servo_id} -> {angle}")

    def save(self, name):
        """Save to .d6a file"""
        full_path = os.path.join(SAVE_DIR, f"{name}.d6a")
        if os.path.exists(full_path): os.remove(full_path)
        
        conn = sqlite3.connect(full_path)
        c = conn.cursor()
        cols = ", ".join([f"Servo{i}" for i in range(1, 25)])
        c.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INT, {cols})")
        
        # Frame 0: Transition (1000ms)
        vals = [0, 1000]
        for i in range(1, 25):
            # Use current position or default to 500 (Center)
            vals.append(self.current_pos.get(i, 500))
            
        placeholders = "?, " * 26
        c.execute(f"INSERT INTO ActionGroup VALUES ({placeholders[:-2]})", vals)
        conn.commit()
        conn.close()
        print(f"\n?? ACTION SAVED: {full_path}")

def main():
    editor = PoseEditor()
    os.system('clear')
    print("="*60)
    print("??  DIGITAL POSE EDITOR - PRECISION MODE")
    print("="*60)
    print("COMMANDS:")
    print("  [ID] [ANGLE]   -> Move Joint (e.g., '23 800')")
    print("  save [NAME]    -> Save Action (e.g., 'save hands_up')")
    print("  q              -> Quit")
    print("-" * 60)
    print("KEY JOINTS:")
    for id, name in JOINTS.items():
        print(f"  ID {id}: {name}")
    print("="*60)

    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd: continue
            
            if cmd[0] == 'q': break
            
            if cmd[0] == 'save':
                if len(cmd) < 2: print("??  Name required!"); continue
                editor.save(cmd[1])
                continue
                
            # Move Command
            if len(cmd) == 2 and cmd[0].isdigit():
                editor.move(int(cmd[0]), int(cmd[1]))
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
