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
    print("❌ Critical: Could not import AiNex SDK.")
    sys.exit(1)

# --- CONFIG ---
SAVE_DIR = os.path.expanduser("~/humanoid_interaction_project/actions/custom")
TOTAL_SERVOS = 22

# Helpful ID Map for AiNex
JOINT_MAP = {
    # Right Arm
    23: "Right Shoulder Pitch (Up/Down)",
    24: "Right Shoulder Roll (Out/In)",
    25: "Right Elbow",
    26: "Right Hand/Gripper",
    # Left Arm
    13: "Left Shoulder Pitch (Up/Down)",
    14: "Left Shoulder Roll (Out/In)",
    15: "Left Elbow",
    16: "Left Hand/Gripper",
    # Head
    1: "Head Pan (Left/Right)",
    2: "Head Tilt (Up/Down)"
}

def print_status(current_positions):
    os.system('clear')
    print("="*60)
    print("🎮  DIGITAL POSE EDITOR")
    print("="*60)
    print(f"{'ID':<5} {'Value':<8} {'Description'}")
    print("-" * 40)
    
    # Print Arm Joints first (most important for you)
    key_joints = [23, 24, 25, 26, 13, 14, 15, 16, 1, 2]
    
    for servo_id in key_joints:
        val = current_positions.get(servo_id, "???")
        name = JOINT_MAP.get(servo_id, "Unknown")
        print(f"{servo_id:<5} {str(val):<8} {name}")
        
    print("-" * 40)
    print("Commands:")
    print("  [ID] [VALUE]  -> Move servo (e.g. '23 800')")
    print("                 (Range: 0-1000, 500 is usually center)")
    print("  relax         -> Turn off torque (careful!)")
    print("  stiff         -> Turn on torque")
    print("  save [name]   -> Save current pose (e.g. 'save hands_up')")
    print("  q             -> Quit")
    print("="*60)

def save_action(positions, name):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    full_path = os.path.join(SAVE_DIR, f"{name}.d6a")
    
    # Database Setup
    if os.path.exists(full_path):
        os.remove(full_path)
    
    conn = sqlite3.connect(full_path)
    c = conn.cursor()
    
    # Create Columns Servo1...Servo24
    cols = ", ".join([f"Servo{i}" for i in range(1, 25)])
    c.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INT, {cols})")
    
    # Prepare Data (Frame 0)
    # 1000ms duration for the move
    row_values = [0, 1000] 
    for i in range(1, 25):
        row_values.append(positions.get(i, 500)) # Default to 500 if missing
        
    placeholders = "?, " * 26
    c.execute(f"INSERT INTO ActionGroup VALUES ({placeholders[:-2]})", row_values)
    
    conn.commit()
    conn.close()
    print(f"\n✅ Saved action to: {full_path}")
    time.sleep(2)

def main():
    board = Board.Board()
    
    # Initialize dictionary with current positions
    current_pos = {}
    print("Reading initial positions...")
    for i in range(1, 25):
        pos = board.get_motor_servo_position(i)
        if pos and pos != -1:
            current_pos[i] = pos
        else:
            current_pos[i] = 500 # Default if read fails

    while True:
        print_status(current_pos)
        user_input = input("\n> ").strip().lower().split()
        
        if not user_input:
            continue
            
        cmd = user_input[0]
        
        if cmd == 'q':
            break
            
        elif cmd == 'save':
            if len(user_input) < 2:
                print("⚠️  Please provide a name (e.g., 'save hands_up')")
                time.sleep(1)
            else:
                save_action(current_pos, user_input[1])
                
        elif cmd == 'relax':
            # Emergency relax if you want to adjust by hand briefly
            board.set_motor_servo_torque(23, 0) # Example: Right Shoulder
            print("Relaxed right shoulder only for testing...")
            time.sleep(1)
            
        elif cmd == 'stiff':
            for i in range(1, 25):
                board.set_motor_servo_torque(i, 1)
                
        elif cmd.isdigit():
            # Handle Servo Move command
            if len(user_input) < 2:
                print("⚠️  Need a value (e.g. '23 500')")
                time.sleep(1)
                continue
                
            servo_id = int(cmd)
            target_val = int(user_input[1])
            
            # Safety Clamp (0-1000)
            target_val = max(0, min(1000, target_val))
            
            # Send Command (Time=500ms)
            board.bus_servo_set_position(0.5, [[servo_id, target_val]])
            
            # Update local record
            current_pos[servo_id] = target_val
            time.sleep(0.5) # Wait for move

if __name__ == "__main__":
    main()
