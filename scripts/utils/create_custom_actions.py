import sqlite3 as sql
import os
import sys




class ActionCreator:
    """Create custom .d6a action files"""
    
    def __init__(self, output_dir=None):
        """
        Initialize action creator
        
        Args:
            output_dir: Directory to save actions
        """
        if output_dir is None:
            output_dir = os.path.expanduser(
                '~/humanoid_interaction_project/actions/custom'
            )
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_action_file(self, filename):
        """
        Create a new .d6a action file
        
        Args:
            filename: Name of action file (without .d6a)
        
        Returns:
            str: Full path to created file
        """
        filepath = os.path.join(self.output_dir, f"{filename}.d6a")
        
        # Create SQLite database
        conn = sql.connect(filepath)
        cursor = conn.cursor()
        
        # Create ActionGroup table (24 servos)
        cursor.execute('''
            CREATE TABLE ActionGroup(
                [Index] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL ON CONFLICT FAIL UNIQUE ON CONFLICT ABORT,
                Time INT,
                Servo1 INT, Servo2 INT, Servo3 INT, Servo4 INT,
                Servo5 INT, Servo6 INT, Servo7 INT, Servo8 INT,
                Servo9 INT, Servo10 INT, Servo11 INT, Servo12 INT,
                Servo13 INT, Servo14 INT, Servo15 INT, Servo16 INT,
                Servo17 INT, Servo18 INT, Servo19 INT, Servo20 INT,
                Servo21 INT, Servo22 INT, Servo23 INT, Servo24 INT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"? Created action file: {filepath}")
        return filepath

    
    def add_frame(self, filepath, time_ms, servo_positions):
        """
        Add a frame to an action file
        
        Args:
            filepath: Path to .d6a file
            time_ms: Duration of this frame in milliseconds
            servo_positions: Dict of {servo_id: position}
        """
        conn = sql.connect(filepath)
        cursor = conn.cursor()
        
        # Prepare servo values (24 servos, default 500)
        values = [500] * 24
        
        for servo_id, position in servo_positions.items():
            if 1 <= servo_id <= 24:
                values[servo_id - 1] = position
        
        # Insert frame
        cursor.execute(f'''
            INSERT INTO ActionGroup (
                Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo6,
                Servo7, Servo8, Servo9, Servo10, Servo11, Servo12,
                Servo13, Servo14, Servo15, Servo16, Servo17, Servo18,
                Servo19, Servo20, Servo21, Servo22, Servo23, Servo24
            ) VALUES (?, {','.join(['?']*24)})
        ''', [time_ms] + values)
        
        conn.commit()
        conn.close()
    
    def create_hands_up_action(self):
        """Create 'hands_up' action - both arms raised"""
        print("\nCreating 'hands_up' action...")
        
        filepath = self.create_action_file("hands_up")
        
        # Servo mapping (adjust based on your robot)
        # Right arm: 3=shoulder_pitch, 4=shoulder_roll, 5=elbow
        # Left arm: 7=shoulder_pitch, 8=shoulder_roll, 9=elbow
        
        # Frame 1: Start position (arms down)
        self.add_frame(filepath, 500, {
            3: 500, 4: 500, 5: 500,  # Right arm
            7: 500, 8: 500, 9: 500   # Left arm
        })
        
        # Frame 2: Raise both arms gradually
        self.add_frame(filepath, 800, {
            3: 300, 4: 400, 5: 400,  # Right arm going up
            7: 300, 8: 600, 9: 600   # Left arm going up
        })
        
        # Frame 3: Arms fully raised
        self.add_frame(filepath, 800, {
            3: 200, 4: 350, 5: 400,  # Right arm up
            7: 200, 8: 650, 9: 600   # Left arm up
        })
        
        # Frame 4: Hold position
        self.add_frame(filepath, 1000, {
            3: 200, 4: 350, 5: 400,
            7: 200, 8: 650, 9: 600
        })
        
        # Frame 5: Lower arms back
        self.add_frame(filepath, 800, {
            3: 500, 4: 500, 5: 500,
            7: 500, 8: 500, 9: 500
        })
        
        print(f"? 'hands_up' action created with 5 frames")

    
    def create_hands_straight_action(self):
        """Create 'hands_straight' action - T-pose"""
        print("\nCreating 'hands_straight' (T-pose) action...")
        
        filepath = self.create_action_file("hands_straight")
        
        # Frame 1: Start position
        self.add_frame(filepath, 500, {
            3: 500, 4: 500, 5: 500,
            7: 500, 8: 500, 9: 500
        })
        
        # Frame 2: Extend arms to sides gradually
        self.add_frame(filepath, 1000, {
            3: 500, 4: 250, 5: 500,  # Right arm extending right
            7: 500, 8: 750, 9: 500   # Left arm extending left
        })
        
        # Frame 3: Full T-pose with straight arms
        self.add_frame(filepath, 1000, {
            3: 500, 4: 200, 5: 450,  # Right arm straight out
            7: 500, 8: 800, 9: 550   # Left arm straight out
        })
        
        # Frame 4: Hold T-pose
        self.add_frame(filepath, 1500, {
            3: 500, 4: 200, 5: 450,
            7: 500, 8: 800, 9: 550
        })
        
        # Frame 5: Return to neutral
        self.add_frame(filepath, 1000, {
            3: 500, 4: 500, 5: 500,
            7: 500, 8: 500, 9: 500
        })
        
        print(f"? 'hands_straight' action created with 5 frames")
    
    def create_all_custom_actions(self):
        """Create all custom actions"""
        print("="*60)
        print("Creating Custom Actions for Gesture Control")
        print("="*60)
        
        self.create_hands_up_action()
        self.create_hands_straight_action()
        
        print("\n" + "="*60)
        print("? All custom actions created successfully!")
        print("="*60)
        print(f"\nLocation: {self.output_dir}")
        print("\nActions created:")
        print("  - hands_up.d6a (both arms raised)")
        print("  - hands_straight.d6a (T-pose)")
        print("\nThese actions will be triggered by:")
        print("  - Hands Up: Both palms raised gesture (??)")
        print("  - T-Pose: Both arms extended to sides (??)")
        print("\nNote: Servo values may need adjustment based on your")
        print("robot's specific configuration and calibration.")
        print("="*60)


def main():
    """Main function"""
    creator = ActionCreator()
    creator.create_all_custom_actions()


if __name__ == "__main__":
    main()
