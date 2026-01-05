import os
import sys
import time
import sqlite3 as sql
import yaml
import threading
from pathlib import Path

sys.path.append('/home/ubuntu/software/ainex_controller')

try:
    from ros_robot_controller_sdk import Board
except ImportError:
    print("Warning: Could not import Board from ros_robot_controller_sdk")
    Board = None


class ActionController:
    """
    Controls robot actions using .d6a files
    Compatible with AiNex action file format
    """
    
    def __init__(self, config_path=None, logger=None):
        self.running_action = False
        self.stop_running = False
        self.action_thread = None
        self.logger = logger
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize board connection
        try:
            serial_config = self.config['serial']
            self.board = Board(
                device=serial_config['port'],
                baudrate=serial_config['baudrate'],
                timeout=serial_config['timeout']
            )
            if self.logger:
                self.logger.info(f"Board initialized on {serial_config['port']}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize board: {e}")
            self.board = None
        
        # Set action paths
        self.action_path = self.config['paths']['action_groups']
        self.custom_action_path = self.config['paths']['custom_actions']
        
        if self.logger:
            self.logger.info("ActionController initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.expanduser(
                '~/humanoid_interaction_project/config/robot_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def stop_servo(self):
        """Stop all servos immediately"""
        if self.board:
            try:
                servo_ids = list(range(1, self.config['robot']['total_servos'] + 1))
                self.board.stopBusServo(servo_ids)
                if self.logger:
                    self.logger.warning("All servos stopped")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to stop servos: {e}")
    
    def stop_action(self):
        """Stop the currently running action"""
        self.stop_running = True
        if self.logger:
            self.logger.info("Action stop requested")
    
    def is_action_running(self):
        """Check if an action is currently running"""
        return self.running_action
    
    def get_action_path(self, action_name):
        """
        Get full path to action file
        Checks custom actions first, then default actions
        """
        # Try custom actions first
        custom_path = os.path.join(self.custom_action_path, action_name + ".d6a")
        if os.path.exists(custom_path):
            return custom_path
        
        # Try default actions
        default_path = os.path.join(self.action_path, action_name + ".d6a")
        if os.path.exists(default_path):
            return default_path
        
        return None
    
    def list_available_actions(self):
        """List all available actions"""
        actions = []
        
        # Get default actions
        if os.path.exists(self.action_path):
            for file in os.listdir(self.action_path):
                if file.endswith('.d6a'):
                    actions.append(file[:-4])  # Remove .d6a extension
        
        # Get custom actions
        if os.path.exists(self.custom_action_path):
            for file in os.listdir(self.custom_action_path):
                if file.endswith('.d6a'):
                    action_name = f"custom/{file[:-4]}"
                    actions.append(action_name)
        
        return sorted(actions)

    
    def run_action(self, action_name, blocking=False):
        """
        Run an action from .d6a file
        
        Args:
            action_name (str): Name of the action (without .d6a extension)
            blocking (bool): If True, wait for action to complete
        
        Returns:
            bool: True if action started successfully, False otherwise
        """
        if action_name is None:
            if self.logger:
                self.logger.error("Action name is None")
            return False
        
        # Check if action is already running
        if self.running_action:
            if self.logger:
                self.logger.warning(f"Action already running, cannot start '{action_name}'")
            return False
        
        # Get action file path
        action_file = self.get_action_path(action_name)
        
        if action_file is None or not os.path.exists(action_file):
            if self.logger:
                self.logger.error(f"Action file not found: {action_name}")
            return False
        
        if self.logger:
            self.logger.action(action_name, "starting")
        
        # Run action in thread
        if blocking:
            self._execute_action(action_file, action_name)
        else:
            self.action_thread = threading.Thread(
                target=self._execute_action,
                args=(action_file, action_name)
            )
            self.action_thread.daemon = True
            self.action_thread.start()
        
        return True

    
    def _execute_action(self, action_file, action_name):
        """Execute action from SQLite database file"""
        self.running_action = True
        self.stop_running = False
        
        if not self.board:
            if self.logger:
                self.logger.error("Board not initialized, cannot execute action")
            self.running_action = False
            return
        
        try:
            # Connect to action database
            ag = sql.connect(action_file)
            cu = ag.cursor()
            cu.execute("SELECT * FROM ActionGroup")
            
            frame_count = 0
            
            while True:
                # Check for stop signal
                if self.stop_running:
                    if self.logger:
                        self.logger.action(action_name, "stopped")
                    self.stop_running = False
                    break
                
                # Get next frame
                act = cu.fetchone()
                
                if act is not None:
                    frame_count += 1
                    
                    # Parse servo data
                    # Format: [Index, Time, Servo1, Servo2, ..., ServoN]
                    time_ms = act[1]
                    servo_data = []
                    
                    for i in range(0, len(act) - 2):
                        servo_id = i + 1
                        position = act[2 + i]
                        if position is not None:  # Only add if position is specified
                            servo_data.append([servo_id, position])
                    
                    # Send command to servos
                    try:
                        self.board.bus_servo_set_position(time_ms / 1000.0, servo_data)
                        time.sleep(time_ms / 1000.0)
                        
                        if self.logger:
                            self.logger.debug(
                                f"Frame {frame_count}: {len(servo_data)} servos, "
                                f"duration {time_ms}ms"
                            )
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error executing frame {frame_count}: {e}")
                        break
                
                else:
                    # Action completed
                    if self.logger:
                        self.logger.action(action_name, "completed")
                    break
            
            cu.close()
            ag.close()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error running action '{action_name}': {e}")
        
        finally:
            self.running_action = False

    
    def wait_for_action(self, timeout=None):
        """
        Wait for current action to complete
        
        Args:
            timeout (float): Maximum time to wait in seconds
        
        Returns:
            bool: True if action completed, False if timeout
        """
        if self.action_thread and self.action_thread.is_alive():
            self.action_thread.join(timeout)
            return not self.action_thread.is_alive()
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_action()
        if self.action_thread:
            self.action_thread.join(timeout=2)
        if self.logger:
            self.logger.info("ActionController cleanup complete")

if __name__ == "__main__":
    # Import logger
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.logger import get_logger
    
    # Create logger
    logger = get_logger("ActionControllerTest")
    
    # Create controller
    controller = ActionController(logger=logger)
    
    # List available actions
    print("\nAvailable actions:")
    for action in controller.list_available_actions():
        print(f"  - {action}")
    
    # Test wave action
    print("\n=== Testing Wave Action ===")
    if controller.run_action("wave", blocking=False):
        print("Wave action started")
        controller.wait_for_action(timeout=10)
        print("Wave action completed")
    
    # Cleanup
    controller.cleanup()
