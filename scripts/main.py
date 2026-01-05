import os
import sys
import rospy
import signal
import time
import argparse
from pathlib import Path
project_root = os.path.expanduser('~/humanoid_interaction_project/scripts')
sys.path.insert(0, project_root)

# Import custom modules
# Note: Ensure these files exist in your directory structure
try:
    from utils.logger import get_logger
    from controllers.action_controller import ActionController
    from triggers.pose_mimic_trigger import PoseMimicTrigger
    from triggers.gesture_trigger import GestureTrigger
    from triggers.ros_trigger import ROSTopicTrigger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure your project structure is correct at: {project_root}")
    sys.exit(1)


class UltimateHumanoidSystem:
    def __init__(self, mode="mimic", enable_ros=True, camera_id=0):
        """
        Initialize the system
        
        Args:
            mode: "mimic" (pose matching) or "gesture" (hand gestures)
            enable_ros: Enable ROS topic control
            camera_id: Camera device ID
        """
        self.mode = mode
        self.config_path = os.path.expanduser(
            '~/humanoid_interaction_project/config/robot_config.yaml'
        )
        
        # Initialize logger
        # Note: If get_logger isn't available, we might need a fallback, 
        # but assuming it works based on imports above.
        self.logger = get_logger("UltimateHumanoid", self.config_path)
        self._print_banner()
        
        # Initialize ROS node
        try:
            rospy.init_node('ultimate_humanoid_system', anonymous=False)
            self.logger.info("[OK] ROS node initialized")
        except Exception as e:
            self.logger.warning(f"ROS initialization failed: {e}")
            self.logger.info("System will run without ROS")
        
        # Initialize action controller
        self.logger.info("Initializing action controller...")
        try:
            self.action_controller = ActionController(
                config_path=self.config_path,
                logger=self.logger
            )
            self.logger.info("[OK] Action controller ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize action controller: {e}")
            self.logger.error("Is the robot powered on?")
            self.logger.error("Is serial port /dev/ttyAMA0 accessible?")
            sys.exit(1)
        
        # Get available actions
        self.available_actions = self.action_controller.list_available_actions()
        self.logger.info(f"[OK] Found {len(self.available_actions)} available actions")
        
        # Verify critical actions exist
        self._verify_actions()
        
        # Initialize triggers based on mode
        self.triggers = {}
        self._initialize_triggers(mode, enable_ros, camera_id)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        
        self._print_ready_message()
    def _print_banner(self):
        """Print startup banner"""
        print("\n" + "="*70)
        print("  ULTIMATE HUMANOID MIMIC SYSTEM")
        print("  Advanced Pose Matching & Gesture Recognition")
        print("="*70)
        self.logger.info("System initializing...")

    def _verify_actions(self):
        """Verify that critical actions exist"""
        critical_actions = ['wave', 'greet', 'stand']
        missing = []
        
        for action in critical_actions:
            if action not in self.available_actions:
                missing.append(action)
        
        if missing:
            self.logger.warning(f"Missing critical actions: {', '.join(missing)}")
            self.logger.info("System will continue, but some features may not work")
        else:
            self.logger.info("[OK] All critical actions available")
        
        # Show available actions
        self.logger.info("Available actions:")
        for i, action in enumerate(self.available_actions[:15], 1):
            self.logger.info(f"  {i}. {action}")
        if len(self.available_actions) > 15:
            self.logger.info(f"  ... and {len(self.available_actions) - 15} more")

    def _initialize_triggers(self, mode, enable_ros, camera_id):
        """Initialize trigger systems"""
        self.logger.info(f"Initializing triggers (mode: {mode})...")
        
        # Pose mimic trigger (MAIN MODE)
        if mode == "mimic":
            try:
                self.triggers['pose_mimic'] = PoseMimicTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger,
                    camera_id=camera_id
                )
                self.logger.info("[OK] Pose mimic trigger initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize pose mimic: {e}")
                sys.exit(1)
        
        # Gesture recognition trigger (ALTERNATE MODE)
        elif mode == "gesture":
            try:
                self.triggers['gesture'] = GestureTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger,
                    camera_id=camera_id
                )
                self.logger.info("[OK] Gesture recognition trigger initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize gesture recognition: {e}")
                sys.exit(1)
        
        # ROS topic trigger (ALWAYS AVAILABLE)
        if enable_ros:
            try:
                self.triggers['ros'] = ROSTopicTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger
                )
                self.logger.info("[OK] ROS topic trigger initialized")
            except Exception as e:
                self.logger.warning(f"ROS trigger failed: {e}")

def handle_action_request(self, action_name):
        """
        Handle action request with full verification
        
        Args:
            action_name: Name of action to execute
        
        Returns:
            bool: Success status
        """
        # Normalize action name
        if action_name.startswith("custom/"):
            check_name = action_name
        else:
            check_name = action_name
        
        # Verify action exists
        if check_name not in self.available_actions:
            # Try custom path
            custom_name = f"custom/{action_name}"
            if custom_name not in self.available_actions:
                self.logger.warning(f"Unknown action: {action_name}")
                return False
            action_name = custom_name
        
        # Check if action currently running
        if self.action_controller.is_action_running():
            self.logger.debug(f"Action {action_name} queued (current action running)")
            return False
        
        # Execute action
        self.logger.info(f">> EXECUTING: {action_name}")
        success = self.action_controller.run_action(action_name, blocking=False)
        
        if success:
            self.logger.info(f"[+] Action {action_name} started")
        else:
            self.logger.error(f"[-] Failed to start {action_name}")
        
        return success

def _print_ready_message(self):
        """Print ready message with instructions"""
        print("\n" + "="*70)
        print("  SYSTEM READY!")
        
        if self.mode == "mimic":
            print("\n POSE MIMIC MODE")
            print("  Stand in front of camera and perform these poses:")
            print("  Wave - Raise right hand")
            print("  Greet - Raise both hands")
            print("  Hands Up - Both hands straight up")
            print("  T-Pose - Arms extended to sides")
            print("  Stand - Neutral standing")
            print("\n  Hold each pose for 1-2 seconds for recognition")
        
        elif self.mode == "gesture":
            print("\nGESTURE RECOGNITION MODE")
            print("  Show hand gestures to camera:")
            print("  Open palm -> Greet")
            print("  Rock sign -> Wave")
            print("  Peace sign -> Raise hand")
            print("  Fist -> Stand")
            print("  Both hands up -> Hands up")
            print("  T-pose -> Hands straight")
        
        print("\n CONTROLS:")
        print("  SPACE - Pause/Resume detection")
        print("  Q - Quit camera window")
        print("  Ctrl+C - Shutdown system")

        if 'ros' in self.triggers:
            print("\nROS TOPIC CONTROL:")
            print("  rostopic pub /humanoid/trigger_action std_msgs/String \"data: 'ACTION'\"")
        
        print("\nSTATUS:")
        print(f"  Mode: {self.mode.upper()}")
        print(f"  Actions Available: {len(self.available_actions)}")
        print(f"  Camera: Active")
        print(f"  Robot: {'Connected' if self.action_controller.board else 'Disconnected'}")
        
        print("\n" + "="*70)
        print("  System running... Waiting for input")
        print("="*70 + "\n")

    def run(self):
        """Main run loop"""
        self.logger.info("Starting main loop...")
        
        # Start primary trigger
        if 'pose_mimic' in self.triggers:
            self.triggers['pose_mimic'].start(blocking=False)
        elif 'gesture' in self.triggers:
            self.triggers['gesture'].start(blocking=False)
        
        # Main loop
        try:
            if rospy.core.is_initialized():
                rate = rospy.Rate(10)
                while not rospy.is_shutdown() and self.running:
                    rate.sleep()
            else:
                # Non-ROS mode
                while self.running:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Signal {signum} received, shutting down...")
        self.running = False

    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("="*70)
        self.logger.info("Shutting down system...")
        
        # Stop triggers
        for name, trigger in self.triggers.items():
            try:
                if hasattr(trigger, 'stop'):
                    trigger.stop()
                    self.logger.info(f"[OK] {name} trigger stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping {name}: {e}")
        
        # Stop actions
        if self.action_controller:
            try:
                self.action_controller.stop_action()
                self.action_controller.cleanup()
                self.logger.info("[OK] Action controller stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping controller: {e}")
        
        self.logger.info("="*70)
        self.logger.info("Shutdown complete")
        print("\nGoodbye! Thanks for using Ultimate Humanoid System\n")
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ultimate Humanoid Mimic System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                     # Pose mimic mode (default)
  python3 main.py --mode gesture      # Hand gesture mode
  python3 main.py --camera 1          # Use camera device 1
  python3 main.py --no-ros            # Disable ROS control
        """
    )
    
    parser.add_argument('--mode', choices=['mimic', 'gesture'], default='mimic',
                        help='Control mode: mimic (pose matching) or gesture (hand gestures)')
    parser.add_argument('--no-ros', action='store_true',
                        help='Disable ROS topic control')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    try:
        system = UltimateHumanoidSystem(
            mode=args.mode,
            enable_ros=not args.no_ros,
            camera_id=args.camera
        )
        system.run()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
