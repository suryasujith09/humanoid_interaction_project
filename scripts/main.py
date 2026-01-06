import os
import sys
import rospy
import signal
import time
import argparse
from pathlib import Path

# Add project paths
project_root = os.path.expanduser('~/humanoid_interaction_project/scripts')
sys.path.insert(0, project_root)

# Import modules
try:
    from utils.logger import get_logger
    from controllers.action_controller import ActionController
    from triggers.pose_mimic_trigger import PoseMimicTrigger
    from triggers.gesture_trigger import GestureTrigger
    from triggers.ros_trigger import ROSTopicTrigger
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


class UltimateHumanoidSystem:
    def __init__(self, mode="mimic", enable_ros=True, camera_id="0"):
        self.mode = mode
        self.config_path = os.path.expanduser('~/humanoid_interaction_project/config/robot_config.yaml')
        self.logger = get_logger("UltimateHumanoid", self.config_path)
        self._print_banner()
        
        # Initialize ROS node
        try:
            rospy.init_node('ultimate_humanoid_system', anonymous=False)
            self.logger.info("✓ ROS node initialized")
        except Exception as e:
            self.logger.warning(f"ROS initialization failed: {e}")
            self.logger.info("  System will run without ROS")
        
        # Initialize action controller
        self.logger.info("Initializing action controller...")
        try:
            self.action_controller = ActionController(config_path=self.config_path, logger=self.logger)
            self.logger.info("✓ Action controller ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize action controller: {e}")
            sys.exit(1)
        
        self.available_actions = self.action_controller.list_available_actions()
        self.logger.info(f"✓ Found {len(self.available_actions)} available actions")
        
        self.triggers = {}
        self._initialize_triggers(mode, enable_ros, camera_id)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True
        self._print_ready_message()
    
    def _print_banner(self):
        print("\n" + "="*70)
        print("  🤖 ULTIMATE HUMANOID MIMIC SYSTEM")
        print("="*70)
        self.logger.info("System initializing...")
    
    def _initialize_triggers(self, mode, enable_ros, camera_id):
        self.logger.info(f"Initializing triggers (mode: {mode})...")
        
        if mode == "mimic":
            try:
                self.triggers['pose_mimic'] = PoseMimicTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger,
                    camera_id=camera_id 
                )
                self.logger.info("✓ Pose mimic trigger initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize pose mimic: {e}")
                sys.exit(1)
        
        elif mode == "gesture":
            try:
                self.triggers['gesture'] = GestureTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger,
                    camera_id=camera_id
                )
                self.logger.info("✓ Gesture recognition trigger initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize gesture: {e}")
                sys.exit(1)
        
        if enable_ros:
            try:
                self.triggers['ros'] = ROSTopicTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger
                )
                self.logger.info("✓ ROS topic trigger initialized")
            except Exception as e:
                self.logger.warning(f"ROS trigger failed: {e}")
    
    def handle_action_request(self, action_name):
        if action_name.startswith("custom/"):
            check_name = action_name
        else:
            check_name = action_name
        
        if check_name not in self.available_actions:
            custom_name = f"custom/{action_name}"
            if custom_name not in self.available_actions:
                self.logger.warning(f"Unknown action: {action_name}")
                return False
            action_name = custom_name
        
        if self.action_controller.is_action_running():
            return False
        
        self.logger.info(f"🎬 EXECUTING: {action_name}")
        success = self.action_controller.run_action(action_name, blocking=False)
        return success
    
    def _print_ready_message(self):
        print("\n" + "="*70)
        print("  ✅ SYSTEM READY!")
        print("="*70)
        print(f"  Using Camera: {self.triggers.get('pose_mimic', self.triggers.get('gesture')).camera_id}")
        print("\n  🚀 System running... Waiting for input")
    
    def run(self):
        self.logger.info("Starting main loop...")
        if 'pose_mimic' in self.triggers:
            self.triggers['pose_mimic'].start(blocking=False)
        elif 'gesture' in self.triggers:
            self.triggers['gesture'].start(blocking=False)
        
        try:
            if rospy.core.is_initialized():
                rate = rospy.Rate(10)
                while not rospy.is_shutdown() and self.running:
                    rate.sleep()
            else:
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        self.logger.info(f"Signal {signum} received, shutting down...")
        self.running = False
    
    def shutdown(self):
        self.logger.info("Shutting down system...")
        for name, trigger in self.triggers.items():
            try:
                if hasattr(trigger, 'stop'): trigger.stop()
            except Exception: pass
        if self.action_controller:
            try:
                self.action_controller.stop_action()
                self.action_controller.cleanup()
            except Exception: pass
        print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='Ultimate Humanoid Mimic System')
    parser.add_argument('--mode', choices=['mimic', 'gesture'], default='mimic', help='Control mode')
    parser.add_argument('--no-ros', action='store_true', help='Disable ROS topic control')
    
    
    args = parser.parse_args()
    
    try:
        system = UltimateHumanoidSystem(
            mode=args.mode,
            enable_ros=not args.no_ros,
            camera_id=args.camera
        )
        system.run()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
