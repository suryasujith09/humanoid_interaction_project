import os
import sys
import rospy
import signal
import time
import argparse
from pathlib import Path

# Define project root and add to path
project_root = os.path.expanduser('~/humanoid_interaction_project/scripts')
sys.path.insert(0, project_root)

try:
    from utils.logger import get_logger
    from controllers.action_controller import ActionController
    from triggers.pose_mimic_trigger import PoseMimicTrigger
    from triggers.gesture_trigger import GestureTrigger
    from triggers.ros_trigger import ROSTopicTrigger
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class UltimateHumanoidSystem:
    def __init__(self, mode="mimic", enable_ros=True, camera_id=0):
        self.mode = mode
        self.config_path = os.path.expanduser(
            '~/humanoid_interaction_project/config/robot_config.yaml'
        )
        self.logger = get_logger("UltimateHumanoid", self.config_path)
        self._print_banner()
        
        try:
            rospy.init_node('ultimate_humanoid_system', anonymous=False)
            self.logger.info("[OK] ROS node initialized")
        except Exception as e:
            self.logger.warning(f"ROS initialization failed: {e}")
        
        self.logger.info("Initializing action controller...")
        try:
            self.action_controller = ActionController(
                config_path=self.config_path,
                logger=self.logger
            )
            self.logger.info("[OK] Action controller ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize action controller: {e}")
            sys.exit(1)
        
        self.available_actions = self.action_controller.list_available_actions()
        self._verify_actions()
        
        self.triggers = {}
        self._initialize_triggers(mode, enable_ros, camera_id)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        self._print_ready_message()

    def _print_banner(self):
        print("\n" + "="*70)
        print("  ULTIMATE HUMANOID MIMIC SYSTEM")
        print("="*70)
        self.logger.info("System initializing...")

    def _verify_actions(self):
        critical_actions = ['wave', 'greet', 'stand']
        missing = [a for a in critical_actions if a not in self.available_actions]
        if missing:
            self.logger.warning(f"Missing critical actions: {missing}")
        else:
            self.logger.info("[OK] All critical actions available")

    def _initialize_triggers(self, mode, enable_ros, camera_id):
        self.logger.info(f"Initializing triggers (mode: {mode})...")
        if mode == "mimic":
            self.triggers['pose_mimic'] = PoseMimicTrigger(
                action_callback=self.handle_action_request,
                config_path=self.config_path,
                logger=self.logger,
                camera_id=camera_id
            )
        elif mode == "gesture":
            self.triggers['gesture'] = GestureTrigger(
                action_callback=self.handle_action_request,
                config_path=self.config_path,
                logger=self.logger,
                camera_id=camera_id
            )
        
        if enable_ros:
            try:
                self.triggers['ros'] = ROSTopicTrigger(
                    action_callback=self.handle_action_request,
                    config_path=self.config_path,
                    logger=self.logger
                )
            except Exception:
                pass

    def handle_action_request(self, action_name):
        if action_name not in self.available_actions:
            custom_name = f"custom/{action_name}"
            if custom_name not in self.available_actions:
                return False
            action_name = custom_name
        
        if self.action_controller.is_action_running():
            return False
        
        self.logger.info(f">> EXECUTING: {action_name}")
        return self.action_controller.run_action(action_name, blocking=False)

    def _print_ready_message(self):
        print("\n" + "="*70)
        print("  SYSTEM READY!")
        print(f"  Mode: {self.mode.upper()}")
        print("="*70 + "\n")

    def run(self):
        self.logger.info("Starting main loop...")
        if 'pose_mimic' in self.triggers:
            self.triggers['pose_mimic'].start(blocking=False)
        elif 'gesture' in self.triggers:
            self.triggers['gesture'].start(blocking=False)
        
        try:
            while self.running and not (rospy.core.is_initialized() and rospy.is_shutdown()):
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")
        finally:
            self.shutdown()

    def _signal_handler(self, signum, frame):
        self.running = False

    def shutdown(self):
        self.logger.info("Shutting down...")
        for trigger in self.triggers.values():
            if hasattr(trigger, 'stop'): trigger.stop()
        if self.action_controller:
            self.action_controller.stop_action()
            self.action_controller.cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mimic', 'gesture'], default='mimic')
    parser.add_argument('--no-ros', action='store_true')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    try:
        UltimateHumanoidSystem(args.mode, not args.no_ros, args.camera).run()
    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    main()
