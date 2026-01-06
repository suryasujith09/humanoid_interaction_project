#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ULTIMATE HUMANOID MIMIC SYSTEM - FINAL VERSION
Complete pose mimicking with gesture recognition

Save as: ~/humanoid_interaction_project/scripts/main.py
Run: python3 main.py --camera /dev/usb_cam

Features:
- Real-time pose matching and mimicking
- Hand gesture recognition
- Supports Integer IDs (0) AND Paths (/dev/usb_cam)
"""

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
    print("Are you running from the 'scripts' folder?")
    sys.exit(1)


class UltimateHumanoidSystem:
    """
    Ultimate Humanoid Mimic System
    Complete integration of all features
    """
    
    def __init__(self, mode="mimic", enable_ros=True, camera_id="0"):
        """
        Initialize the system
        
        Args:
            mode: "mimic" (pose matching) or "gesture" (hand gestures)
            enable_ros: Enable ROS topic control
            camera_id: Camera device ID (int) or Path (str)
        """
        self.mode = mode
        self.config_path = os.path.expanduser(
            '~/humanoid_interaction_project/config/robot_config.yaml'
        )
        
        # Initialize logger
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
            self.action_controller = ActionController(
                config_path=self.config_path,
                logger=self.logger
            )
            self.logger.info("✓ Action controller ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize action controller: {e}")
            self.logger.error("  Is the robot powered on?")
            self.logger.error("  Is serial port /dev/ttyAMA0 accessible?")
            sys.exit(1)
        
        # Get available actions
        self.available_actions = self.action_controller.list_available_actions()
        self.logger.info(f"✓ Found {len(self.available_actions)} available actions")
        
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
        print("  🤖 ULTIMATE HUMANOID MIMIC SYSTEM")
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
        else:
            self.logger.info("✓ All critical actions available")
    
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
                    camera_id=camera_id  # Passes "0" or "/dev/usb_cam"
                )
                self.logger.info("✓ Pose mimic trigger initialized")
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
                self.logger.info("✓ Gesture recognition trigger initialized")
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
                self.logger.info("✓ ROS topic trigger initialized")
            except Exception as e:
                self.logger.warning(f"ROS trigger failed: {e}")
    
    def handle_action_request(self, action_name):
        """Handle action request with full verification"""
        # Normalize action name
        if action_name.startswith("custom/"):
            check_name = action_name
        else:
            check_name = action_name
        
        # Verify action exists
        if check_name not in self.available_actions:
            custom_name = f"custom/{action_name}"
            if custom_name not in self.available_actions:
                self.logger.warning(f"Unknown action: {action_name}")
                return False
            action_name = custom_name
        
        # Check if action currently running
        if self.action_controller.is_action_running():
            return False
        
        # Execute action
        self.logger.info(f"🎬 EXECUTING: {action_name}")
        success = self.action_controller.run_action(action_name, blocking=False)
        return success
    
    def _print_ready_message(self):
        print("\n" + "="*70)
        print("  ✅ SYSTEM READY!")
        print("="*70)
        print(f"  Camera Mode: {self.triggers.get('pose_mimic', self.triggers.get('gesture')).camera_id}")
        print("\n  🚀 System running... Waiting for input")
        print("="*70 + "\n")
    
    def run(self):
        """Main run loop"""
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
                if hasattr(trigger, 'stop'):
                    trigger.stop()
            except Exception:
                pass
        
        if self.action_controller:
            try:
                self.action_controller.stop_action()
                self.action_controller.cleanup()
            except Exception:
                pass
        print("\n👋 Goodbye!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ultimate Humanoid Mimic System'
    )
    
    parser.add_argument('--mode', choices=['mimic', 'gesture'], default='mimic',
                        help='Control mode: mimic or gesture')
    parser.add_argument('--no-ros', action='store_true',
                        help='Disable ROS topic control')
    
    # --- CRITICAL FIX: type=str allows "/dev/usb_cam" ---
    parser.add_argument('--camera', type=str, default="0",
                        help='Camera device ID (0) or path (/dev/usb_cam)')
    
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
