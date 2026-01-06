#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Pose Mimic Trigger - Robot mimics human poses
Save as: ~/humanoid_interaction_project/scripts/triggers/pose_mimic_trigger.py

Watches human and triggers matching robot actions.
Includes fix for OpenCV GStreamer errors on Raspberry Pi.
"""

import cv2
import time
import threading
import yaml
import os
import sys

# Ensure we can import from the utils directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.action_matcher import ActionMatcher
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.action_matcher import ActionMatcher


class PoseMimicTrigger:
    """
    Watches human poses and triggers matching robot actions
    """
    
    def __init__(self, action_callback, config_path=None, logger=None, camera_id=0):
        """
        Initialize pose mimic trigger
        
        Args:
            action_callback: Function to call with action name
            config_path: Path to config file
            logger: Logger instance
            camera_id: Camera device ID (usually 0)
        """
        self.action_callback = action_callback
        self.logger = logger
        self.camera_id = camera_id
        
        # State management
        self.enabled = True
        self.running = False
        self.cooldown = 2.0  # Seconds between triggers
        self.last_trigger_time = {}
        self.min_confidence = 0.7
        self.min_stable_frames = 8  # Frames pose must be stable
        
        # Pose tracking
        self.current_action = None
        self.action_frame_count = 0
        self.last_matched_action = None
        
        # Initialize action matcher
        self.matcher = ActionMatcher(logger=logger)
        
        # Camera
        self.cap = None
        self.camera_thread = None
        self.show_window = True
        
        # Display settings
        self.window_name = 'Robot Mimic System - Press Q to quit, SPACE to pause'
        
        if self.logger:
            self.logger.info("Pose mimic trigger initialized")
            self.logger.info(f"Min confidence: {self.min_confidence}")
            self.logger.info(f"Stable frames required: {self.min_stable_frames}")
    
    def _can_trigger(self, action_name):
        """Check if action can trigger (cooldown check)"""
        current_time = time.time()
        last_time = self.last_trigger_time.get(action_name, 0)
        
        if current_time - last_time >= self.cooldown:
            self.last_trigger_time[action_name] = current_time
            return True
        
        return False
    
    def _process_matched_action(self, action_name, confidence):
        """Process matched action with stability check"""
        if not self.enabled:
            return
        
        if action_name is None:
            self.current_action = None
            self.action_frame_count = 0
            return
        
        if confidence < self.min_confidence:
            return
        
        # Check if same action as before
        if action_name == self.current_action:
            self.action_frame_count += 1
        else:
            self.current_action = action_name
            self.action_frame_count = 1
        
        # Trigger if pose is stable
        if self.action_frame_count >= self.min_stable_frames:
            if action_name != self.last_matched_action:
                if self._can_trigger(action_name):
                    self._trigger_action(action_name, confidence)
                    self.last_matched_action = action_name
                    self.action_frame_count = 0
    
    def _trigger_action(self, action_name, confidence):
        """Trigger the robot action"""
        if self.logger:
            self.logger.trigger(
                "pose_mimic",
                {
                    "action": action_name,
                    "confidence": confidence,
                    "description": self.matcher.get_action_description(action_name)
                }
            )
        
        try:
            self.action_callback(action_name)
            if self.logger:
                self.logger.info(f"🤖 MIMICKING: {action_name} (confidence: {confidence:.2f})")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error triggering action: {e}")
    
    def _camera_loop(self):
        """Main camera processing loop"""
        if self.logger:
            self.logger.info(f"Starting camera on device {self.camera_id} (V4L2 Backend)")
        
        # --- CRITICAL FIX: FORCE V4L2 BACKEND ---
        # This prevents GStreamer errors on Raspberry Pi/Ubuntu
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            if self.logger:
                self.logger.error(f"Failed to open camera {self.camera_id}. Try checking connections or index.")
            print(f"❌ Error: Could not open video device {self.camera_id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res for faster FPS
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if self.logger:
            self.logger.info("Camera started successfully")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while self.running:
            success, frame = self.cap.read()
            
            if not success:
                if self.logger:
                    self.logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Flip for mirror view (more natural for mimicking)
            frame = cv2.flip(frame, 1)
            
            # Process pose and match action
            matched_action, confidence, annotated_frame = self.matcher.process_frame(frame)
            
            # Process the matched action
            self._process_matched_action(matched_action, confidence)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add UI overlays
            self._add_ui_overlay(annotated_frame, matched_action, confidence, current_fps)
            
            # Display frame
            if self.show_window:
                cv2.imshow(self.window_name, annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if self.logger:
                        self.logger.info("User requested stop")
                    self.stop()
                    break
                elif key == ord(' '):
                    self.enabled = not self.enabled
                    if self.logger:
                        self.logger.info(f"Pose mimic {'ENABLED' if self.enabled else 'DISABLED'}")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        
        # Only cleanup matcher if we initialized it
        if hasattr(self, 'matcher'):
            self.matcher.cleanup()
        
        if self.logger:
            self.logger.info("Camera loop stopped")
    
    def _add_ui_overlay(self, frame, matched_action, confidence, fps):
        """Add UI elements to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for status
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status
        status_text = "🟢 ACTIVE" if self.enabled else "🔴 PAUSED"
        status_color = (0, 255, 0) if self.enabled else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Matched action
        if matched_action and confidence > self.min_confidence:
            action_text = f"DETECTED: {matched_action.upper()}"
            conf_text = f"Confidence: {confidence:.2f}"
            
            # Stability indicator
            stability = min(self.action_frame_count / self.min_stable_frames, 1.0)
            stability_text = f"Stability: {'█' * int(stability * 10)}{' ' * (10 - int(stability * 10))} {int(stability * 100)}%"
            
            cv2.putText(frame, action_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, conf_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stability bar at bottom
            cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, stability_text, (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Instructions at bottom
        cv2.putText(frame, "Controls: SPACE=Pause/Resume | Q=Quit", (w-400, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def start(self, blocking=False):
        """Start pose mimic system"""
        if self.running:
            if self.logger:
                self.logger.warning("Pose mimic already running")
            return
        
        self.running = True
        
        if blocking:
            self._camera_loop()
        else:
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            if self.logger:
                self.logger.info("Pose mimic started in background")
    
    def stop(self):
        """Stop pose mimic system"""
        if self.logger:
            self.logger.info("Stopping pose mimic...")
        
        self.running = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
    
    def enable(self):
        """Enable pose processing"""
        self.enabled = True
        if self.logger:
            self.logger.info("Pose mimic enabled")
    
    def disable(self):
        """Disable pose processing"""
        self.enabled = False
        if self.logger:
            self.logger.info("Pose mimic disabled")
    
    def set_cooldown(self, seconds):
        """Set cooldown between triggers"""
        self.cooldown = seconds
        if self.logger:
            self.logger.info(f"Cooldown set to {seconds}s")
    
    def set_min_confidence(self, confidence):
        """Set minimum confidence threshold"""
        self.min_confidence = confidence
        if self.logger:
            self.logger.info(f"Min confidence set to {confidence}")


# Test function
if __name__ == "__main__":
    # Ensure correct paths for testing
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    try:
        from utils.logger import get_logger
    except ImportError:
        # Simple fallback logger if running standalone
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARN] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
            def trigger(self, t, d): print(f"[TRIGGER] {t} - {d}")
        
        def get_logger(name): return SimpleLogger()
    
    logger = get_logger("PoseMimicTest")
    
    def test_callback(action_name):
        print(f"\n✨ ROBOT TRIGGERED: {action_name} ✨")
        logger.info(f"Action triggered: {action_name}")
    
    # Initialize with Camera 0 (Change to 1 or -1 if needed)
    trigger = PoseMimicTrigger(
        action_callback=test_callback,
        logger=logger,
        camera_id=0
    )
    
    print("="*60)
    print("🤖 POSE MIMIC SYSTEM TEST (V4L2 MODE)")
    print("="*60)
    print("\nStand in front of the camera and perform these poses:")
    print("  • Wave - Raise right hand and move")
    print("  • Hands Up - Both hands straight up")
    print("  • T-Pose - Arms extended to sides")
    print("\nHold each pose for 1-2 seconds")
    print("Press SPACE to pause/resume, Q to quit")
    print("="*60)
    
    # Run loop
    try:
        trigger.start(blocking=True)
    except KeyboardInterrupt:
        trigger.stop()
