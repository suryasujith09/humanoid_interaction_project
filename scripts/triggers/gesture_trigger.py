
import cv2
import time
import threading
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gesture_recognizer import GestureRecognizer, GestureType




class GestureTrigger:
    """
    Camera-based gesture recognition trigger
    Maps gestures to robot actions
    """
    
    def __init__(self, action_callback, config_path=None, logger=None, camera_id=0):
        """
        Initialize gesture trigger
        
        Args:
            action_callback: Function to call with action name
            config_path: Path to config file
            logger: Logger instance
            camera_id: Camera device ID (default 0)
        """
        self.action_callback = action_callback
        self.logger = logger
        self.camera_id = camera_id
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Gesture to action mapping
        self.gesture_action_map = {
            GestureType.OPEN_PALM: "greet",
            GestureType.ROCK_SIGN: "wave",
            GestureType.PEACE_SIGN: "raise_right_hand",
            GestureType.FIST: "stand",
            GestureType.BOTH_HANDS_UP: "hands_up",
            GestureType.T_POSE: "hands_straight",
            GestureType.POINTING: "forward_one_step",
            GestureType.THUMBS_UP: "greet"
        }
        
        # State management
        self.enabled = True
        self.running = False
        self.cooldown = 3.0  # Seconds between same gesture triggers
        self.last_trigger_time = {}
        self.min_confidence = 0.7
        
        # Initialize gesture recognizer
        self.recognizer = GestureRecognizer(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Camera
        self.cap = None
        self.camera_thread = None
        
        # Display window
        self.show_window = True
        
        if self.logger:
            self.logger.info("Gesture trigger initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.expanduser(
                '~/humanoid_interaction_project/config/robot_config.yaml'
            )
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not load config: {e}")
            return {}
    
    def set_gesture_action(self, gesture_type, action_name):
        """
        Map a gesture to an action
        
        Args:
            gesture_type (GestureType): Gesture to map
            action_name (str): Action to trigger
        """
        self.gesture_action_map[gesture_type] = action_name
        if self.logger:
            self.logger.info(f"Mapped gesture '{gesture_type.value}' ? action '{action_name}'")
    

    
    def _can_trigger(self, gesture_type):
        """
        Check if gesture can trigger (cooldown check)
        
        Args:
            gesture_type (GestureType): Gesture to check
        
        Returns:
            bool: True if can trigger
        """
        current_time = time.time()
        last_time = self.last_trigger_time.get(gesture_type, 0)
        
        if current_time - last_time >= self.cooldown:
            self.last_trigger_time[gesture_type] = current_time
            return True
        
        return False
    
    def _process_gesture(self, gesture, confidence):
        """
        Process detected gesture and trigger action
        
        Args:
            gesture (GestureType): Detected gesture
            confidence (float): Detection confidence
        """
        if not self.enabled:
            return
        
        if gesture == GestureType.UNKNOWN:
            return
        
        if confidence < self.min_confidence:
            return
        
        if not self._can_trigger(gesture):
            return
        
        # Get mapped action
        action_name = self.gesture_action_map.get(gesture)
        
        if action_name is None:
            if self.logger:
                self.logger.warning(f"No action mapped for gesture: {gesture.value}")
            return
        
        if self.logger:
            self.logger.trigger(
                "gesture",
                {
                    "gesture": gesture.value,
                    "confidence": confidence,
                    "action": action_name
                }
            )
        
        # Trigger action
        try:
            self.action_callback(action_name)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error triggering action: {e}")

    
    def _camera_loop(self):
        """Main camera processing loop"""
        if self.logger:
            self.logger.info(f"Starting camera on device {self.camera_id}")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            if self.logger:
                self.logger.error(f"Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process gesture
            gesture, confidence, annotated_frame = self.recognizer.process_frame(frame)
            
            # Process detected gesture
            self._process_gesture(gesture, confidence)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add FPS to frame
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps}",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
            
            # Add status
            status_text = "ACTIVE" if self.enabled else "DISABLED"
            status_color = (0, 255, 0) if self.enabled else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                f"Status: {status_text}",
                (annotated_frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            
            # Display frame
            if self.show_window:
                cv2.imshow('Gesture Control - Press Q to quit', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if self.logger:
                        self.logger.info("User requested stop")
                    self.stop()
                    break
                elif key == ord(' '):  # Space to toggle
                    self.enabled = not self.enabled
                    if self.logger:
                        self.logger.info(f"Gesture trigger {'enabled' if self.enabled else 'disabled'}")
        
        # Cleanup
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        self.recognizer.cleanup()
        
        if self.logger:
            self.logger.info("Camera loop stopped")
    
    def start(self, blocking=False):
        """
        Start gesture recognition
        
        Args:
            blocking (bool): If True, run in current thread
        """
        if self.running:
            if self.logger:
                self.logger.warning("Gesture trigger already running")
            return
        
        self.running = True
        
        if blocking:
            self._camera_loop()
        else:
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            if self.logger:
                self.logger.info("Gesture trigger started in background")

    
    def stop(self):
        """Stop gesture recognition"""
        if self.logger:
            self.logger.info("Stopping gesture trigger...")
        
        self.running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
    
    def enable(self):
        """Enable gesture processing"""
        self.enabled = True
        if self.logger:
            self.logger.info("Gesture trigger enabled")
    
    def disable(self):
        """Disable gesture processing"""
        self.enabled = False
        if self.logger:
            self.logger.info("Gesture trigger disabled")
    
    def set_cooldown(self, seconds):
        """Set cooldown between triggers"""
        self.cooldown = seconds
        if self.logger:
            self.logger.info(f"Gesture cooldown set to {seconds}s")
    
    def set_show_window(self, show):
        """Enable/disable display window"""
        self.show_window = show

# Test function
if __name__ == "__main__":
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.logger import get_logger
    
    # Create logger
    logger = get_logger("GestureTriggerTest")
    
    # Define test callback
    def test_callback(action_name):
        logger.info(f"? WOULD TRIGGER ACTION: {action_name}")
    
    # Create trigger
    trigger = GestureTrigger(
        action_callback=test_callback,
        logger=logger,
        camera_id=0
    )
    
    logger.info("="*60)
    logger.info("Gesture Recognition Test")
    logger.info("="*60)
    logger.info("Show these gestures:")
    logger.info("  ? Open palm (5 fingers) ? Greet")
    logger.info("  ?? Rock sign (index + pinky) ? Wave")
    logger.info("  ??  Peace sign (2 fingers) ? Raise hand")
    logger.info("  ? Fist ? Stand")
    logger.info("  ?? Both hands up ? Hands up action")
    logger.info("  ?? T-pose (arms straight) ? Hands straight")
    logger.info("")
    logger.info("Press SPACE to enable/disable")
    logger.info("Press Q to quit")
    logger.info("="*60)
    
    # Start trigger (blocking mode for testing)
    trigger.start(blocking=True)
