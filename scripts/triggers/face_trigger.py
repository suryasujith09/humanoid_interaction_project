import rospy
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
import os


class FaceDetectionTrigger:
    """
    Monitors face detection and triggers robot actions
    Subscribes to /face_detect/faces topic
    """
    
    def __init__(self, action_callback, config_path=None, logger=None):
        """
        Initialize face detection trigger
        
        Args:
            action_callback: Function to call when face detected (takes action_name)
            config_path: Path to config file
            logger: Logger instance
        """
        self.action_callback = action_callback
        self.logger = logger
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.trigger_config = self.config['triggers']['face_detection']
        
        # State management
        self.last_trigger_time = 0
        self.cooldown = self.trigger_config.get('cooldown', 5.0)
        self.enabled = self.trigger_config.get('enabled', True)
        self.face_detected = False
        
        # Action to trigger
        self.trigger_action = "wave"  # Default action
        
        if not self.enabled:
            if self.logger:
                self.logger.info("Face detection trigger is disabled")
            return
        
        # Subscribe to face detection topic
        self.face_topic = self.trigger_config.get('topic', '/face_detect/faces')
        
        try:
            # The face_detect node might publish to different topics
            # Common topics: /face_detect/faces, /face_detect/result
            self.face_sub = rospy.Subscriber(
                self.face_topic,
                Image,  # Might be Image or custom message
                self.face_callback,
                queue_size=10
            )
            
            if self.logger:
                self.logger.info(f"Subscribed to face detection: {self.face_topic}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to subscribe to face detection: {e}")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.expanduser(
                '~/humanoid_interaction_project/config/robot_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def face_callback(self, msg):
        """
        Callback when face detection message received
        
        Args:
            msg: ROS message (format depends on face_detect node)
        """
        if not self.enabled:
            return
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_trigger_time < self.cooldown:
            return
        
        # Face detected!
        self.face_detected = True
        
        if self.logger:
            self.logger.trigger("face_detection", {"timestamp": current_time})
        
        # Trigger action
        self.last_trigger_time = current_time
        
        try:
            self.action_callback(self.trigger_action)
            if self.logger:
                self.logger.info(f"Triggered action '{self.trigger_action}' from face detection")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error triggering action: {e}")
    
    
    def set_trigger_action(self, action_name):
        """
        Set which action to trigger when face detected
        
        Args:
            action_name (str): Name of action to trigger
        """
        self.trigger_action = action_name
        if self.logger:
            self.logger.info(f"Face detection will now trigger: {action_name}")
    
    def enable(self):
        """Enable face detection trigger"""
        self.enabled = True
        if self.logger:
            self.logger.info("Face detection trigger enabled")
    
    def disable(self):
        """Disable face detection trigger"""
        self.enabled = False
        if self.logger:
            self.logger.info("Face detection trigger disabled")
    
    def set_cooldown(self, seconds):
        """
        Set cooldown period between triggers
        
        Args:
            seconds (float): Cooldown period in seconds
        """
        self.cooldown = seconds
        if self.logger:
            self.logger.info(f"Face detection cooldown set to {seconds}s")
    
    def is_face_detected(self):
        """Check if face is currently detected"""
        return self.face_detected
    
    def reset_face_status(self):
        """Reset face detection status"""
        self.face_detected = False

# Test function
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.logger import get_logger
    
    # Initialize ROS node
    rospy.init_node('face_trigger_test', anonymous=True)
    
    # Create logger
    logger = get_logger("FaceTriggerTest")
    
    # Define test callback
    def test_callback(action_name):
        logger.info(f"TEST: Would trigger action '{action_name}'")
    
    # Create trigger
    trigger = FaceDetectionTrigger(
        action_callback=test_callback,
        logger=logger
    )
    
    logger.info("Face detection trigger test running...")
    logger.info("Show your face to the camera to test!")
    
    # Keep node running
    rospy.spin()

