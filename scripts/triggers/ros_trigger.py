
import rospy
from std_msgs.msg import String
import yaml
import os

class ROSTopicTrigger:
    """
    Listens to ROS topic for action commands
    Allows programmatic control of robot actions
    """
    
    def __init__(self, action_callback, config_path=None, logger=None):
        """
        Initialize ROS topic trigger
        
        Args:
            action_callback: Function to call with action name
            config_path: Path to config file
            logger: Logger instance
        """
        self.action_callback = action_callback
        self.logger = logger
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.trigger_config = self.config['triggers']['ros_topic']
        
        # State
        self.enabled = self.trigger_config.get('enabled', True)
        
        if not self.enabled:
            if self.logger:
                self.logger.info("ROS topic trigger is disabled")
            return
        
        # Subscribe to trigger topic
        self.trigger_topic = self.trigger_config.get(
            'topic',
            '/humanoid/trigger_action'
        )
        
        try:
            self.trigger_sub = rospy.Subscriber(
                self.trigger_topic,
                String,
                self.trigger_callback,
                queue_size=10
            )
            
            if self.logger:
                self.logger.info(f"Subscribed to trigger topic: {self.trigger_topic}")
                self.logger.info(f"Send action commands with: rostopic pub {self.trigger_topic} std_msgs/String \"data: 'wave'\"")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to subscribe to trigger topic: {e}")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.expanduser(
                '~/humanoid_interaction_project/config/robot_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    
    def trigger_callback(self, msg):
        """
        Callback when trigger message received
        
        Args:
            msg (String): ROS String message containing action name
        """
        if not self.enabled:
            return
        
        action_name = msg.data.strip()
        
        if not action_name:
            if self.logger:
                self.logger.warning("Received empty action name")
            return
        
        if self.logger:
            self.logger.trigger("ros_topic", {"action": action_name})
        
        try:
            self.action_callback(action_name)
            if self.logger:
                self.logger.info(f"Triggered action '{action_name}' from ROS topic")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error triggering action: {e}")
    
    def enable(self):
        """Enable ROS topic trigger"""
        self.enabled = True
        if self.logger:
            self.logger.info("ROS topic trigger enabled")
    
    def disable(self):
        """Disable ROS topic trigger"""
        self.enabled = False
        if self.logger:
            self.logger.info("ROS topic trigger disabled")


# Test function
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.logger import get_logger
    
    # Initialize ROS node
    rospy.init_node('ros_trigger_test', anonymous=True)
    
    # Create logger
    logger = get_logger("ROSTriggerTest")
    
    # Define test callback
    def test_callback(action_name):
        logger.info(f"TEST: Would trigger action '{action_name}'")
    
    # Create trigger
    trigger = ROSTopicTrigger(
        action_callback=test_callback,
        logger=logger
    )
    
    logger.info("ROS topic trigger test running...")
    logger.info("Test with: rostopic pub /humanoid/trigger_action std_msgs/String \"data: 'wave'\"")
    
    # Keep node running
    rospy.spin()
