import os
import logging
import yaml
from datetime import datetime
from pathlib import Path

class RobotLogger:
    """Custom logger for robot operations"""
    
    def __init__(self, name="HumanoidRobot", config_path=None):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                log_config = config.get('logging', {})
        else:
            # Default config
            log_config = {
                'level': 'INFO',
                'log_dir': os.path.expanduser('~/humanoid_interaction_project/logs'),
                'console_output': True,
                'file_output': True
            }
        
        # Set log level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = level_map.get(log_config.get('level', 'INFO'), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file_output', True):
            log_dir = Path(log_config.get('log_dir', './logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Log file created: {log_file}")

    
    def debug(self, msg):
        """Log debug message"""
        self.logger.debug(msg)
    
    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """Log error message"""
        self.logger.error(msg)
    
    def critical(self, msg):
        """Log critical message"""
        self.logger.critical(msg)
    
    def action(self, action_name, status="started"):
        """Log action events"""
        self.logger.info(f"Action '{action_name}' - {status}")
    
    def servo(self, servo_id, position, status="moving"):
        """Log servo movements"""
        self.logger.debug(f"Servo {servo_id} - Position: {position} - {status}")
    
    def trigger(self, trigger_type, data=None):
        """Log trigger events"""
        msg = f"Trigger detected: {trigger_type}"
        if data:
            msg += f" - Data: {data}"
        self.logger.info(msg)



# Convenience function
def get_logger(name="HumanoidRobot", config_path=None):
    """Get a logger instance"""
    return RobotLogger(name, config_path)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("TestLogger")
    
    logger.debug("This is a debug message")
    logger.info("Robot initialized successfully")
    logger.warning("Battery level low")
    logger.error("Servo communication failed")
    logger.action("wave", "started")
    logger.action("wave", "completed")
    logger.servo(5, 750, "reached")
    logger.trigger("face_detection", {"confidence": 0.95})
