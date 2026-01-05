#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Action Matcher - Matches human poses to robot actions
Save as: ~/humanoid_interaction_project/scripts/utils/action_matcher.py

Watches human movement and finds nearest matching robot action
"""

import numpy as np
import mediapipe as mp
from collections import deque
import math


class PoseSignature:
    """Represents a pose signature for matching"""
    
    def __init__(self, landmarks=None):
        self.landmarks = landmarks
        self.features = self._extract_features() if landmarks else None
    
    def _extract_features(self):
        """Extract normalized pose features"""
        if not self.landmarks:
            return None
        
        # Key body points
        features = {}
        
        # Get key landmarks
        left_shoulder = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        left_hip = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        nose = self.landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        
        # Calculate shoulder width for normalization
        shoulder_width = math.sqrt(
            (right_shoulder.x - left_shoulder.x)**2 + 
            (right_shoulder.y - left_shoulder.y)**2
        )
        
        if shoulder_width == 0:
            shoulder_width = 0.1
        
        # Normalize positions relative to center
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Arm angles
        features['left_arm_angle'] = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        features['right_arm_angle'] = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Arm elevation (relative to shoulders)
        features['left_wrist_height'] = (left_wrist.y - left_shoulder.y) / shoulder_width
        features['right_wrist_height'] = (right_wrist.y - right_shoulder.y) / shoulder_width
        
        # Arm spread (T-pose detection)
        features['left_arm_spread'] = (left_wrist.x - center_x) / shoulder_width
        features['right_arm_spread'] = (right_wrist.x - center_x) / shoulder_width
        
        # Body posture
        features['body_lean'] = (nose.x - center_x) / shoulder_width
        
        # Elbow positions
        features['left_elbow_height'] = (left_elbow.y - left_shoulder.y) / shoulder_width
        features['right_elbow_height'] = (right_elbow.y - right_shoulder.y) / shoulder_width
        
        return features
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = [point1.x - point2.x, point1.y - point2.y]
        vector2 = [point3.x - point2.x, point3.y - point2.y]
        
        dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.acos(cos_angle)
        
        return math.degrees(angle)
    
    def distance_to(self, other):
        """Calculate distance to another pose signature"""
        if not self.features or not other.features:
            return float('inf')
        
        total_distance = 0
        count = 0
        
        for key in self.features:
            if key in other.features:
                diff = self.features[key] - other.features[key]
                total_distance += diff ** 2
                count += 1
        
        if count == 0:
            return float('inf')
        
        return math.sqrt(total_distance / count)


class ActionMatcher:
    """
    Matches detected human poses to available robot actions
    """
    
    def __init__(self, logger=None):
        """
        Initialize action matcher
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Predefined action signatures
        self.action_signatures = self._create_action_signatures()
        
        # Pose history for stability
        self.pose_history = deque(maxlen=10)
        self.stable_pose_threshold = 0.15
        
        if self.logger:
            self.logger.info("Action matcher initialized")
    
    def _create_action_signatures(self):
        """Create expected pose signatures for each action"""
        signatures = {}
        
        # Wave - right hand up and moving
        signatures['wave'] = {
            'description': 'Right hand raised above shoulder',
            'features': {
                'right_wrist_height': -1.5,  # Above shoulder
                'right_arm_angle': 140,
                'left_wrist_height': 0.5,
                'right_arm_spread': 0.5
            },
            'tolerance': 0.5
        }
        
        # Greet - both hands up
        signatures['greet'] = {
            'description': 'Both hands raised',
            'features': {
                'left_wrist_height': -1.0,
                'right_wrist_height': -1.0,
                'left_arm_angle': 150,
                'right_arm_angle': 150
            },
            'tolerance': 0.5
        }
        
        # Hands up - surrender pose
        signatures['hands_up'] = {
            'description': 'Both hands straight up',
            'features': {
                'left_wrist_height': -2.0,
                'right_wrist_height': -2.0,
                'left_arm_angle': 170,
                'right_arm_angle': 170,
                'left_arm_spread': 0.2,
                'right_arm_spread': -0.2
            },
            'tolerance': 0.6
        }
        
        # T-pose / Hands straight
        signatures['hands_straight'] = {
            'description': 'Arms extended to sides (T-pose)',
            'features': {
                'left_arm_spread': -3.0,
                'right_arm_spread': 3.0,
                'left_wrist_height': 0.0,
                'right_wrist_height': 0.0,
                'left_arm_angle': 170,
                'right_arm_angle': 170
            },
            'tolerance': 0.7
        }
        
        # Stand - neutral pose
        signatures['stand'] = {
            'description': 'Standing neutral',
            'features': {
                'left_wrist_height': 1.5,
                'right_wrist_height': 1.5,
                'left_arm_spread': -0.5,
                'right_arm_spread': 0.5,
                'body_lean': 0.0
            },
            'tolerance': 0.4
        }
        
        # Raise right hand
        signatures['raise_right_hand'] = {
            'description': 'Right hand raised',
            'features': {
                'right_wrist_height': -1.8,
                'right_arm_angle': 160,
                'left_wrist_height': 1.0,
                'right_arm_spread': 0.3
            },
            'tolerance': 0.5
        }
        
        # Forward pointing
        signatures['forward_one_step'] = {
            'description': 'Pointing forward',
            'features': {
                'right_arm_angle': 170,
                'right_wrist_height': -0.5,
                'body_lean': 0.2
            },
            'tolerance': 0.5
        }
        
        return signatures
    
    def process_frame(self, frame):
        """
        Process frame and detect pose
        
        Args:
            frame: BGR image from camera
        
        Returns:
            tuple: (matched_action, confidence, annotated_frame)
        """
        # Convert to RGB
        image_rgb = frame.copy()
        image_rgb = np.ascontiguousarray(image_rgb[:, :, ::-1])
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)
        
        annotated_frame = frame.copy()
        matched_action = None
        confidence = 0.0
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Create pose signature
            current_pose = PoseSignature(results.pose_landmarks)
            
            # Add to history
            self.pose_history.append(current_pose)
            
            # Match to action if pose is stable
            if len(self.pose_history) >= 5:
                matched_action, confidence = self._match_action(current_pose)
        
        return matched_action, confidence, annotated_frame
    
    def _match_action(self, pose):
        """
        Match current pose to best action
        
        Args:
            pose: PoseSignature object
        
        Returns:
            tuple: (action_name, confidence)
        """
        if not pose.features:
            return None, 0.0
        
        best_action = None
        best_score = float('inf')
        
        for action_name, signature in self.action_signatures.items():
            # Calculate weighted distance
            distance = 0
            feature_count = 0
            
            for feature_key, expected_value in signature['features'].items():
                if feature_key in pose.features:
                    actual_value = pose.features[feature_key]
                    diff = abs(actual_value - expected_value)
                    distance += diff ** 2
                    feature_count += 1
            
            if feature_count > 0:
                distance = math.sqrt(distance / feature_count)
                
                # Check if within tolerance
                if distance < signature['tolerance'] and distance < best_score:
                    best_score = distance
                    best_action = action_name
        
        if best_action:
            # Convert distance to confidence (0-1)
            confidence = max(0, 1 - (best_score / 2))
            return best_action, confidence
        
        return None, 0.0
    
    def get_action_description(self, action_name):
        """Get description of an action"""
        if action_name in self.action_signatures:
            return self.action_signatures[action_name]['description']
        return "Unknown action"
    
    def cleanup(self):
        """Release resources"""
        self.pose.close()


# Test function
if __name__ == "__main__":
    import cv2
    import sys
    import os
    
    sys.path.append(os.path.expanduser('~/humanoid_interaction_project/scripts'))
    from utils.logger import get_logger
    
    logger = get_logger("ActionMatcherTest")
    matcher = ActionMatcher(logger=logger)
    
    print("="*60)
    print("🎯 ACTION MATCHER TEST")
    print("="*60)
    print("\nTry these poses:")
    for action, sig in matcher.action_signatures.items():
        print(f"  • {action}: {sig['description']}")
    print("\nPress 'q' to quit")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        matched_action, confidence, annotated_frame = matcher.process_frame(frame)
        
        # Display match
        if matched_action and confidence > 0.6:
            text = f"MATCH: {matched_action} ({confidence:.2f})"
            cv2.putText(annotated_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            print(f"✓ Matched: {matched_action} - confidence: {confidence:.2f}")
        
        cv2.imshow('Action Matcher Test', annotated_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    matcher.cleanup()
