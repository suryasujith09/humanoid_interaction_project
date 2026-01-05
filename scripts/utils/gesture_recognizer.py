
import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import math


class GestureType(Enum):
    """Supported gesture types"""
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"          # All 5 fingers up ? Hi/Greet
    ROCK_SIGN = "rock_sign"           # Index + Pinky up ? Wave
    PEACE_SIGN = "peace_sign"         # Index + Middle up ? Raise hand
    FIST = "fist"                     # All fingers closed ? Stand
    THUMBS_UP = "thumbs_up"           # Thumb up ? Approve
    BOTH_HANDS_UP = "both_hands_up"   # Both hands raised ? Hands up
    T_POSE = "t_pose"                 # Arms straight to sides ? T-pose
    POINTING = "pointing"             # Index finger pointing ? Point

class GestureRecognizer:
    """
    Real-time hand gesture recognition using MediaPipe
    """
    
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize gesture recognizer
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Finger tip and pip landmarks
        self.finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        self.finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        # State
        self.last_gesture = GestureType.UNKNOWN
        self.gesture_confidence = 0.0

    
    def _is_finger_extended(self, hand_landmarks, finger_idx):
        """
        Check if a finger is extended
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            finger_idx: Index of finger (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
        
        Returns:
            bool: True if finger is extended
        """
        tip = hand_landmarks.landmark[self.finger_tips[finger_idx]]
        pip = hand_landmarks.landmark[self.finger_pips[finger_idx]]
        
        # Special case for thumb (horizontal comparison)
        if finger_idx == 0:
            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
            return tip.x > mcp.x if hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x < 0.5 else tip.x < mcp.x
        
        # For other fingers, compare y-coordinates (tip should be above pip)
        return tip.y < pip.y
    
    def _count_extended_fingers(self, hand_landmarks):
        """
        Count number of extended fingers
        
        Returns:
            list: [thumb, index, middle, ring, pinky] - 1 if extended, 0 if not
        """
        extended = []
        for i in range(5):
            extended.append(1 if self._is_finger_extended(hand_landmarks, i) else 0)
        return extended
    

    
    def _detect_single_hand_gesture(self, hand_landmarks):
        """
        Detect gesture from a single hand
        
        Returns:
            GestureType: Detected gesture
        """
        # Count extended fingers
        fingers = self._count_extended_fingers(hand_landmarks)
        extended_count = sum(fingers)
        
        # Gesture patterns
        if extended_count == 5:
            return GestureType.OPEN_PALM
        
        elif extended_count == 0:
            return GestureType.FIST
        
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb
            return GestureType.THUMBS_UP
        
        elif fingers == [0, 1, 0, 0, 0]:  # Only index
            return GestureType.POINTING
        
        elif fingers == [0, 1, 1, 0, 0]:  # Index + middle
            return GestureType.PEACE_SIGN
        
        elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:  # Index + pinky (with/without thumb)
            return GestureType.ROCK_SIGN
        
        return GestureType.UNKNOWN
    
    def _detect_two_hand_gesture(self, hand_landmarks_list):
        """
        Detect gestures requiring two hands
        
        Args:
            hand_landmarks_list: List of hand landmarks (length 2)
        
        Returns:
            GestureType: Detected gesture or UNKNOWN
        """
        if len(hand_landmarks_list) != 2:
            return GestureType.UNKNOWN
        
        hand1, hand2 = hand_landmarks_list[0], hand_landmarks_list[1]
        
        # Get wrist positions
        wrist1 = hand1.landmark[self.mp_hands.HandLandmark.WRIST]
        wrist2 = hand2.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Get middle finger tips for hand positioning
        tip1 = hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        tip2 = hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Check if both hands are raised (hands up)
        if wrist1.y > tip1.y and wrist2.y > tip2.y:  # Tips above wrists
            fingers1 = self._count_extended_fingers(hand1)
            fingers2 = self._count_extended_fingers(hand2)
            
            # Both palms open and raised
            if sum(fingers1) >= 4 and sum(fingers2) >= 4:
                return GestureType.BOTH_HANDS_UP
        
        # Check for T-pose (arms extended to sides)
        # Hands should be at similar height and far apart horizontally
        vertical_diff = abs(wrist1.y - wrist2.y)
        horizontal_dist = abs(wrist1.x - wrist2.x)
        
        if vertical_diff < 0.15 and horizontal_dist > 0.6:  # Similar height, far apart
            # Check if arms are roughly straight
            shoulder_y = (wrist1.y + wrist2.y) / 2
            if abs(tip1.y - shoulder_y) < 0.2 and abs(tip2.y - shoulder_y) < 0.2:
                fingers1 = self._count_extended_fingers(hand1)
                fingers2 = self._count_extended_fingers(hand2)
                
                # Hands open
                if sum(fingers1) >= 4 and sum(fingers2) >= 4:
                    return GestureType.T_POSE
        
        return GestureType.UNKNOWN
    
    def process_frame(self, frame):
        """
        Process a video frame and detect gestures
        
        Args:
            frame: BGR image from camera
        
        Returns:
            tuple: (gesture_type, confidence, annotated_frame)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        # Default values
        gesture = GestureType.UNKNOWN
        confidence = 0.0
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Detect gestures
            if num_hands == 2:
                # Check for two-hand gestures first
                two_hand_gesture = self._detect_two_hand_gesture(results.multi_hand_landmarks)
                if two_hand_gesture != GestureType.UNKNOWN:
                    gesture = two_hand_gesture
                    confidence = 0.9
                else:
                    # Fallback to single hand gesture (first hand)
                    gesture = self._detect_single_hand_gesture(results.multi_hand_landmarks[0])
                    confidence = 0.7
            
            elif num_hands == 1:
                # Single hand gesture
                gesture = self._detect_single_hand_gesture(results.multi_hand_landmarks[0])
                confidence = 0.8
            
            # Update state
            self.last_gesture = gesture
            self.gesture_confidence = confidence
            
            # Display gesture on frame
            if gesture != GestureType.UNKNOWN:
                cv2.putText(
                    annotated_frame,
                    f"Gesture: {gesture.value}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    annotated_frame,
                    f"Confidence: {confidence:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        return gesture, confidence, annotated_frame
    
    def get_last_gesture(self):
        """Get the last detected gesture"""
        return self.last_gesture, self.gesture_confidence
    
    def cleanup(self):
        """Release resources"""
        self.hands.close()


if __name__ == "__main__":
    print("Testing Gesture Recognizer...")
    print("Show different hand gestures to the camera!")
    print("Press 'q' to quit")
    
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        gesture, confidence, annotated_frame = recognizer.process_frame(frame)
        
        # Display
        cv2.imshow('Gesture Recognition Test', annotated_frame)
        
        # Print detected gesture
        if gesture != GestureType.UNKNOWN:
            print(f"Detected: {gesture.value} (confidence: {confidence:.2f})")
        
        # Quit on 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.cleanup()
