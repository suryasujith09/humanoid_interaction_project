import cv2
import time

# explicit index 0, explicit backend V4L2
print("Attempting to open Camera 0 with V4L2 backend...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Force common USB camera settings (MJPG is faster/safer)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1) # Warmup

if cap.isOpened():
    print("SUCCESS: Camera opened!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured! Shape: {frame.shape}")
        cv2.imwrite("test_image.jpg", frame)
        print("Saved 'test_image.jpg' - check this file to verify image")
    else:
        print("Camera opened, but failed to return a frame.")
else:
    print("FAILURE: Could not open Camera 0 even with V4L2.")

cap.release()
