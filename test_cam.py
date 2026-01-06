import cv2
print("Testing Camera 0...")
cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("SUCCESS: Camera 0 is working!")
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully.")
    else:
        print("Warning: Camera opened but returned empty frame.")
    cap.release()
else:
    print("FAILURE: Could not open Camera 0.")
