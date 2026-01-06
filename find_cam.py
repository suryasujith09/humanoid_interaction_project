import cv2

print("Searching for active cameras...")
# Check indices 0 through 9
for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[SUCCESS] Camera found at Index: {index}")
            print(f"          Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
        else:
            print(f"[Warning] Index {index} opens, but returns no frames (likely metadata node).")
        cap.release()
    else:
        pass # Index not available
print("Search complete.")
