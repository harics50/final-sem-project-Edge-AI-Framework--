import cv2
import time
from ultralytics import YOLO

# Load trained model
model = YOLO("best9.pt")   # or best9.onnx if using ONNX

# Pi stream URL
PI_STREAM = "http://192.168.29.188:5000"

cap = cv2.VideoCapture(PI_STREAM)

# Reduce buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#  Create large window
cv2.namedWindow("Crack Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Crack Detection", 1280, 720)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    #  Skip old frames to reduce lag
    for _ in range(2):
        cap.read()

    # inference
    results = model(frame, imgsz=416, conf=0.5, verbose=False)

    # Draw results
    frame = results[0].plot()

    #FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display title
    cv2.putText(frame, "Real-Time Crack Detection", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Crack Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
