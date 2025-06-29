
from ultralytics import YOLO
YOLO("yolov8m.pt")  # This will download the model automatically

import cv2 # computer vision
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # Ensure this file is in your working directory

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()  # Auto-draws bounding boxes and labels

    # Display the frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

