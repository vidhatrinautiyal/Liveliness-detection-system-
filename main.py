import cv2
import time
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

confidence_threshold = 0.5

# Get actual class names from model
classNames = model.names

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    new_frame_time = time.time()

    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > confidence_threshold:
                label = classNames[cls]

                # Color (green for person, red otherwise)
                color = (0, 255, 0) if label == "person" else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {int(conf*100)}%",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS calculation
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    cv2.putText(img, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Detection", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
