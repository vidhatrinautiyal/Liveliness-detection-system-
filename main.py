import math
import time

import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.8   # Sets a threshold for the confidence score.
# Only detections with confidence above this value will be considered.

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)   # Sets the width and height of the video frame.

model = YOLO("models/myModel.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0   # Variables to calculate and display the frames per second (FPS).

while True:     # Continuously captures frames from the webcam.
    new_frame_time = time.time()    # Records the current time for FPS calculation.
    success, img = cap.read()       # Reads a frame from the webcam.
    results = model(img, stream=True, verbose=False)  # Runs the yolo model on the captured frame, get detection result
    for r in results:  # Iterates through the detection results.

        # Extracts bounding box coordinates, converts them to integers, and calculates the width and height of the box.
        # Retrieves the confidence score and class index.
        # Checks if the confidence score is above the threshold.

        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
                # Sets the color of the bounding box based on the class name (green for real and red for fake).
                # Draws a rectangular bounding box, displays the class name , confidence score on the frame using cvzone
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    # Calculates the FPS based on the time difference between the current and previous frames.
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)   # Displays the processed frame in a window named "Image".
    cv2.waitKey(1)    # Waits for a key press for 1 millisecond to allow the display to update.

"""
import streamlit as st
import cv2
import math
import time
import cvzone
from ultralytics import YOLO

confidence = 0.8  # Sets a threshold for the confidence score.

model = YOLO("models/myModel.pt")
classNames = ["fake", "real"]

# Initialize camera capture
cap = None


# Function to start the camera
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)  # Sets the width and height of the video frame.


# Function to stop the camera
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None


def main():
    st.title("Live Object Detection with Streamlit")

    # Initialize session state variables
    if 'camera_started' not in st.session_state:
        st.session_state.camera_started = False

    # Button to start the camera
    if st.button("Start Camera"):
        start_camera()
        st.session_state.camera_started = True

    # Button to stop the camera
    if st.button("Stop Camera"):
        stop_camera()
        st.session_state.camera_started = False

    # Create a placeholder for the video
    frame_placeholder = st.empty()

    # Display camera feed and perform object detection
    if cap is not None and st.session_state.camera_started:
        prev_frame_time = 0
        while st.session_state.camera_started:
            new_frame_time = time.time()  # Records the current time for FPS calculation.
            success, img = cap.read()  # Reads a frame from the webcam.
            if not success:
                st.warning("Failed to capture frame from camera.")
                break

            results = model(img, stream=True, verbose=False)  # Runs the YOLO model on the captured frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > confidence:
                        if classNames[cls] == 'real':
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)

                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                           (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                           colorB=color)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            frame_placeholder.image(img, channels="BGR")  # Display the processed frame

            st.write(f"FPS: {fps:.2f}")

            # To avoid blocking the interface
            time.sleep(0.03)


if __name__ == "__main__":
    main()
"""