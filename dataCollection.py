from time import time   # Used to get the current time, which is used for naming files
import cv2  # Open CV library for image processing.
import cvzone  # Library that simplifies working with OpenCV.
from cvzone.FaceDetectionModule import FaceDetector  # Module from cvzone for face detection.


####################################
classID = 1   # 0 is for fake and 1 is for real
outputFolderPath = 'Dataset/DataCollect'  # Directory path to save images and text files.
confidence = 0.8   # threshold for face detection.
save = True
blurThreshold = 30  # Larger is more focus , to determine if an image is blurry.


debug = False
offsetPercentageW = 10
offsetPercentageH = 20   # Offsets to adjust bounding box size
camWidth, camHeight = 640, 480  # Width and height of the camera feed.
floatingPoint = 6  # Precision for normalized values.
####################################


cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)  # Set camera resolution.
detector = FaceDetector()  # instance

while True:    # Infinite loop to continuously capture frames.
    success, img = cap.read()   # Reads a frame from the camera.
    imgOut = img.copy()         # Creates a copy of the frame for output.
    img, bboxs = detector.findFaces(img, draw=False)     # Detects faces in the frame.

    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file
    if bboxs:    # Check if any faces are detected.
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]    # Extract bounding box coordinates.
            score = bbox["score"][0]     # Extract confidence score of detection.

            # ------  Check the score --------
            # Check if detection confidence is above the threshold.
            # Adjust bounding box size by adding offsets to x, y, w, and h.
            # Ensure bounding box coordinates are within valid range (non-negative).
            if score > confidence:

                # ------  Adding an offset to the face Detected --------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3)

                # ------  To avoid values below 0 --------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ------  Find Blurriness --------
                imgFace = img[y:y + h, x:x + w]   # Extract face region from the image
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:  # Compute blurriness of face,Append blur status(True/False)to listBlur
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------  Normalize Values  --------
                # Get image dimensions,Compute center of the bounding box.
                # Normalize bounding box values to a range [0, 1], Append normalized values and class ID to listInfo.
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # print(xcn, ycn, wn, hn)
                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------  Drawing --------
                # Draw bounding box and display score and blur value on the output image.
                # If debug is True, draw additional debug information on the original image.
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                   scale=2, thickness=3)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                       scale=2, thickness=3)

        # ------  To Save --------
        # If save is True and all detected faces are not blurry, save the image and information.
        # Generate a unique filename using the current timestamp.
        # Save the image and write bounding box information to a text file.
        if save:
            if all(listBlur) and listBlur != []:
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                # ------  Save Label Text File  --------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()
# Display the output image with bounding boxes and annotations.
# Wait for 1 millisecond to allow for a smooth frame update.
    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
