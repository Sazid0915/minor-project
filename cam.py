import cv2
import time
from datetime import datetime
import argparse
import os

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam
video = cv2.VideoCapture(0)
photo_count = 0

while True:
    check, frame = video.read()
    
    if frame is not None:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

        # Draw rectangle around faces and save images
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if photo_count < 5:
                exact_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
                cv2.imwrite(f"face_detected_{exact_time}.jpg", frame)
                photo_count += 1
        
        # Show the video feed
        cv2.imshow("Home Surveillance", frame)
        
        key = cv2.waitKey(1)

        if key == ord('q'):  # Exit on pressing 'q'
            # Parse command-line arguments
            ap = argparse.ArgumentParser()
            ap.add_argument("-ext", "--extension", required=False, default='jpg')
            ap.add_argument("-o", "--output", required=False, default='output.mp4')
            args = vars(ap.parse_args())

            dir_path = "."
            ext = args['extension']
            output = args['output']

            images = [f for f in os.listdir(dir_path) if f.endswith(ext)]
            
            if images:
                image_path = os.path.join(dir_path, images[0])
                frame = cv2.imread(image_path)
                height, width, channels = frame.shape 

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

                for image in images:
                    image_path = os.path.join(dir_path, image)
                    frame = cv2.imread(image_path)
                    out.write(frame)

                out.release()

            break

# Release resources
video.release()
cv2.destroyAllWindows()
