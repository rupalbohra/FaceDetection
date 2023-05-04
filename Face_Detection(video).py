import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# Choose an image to detect faces in
# img = cv2.imread('rdj.webp')
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (randrange(256), randrange(256),randrange(256)), 5)
    #                                                            color of rect         width of rect


    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q/q is pressed
    if key == 81 or key ==113: # 81 = Q and 113 = q (ASCII value)
        break

# Release the Video Capture object
webcam.release()


print("Code Completed")
