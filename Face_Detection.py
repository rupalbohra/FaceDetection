import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# Choose an image to detect faces in
img = cv2.imread('rdj.webp')

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w, y+h), (randrange(256), randrange(256),randrange(256)), 5)
#                                                            color of rect         width of rect

# Display the image with the faces


# 
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()


print("Code Completed")
