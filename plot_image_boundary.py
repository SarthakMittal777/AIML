import cv2

# Open default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Capture a frame
ret, frame = cap.read()

# Release the camera
cap.release()

# Save the captured image
cv2.imwrite("captured_image.jpg", frame)

# Display the captured image
cv2.imshow("Captured Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image and convert it to grayscale
img = cv2.imread(r"captured_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around each detected face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the image
img = cv2.imread(r"captured_image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours to find the shirt and tie
for contour in contours:
    # Find the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check if the contour is the shirt or the tie
    if h > w * 1.5:
        # This is likely the shirt
        color = np.mean(img[y:y+h, x:x+w], axis=(0, 1))
        print("Shirt color: ", color)
    elif w > h * 1.5:
        # This is likely the tie
        color = np.mean(img[y:y+h, x:x+w], axis=(0, 1))
        print("Tie color: ", color)

# Display the image with the detected regions outlined
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('image', img)
#########################################
cv2.imwrite("traced_image.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

