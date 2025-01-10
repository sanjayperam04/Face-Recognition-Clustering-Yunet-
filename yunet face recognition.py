import cv2
import numpy as np

# Initialize the face detector
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))

# Path to the image file
image_path = "Group pics Classmates/IMG_1774.JPG"

# Read the image
frame = cv2.imread(image_path)

# Check if the image is loaded correctly
if frame is None:
    print("Error: Could not read the image.")
    exit()

# Get image dimensions
img_W = int(frame.shape[1])
img_H = int(frame.shape[0])

# Set input size for face detection
detector.setInputSize((img_W, img_H))

# Perform face detection
_, detections = detector.detect(frame)

# Check if detections is a numpy array and has data
if isinstance(detections, np.ndarray) and detections.size > 0:
    for face in detections:
        # Ensure face is a numpy array and has the expected shape
        if isinstance(face, np.ndarray) and face.shape[0] >= 4:
            x, y, w, h = map(int, face[:4])  # Extract face coordinates
            side_length = max(w, h)  # Determine side length of the square
            x1 = x + (w - side_length) // 2
            y1 = y + (h - side_length) // 2
            x2 = x1 + side_length
            y2 = y1 + side_length
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_W, x2), min(img_H, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw square

# Display the resulting image
cv2.imshow('Image Face Detection', frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
