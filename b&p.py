import cv2
import numpy as np
import os

# ---------------------------------------------
# Load DNN Face Detector (downloads automatically if missing)
# ---------------------------------------------
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"

if not os.path.exists(modelFile) or not os.path.exists(configFile):
    import urllib.request

    print("⏳ Downloading DNN face detection model (one-time only)...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
        configFile,
    )
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        modelFile,
    )

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# ---------------------------------------------
# Helper functions
# ---------------------------------------------
def blur_face(frame, mask):
    blurred = cv2.GaussianBlur(frame, (99, 99), 30)
    return np.where(mask[:, :, np.newaxis] == 255, blurred, frame)

def pixelate_face(frame, mask, pixel_size=15):
    # Downsample and upsample for pixel effect
    temp = cv2.resize(frame, (frame.shape[1] // pixel_size, frame.shape[0] // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask[:, :, np.newaxis] == 255, pixelated, frame)

# ---------------------------------------------
# Start Webcam
# ---------------------------------------------
cap = cv2.VideoCapture(0)
mode = "blur"  # default mode

print("🔵 Press 'b' for BLUR mode | 'p' for PIXELATE mode | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Create oval mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            center = (x1 + (x2 - x1)//2, y1 + (y2 - y1)//2)
            axes = ((x2 - x1)//2, (y2 - y1)//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            # Apply chosen effect
            if mode == "blur":
                frame = blur_face(frame, mask)
            else:
                frame = pixelate_face(frame, mask)

    # Show mode on screen
    cv2.putText(frame, f"Mode: {mode.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("AI Face Filter", frame)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        mode = "blur"
    elif key == ord('p'):
        mode = "pixelate"

cap.release()
cv2.destroyAllWindows()
