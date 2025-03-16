import cv2 as cv
import numpy as np
import json
import pandas as pd
import time

# Load YOLO model (Replace paths with actual YOLO model paths)
# Replace with your trained model file
YOLO_WEIGHTS = "./models/yolo/best.onnx"
YOLO_CONFIDENCE_THRESHOLD = 0.5

# Real-world diameter of the softball in mm (adjust accordingly)
REAL_DIAMETER_MM = 97.0  # Standard softball size (change if needed)

# CSV file to save collected data
DATASET_FILE = "ball_distance_dataset.csv"


def load_calibration(calibration_file="calibration_data.json"):
    """Loads camera calibration data from a JSON file."""
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return np.array(calibration_data["mtx"]), np.array(calibration_data["dist"])


def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using preloaded calibration data."""
    return cv.undistort(image, camera_matrix, dist_coeffs)


def load_yolo_model():
    """Loads the YOLO model using ONNX."""
    net = cv.dnn.readNetFromONNX(YOLO_WEIGHTS)
    return net


def detect_softball(frame, net):
    """Runs YOLO object detection on a frame to detect a softball."""
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    height, width, channels = frame.shape
    blob = cv.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    detected_balls = []

    for detection in outputs[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > YOLO_CONFIDENCE_THRESHOLD:
            # Get bounding box
            center_x, center_y, w, h = (
                detection[:4] * np.array([width, height, width, height])).astype(int)

            detected_balls.append(
                (center_x, center_y, w))  # (X, Y, Diameter)

    return detected_balls


def save_data(x, y, pixel_diameter, real_diameter, true_distance):
    """Saves data to a CSV file."""
    data = pd.DataFrame([[x, y, pixel_diameter, real_diameter, true_distance]],
                        columns=["Ball X (px)", "Ball Y (px)", "Pixel Diameter (px)", "Real Diameter (mm)", "True Distance (mm)"])

    try:
        existing_data = pd.read_csv(DATASET_FILE)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    except FileNotFoundError:
        updated_data = data  # Create new file if none exists

    updated_data.to_csv(DATASET_FILE, index=False)
    print(
        f"Data saved: X={x}, Y={y}, Diameter={pixel_diameter}, Distance={true_distance}mm")


def collect_data():
    """Runs the video capture, undistorts frames, detects the ball, and logs data."""
    camera_matrix, dist_coeffs = load_calibration()
    yolo_net = load_yolo_model()

    cap = cv.VideoCapture(0)

    print("Press 'Enter' to log detected ball data, or 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame
        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)

        # Detect the softball
        detected_balls = detect_softball(undistorted_frame, yolo_net)

        for (x, y, w) in detected_balls:
            # Draw detection
            cv.circle(undistorted_frame, (x, y),  w//2, (0, 255, 0), 2)
            cv.putText(undistorted_frame, f"Size: {w}px", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow("Undistorted Live Feed - Ball Detection", undistorted_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('\r') and detected_balls:
            # Log data for the first detected ball
            (x, y, pixel_diameter) = detected_balls[0]

            # Manually enter the actual measured distance
            true_distance = input("Enter the true distance (mm): ")
            try:
                true_distance = float(true_distance)
                save_data(x, y, pixel_diameter,
                          REAL_DIAMETER_MM, true_distance)
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    collect_data()
