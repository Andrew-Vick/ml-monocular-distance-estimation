from ultralytics import YOLO
import numpy as np
import cv2
import csv
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import json

# Initialize model and other variables
data = []
data_file = './data/softball_data.csv'
snapshot_dir = './data/snapshots'

data_counter = 0
input_distance = ""
snapshot_taken = False
x, y, r = 0, 0, 0

height_of_camera = 1.0  # meters

# Load YOLO model
yolo_model = YOLO("./models/yolo/yolo11l.pt")


def load_calibration(calibration_file="calibration_data.json"):
    """Loads camera calibration data from a JSON file."""
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return np.array(calibration_data["mtx"]), np.array(calibration_data["dist"])


def save_data():
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"Data saved to {data_file}.")


def detect_ball(frame):
    results = yolo_model(frame)
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return None

    # Convert detections to list of indices where class == 32
    filtered = [box for box in detections if int(box.cls[0]) == 32]

    if not filtered:
        return None

    # Pick the detection with highest confidence
    best = max(filtered, key=lambda b: b.conf)
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    radius = (x2 - x1) / 2  # approximate radius

    return int(cx), int(cy), int(radius)


def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using preloaded calibration data."""
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def capture_and_process_data():
    global input_distance, snapshot_taken, x, y, r, data_counter
    cap = cv2.VideoCapture(0)
    camera_matrix, dist_coeffs = load_calibration()

    paused_frame = None  # Holds the frozen frame after snapshot

    while True:
        if not snapshot_taken:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            undistorted_frame = undistort_image(
                frame, camera_matrix, dist_coeffs)

            result = detect_ball(undistorted_frame)
            if result:
                x_circle, y_circle, radius = result
                cv2.circle(frame, (x_circle, y_circle),
                           radius, (0, 255, 255), 2)
                cv2.putText(frame, f"Radius: {radius} px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(frame, f"Position: ({x_circle}, {y_circle})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.putText(frame, f"Distance (input): {input_distance}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            cv2.imshow("Frame", frame)
        else:
            # Display frozen frame for input
            display_frame = paused_frame.copy()
            cv2.putText(display_frame, f"Distance (input): {input_distance}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Frame", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t') and result and not snapshot_taken:
            x, y, r = result
            paused_frame = frame.copy()
            snapshot_taken = True
            print(
                f"Snapshot taken. Paused for input. x={x}, y={y}, radius={r}")

        elif snapshot_taken:
            if key >= ord('0') and key <= ord('9'):
                input_distance += chr(key)
            elif key == ord('.'):
                input_distance += '.'
            elif key == 13:  # Enter
                try:
                    # Save snapshot
                    data_counter += 1
                    snapshot_filename = f"snapshot_{data_counter}.png"
                    snapshot_path = os.path.join(
                        snapshot_dir, snapshot_filename)
                    os.makedirs(snapshot_dir, exist_ok=True)
                    cv2.imwrite(snapshot_path, paused_frame)

                    # Save data
                    distance = float(input_distance)
                    data.append((x, y, r, distance, data_counter))
                    print(
                        f"Data saved: x={x}, y={y}, r={r}, distance={distance}, id={data_counter}")

                    # Reset input state
                    input_distance = ""
                    snapshot_taken = False
                    paused_frame = None
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_data()


capture_and_process_data()
