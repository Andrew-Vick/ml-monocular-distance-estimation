import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from ultralytics import YOLO
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
import cv2

# Initialize model and other variables
data = []
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

data_file = './data/softball_data.csv'

# Load YOLO model
yolo_model = YOLO("./models/yolo/yolo11l.pt")


def load_calibration(calibration_file="calibration_data.json"):
    """Loads camera calibration data from a JSON file."""
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return np.array(calibration_data["mtx"]), np.array(calibration_data["dist"])


# Constants for physics calculation
mtx, dist = load_calibration()
FOCAL_LENGTH_PIXELS = mtx[0, 0]  # f_x from calibration
REAL_DIAMETER_MM = 111  # mm
FOCAL_LENGTH_MM = 3.6  # ASSUMED, APPLE DOES NOT PROVIDE THIS
PIXEL_SIZE_MM = FOCAL_LENGTH_MM / FOCAL_LENGTH_PIXELS


def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using preloaded calibration data."""
    return cv2.undistort(image, camera_matrix, dist_coeffs)


# def load_data():
#     if os.path.exists(data_file):
#         with open(data_file, 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 row.pop()  # Remove the last column (snapshot id)
#                 data.append(tuple(map(float, row)))
#         print(f"Loaded {len(data)} data points from {data_file}.")
#     else:
#         print("No data file found. Starting fresh.")

def load_data():
    global data
    mtx, _ = load_calibration()
    f_pixels = mtx[0, 0]
    pixel_size = FOCAL_LENGTH_MM / f_pixels

    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                row.pop()  # Remove snapshot id
                x, y, r, dist = map(float, row)
                phys_dist = physics_distance(
                    FOCAL_LENGTH_MM, REAL_DIAMETER_MM, r, pixel_size)
                correction = dist - phys_dist
                data.append((x, y, r, phys_dist, correction))  # Extended row
        print(f"Loaded {len(data)} data points from {data_file}.")
    else:
        print("No data file found. Starting fresh.")


# NEW TRAINING CODE USES PHYSICS TO CALCULATE DISTANCE


def fit_hybrid_model():
    if len(data) < 5:
        print("Not enough data.")
        return

    data_np = np.array(data)
    X = data_np[:, 0:3]  # x, y, radius
    y_correction = data_np[:, 4]  # target is the correction

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_correction, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Correction Model - MSE: {mse:.4f}, R²: {r2:.4f}")


def split_and_train():
    if len(data) < 5:
        print("Not enough data to train.")
        return

    data_np = np.array(data)
    X = np.column_stack(
        (data_np[:, 0], data_np[:, 1], data_np[:, 2]))  # x, y, radius
    y = data_np[:, 3]  # distance

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the pipeline model directly
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Plotting predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, c='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test),
             max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Distance")
    plt.ylabel("Predicted Distance")
    plt.title("Actual vs Predicted Distance")
    plt.grid(True)
    plt.savefig("distance_prediction_plot.png")
    plt.show()

    # Plotting residuals
    plt.scatter(y_test, y_pred - y_test)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Distance")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.savefig("residual_plot.png")
    plt.show()

    # Plotting 3D surface
    # Generate a grid over x and y
    x_vals = np.linspace(0, 640, 50)  # image width range
    y_vals = np.linspace(0, 480, 50)  # image height range
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Fixed radius (e.g. average from dataset)
    radius_fixed = np.mean([d[2] for d in data])

    # Flatten and create input array for predictions
    X_input = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, radius_fixed)))
    z_pred = model.predict(X_input).reshape(x_grid.shape)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_pred, cmap='viridis', alpha=0.8)

    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.set_zlabel('Predicted Distance (m)')
    ax.set_title('Predicted Distance Surface (Fixed Radius)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig("3d_surface_fixed_radius.png")
    plt.show()


def physics_distance(focal_length_mm, real_diameter_mm, pixel_radius, pixel_size_mm):
    # Convert diameter to pixel diameter (radius * 2)
    pixel_diameter = pixel_radius * 2
    return (focal_length_mm * real_diameter_mm) / (pixel_diameter * pixel_size_mm)


# def predict_distance(x, y, radius):
#     X_test = np.array([[x, y, radius]])
#     X_test_poly = model.transform(X_test)
#     return model.predict(X_test_poly)[0]

def predict_distance(x, y, radius):
    mtx, _ = load_calibration()
    f_pixels = mtx[0, 0]
    pixel_size = FOCAL_LENGTH_MM / f_pixels

    physics_est = physics_distance(
        FOCAL_LENGTH_MM, REAL_DIAMETER_MM, radius, pixel_size)
    correction = model.predict([[x, y, radius]])[0]
    return physics_est + correction


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


def real_time_prediction():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        result = detect_ball(frame)
        if result:
            x_circle, y_circle, radius = result
            predicted_distance = predict_distance(x_circle, y_circle, radius)

            cv2.circle(frame, (x_circle, y_circle), radius, (0, 255, 255), 2)
            cv2.putText(frame, f"Distance: {predicted_distance:.2f} m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# load_data()                   # Step 1: Load data from CSV
# split_and_train()            # Step 2: Train model and evaluate
# real_time_prediction()       # Step 3: Live video + prediction
load_data()
fit_hybrid_model()
real_time_prediction()
