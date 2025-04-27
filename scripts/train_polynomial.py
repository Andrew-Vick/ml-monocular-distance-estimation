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


def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using preloaded calibration data."""
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def load_data():
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                row.pop()  # Remove the last column (snapshot id)
                data.append(tuple(map(float, row)))
        print(f"Loaded {len(data)} data points from {data_file}.")
    else:
        print("No data file found. Starting fresh.")


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
    # Prepare real datafrom sklearn.neighbors import NearestNeighbors


def predict_distance(x, y, radius):
    X_test = np.array([[x, y, radius]])
    return model.predict(X_test)[0]


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


def plot_error_histogram():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import numpy as np

    data_np = np.array(data)
    X = np.column_stack(
        (data_np[:, 0], data_np[:, 1], data_np[:, 2]))  # u, v, d
    y = data_np[:, 3]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    errors = y_pred - y_test

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Prediction Error (m)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histogram_prediction_errors.png")
    plt.show()


def plot_distance_surface_by_radius():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    data_np = np.array(data)
    u = data_np[:, 0]
    v = data_np[:, 1]

    radius_levels = [5, 10, 15, 20]  # Choose 3 meaningful ball sizes
    u_range = np.linspace(min(u), max(u), 30)
    v_range = np.linspace(min(v), max(v), 30)
    u_grid, v_grid = np.meshgrid(u_range, v_range)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for r_fixed in radius_levels:
        X_features = np.column_stack((
            u_grid.ravel(), v_grid.ravel(), np.full(u_grid.size, r_fixed)
        ))
        Z_pred = model.predict(X_features).reshape(u_grid.shape)
        print(f"Predicting for radius: {r_fixed}, distnce shape: {Z_pred}")
        ax.plot_surface(u_grid, v_grid, Z_pred, alpha=0.7,
                        label=f"r={r_fixed}", cmap='viridis')
        ax.text(u_grid[0, -1], v_grid[-1, 0], Z_pred[-1, -1],
                f"r={r_fixed}", color='black', fontsize=10, weight='bold')

    ax.set_title("Predicted Distance Surface at Fixed Radii")
    ax.set_xlabel("u (image x-pos)")
    ax.set_ylabel("v (image y-pos)")
    ax.set_zlabel("Predicted Distance (m)")
    plt.tight_layout()
    plt.savefig("predicted_distance_surface_by_radius.png")
    plt.show()


load_data()                   # Step 1: Load data from CSV
split_and_train()            # Step 2: Train model and evaluate
# Optional: Visualize distance surface by radius
plot_distance_surface_by_radius()
plot_error_histogram()  # Step 4: Plot histogram of prediction errors
# real_time_prediction()       # Step 3: Live video + prediction
