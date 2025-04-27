from ultralytics import YOLO
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
import cv2
from mpl_toolkits.mplot3d import Axes3D


# Initialize model and other variables
data = []
data_file = './data/softball_data.csv'
yolo_model = YOLO("./models/yolo/yolo11l.pt")


def load_calibration(calibration_file="calibration_data.json"):
    """Loads camera calibration data from a JSON file."""
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return np.array(calibration_data["mtx"]), np.array(calibration_data["dist"])


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


def split_data():
    data_np = np.array(data)
    X = np.column_stack(
        (data_np[:, 0], data_np[:, 1], data_np[:, 2]))  # x, y, radius
    y = data_np[:, 3]  # distance

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def main():
    mlp = MLPRegressor(hidden_layer_sizes=(30, 20, 20, 10, 5), activation='relu',
                       solver='adam', max_iter=2500, random_state=42)
    load_data()
    X_train, X_test, y_train, y_test = split_data()
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    mlp.fit(X_train_scaled, y_train)

    # Predict manually
    y_pred = mlp.predict(X_test_scaled)
    errors = y_pred - y_test

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"R² score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} meters")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} meters")

    # Plotting histogram of absolute errors
    absolute_errors = np.abs(errors)
    plt.figure(figsize=(8, 6))
    plt.hist(absolute_errors, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Histogram of Absolute Prediction Errors")
    plt.xlabel("Absolute Prediction Error (m)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histogram_absolute_prediction_errors.png")
    plt.show()

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
        Z_pred = mlp.predict(X_features).reshape(u_grid.shape)
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


if __name__ == "__main__":
    main()
