import cv2 as cv
import numpy as np
import json
import time

# For best accuracy when calibrating the camera, use an asymmetric checkerboard pattern


def calibrate_camera_live(checkerboard_size=(8, 8), num_images=15, save_file="calibration_data.json", live_feed_buffer=3):
    """
    Calibrates distortion by recording 15 checkerboard images and uses OpenCV's
    calibrate_camera method to write a json file for the camera's distortion

    Parameters:
    - checkerboard_size: the dimensions of the checkerboard
    - num_images: the number of images to measure distortion
    - save_file: name of the file to output the distortion data
    - live_feed_buffer: number of seconds between each data frame can be taken

    """
    # Termination criteria for cornerSubPix
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros(
        ((checkerboard_size[1] - 1) * (checkerboard_size[0] - 1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0] - 1,
                           0:checkerboard_size[1] - 1].T.reshape(-1, 2)

    # 3D points in real-world space
    objpoints = []
    # 2D points in image plane
    imgpoints = []
    # Store captured images for final grid
    captured_images = []

    # Open webcam
    cap = cv.VideoCapture(0)

    print(
        f"Capturing data for camera calibration. Need {num_images} valid checkerboard images with {checkerboard_size[0]}x{checkerboard_size[1]} squares.")

    # To store the most recent captured image
    last_captured_image = None

    # run until all data frames are collected
    while len(objpoints) < num_images:
        # read in the frame
        ret, frame = cap.read()

        # output error if frame can't be read
        if not ret:
            print("Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners (8x6 internal corners)
        ret, corners = cv.findChessboardCorners(
            gray, (checkerboard_size[0] - 1, checkerboard_size[1] - 1), None)

        # If found, calculate the percentage of the frame occupied by the checkerboard
        if ret:

            # Refine and store corners
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(
                frame, (checkerboard_size[0] - 1, checkerboard_size[1] - 1), corners2, ret)
            print(
                f"Captured checkerboard image {len(objpoints)} of {num_images}")

            last_captured_image = frame.copy()
            # Store the captured image
            captured_images.append(last_captured_image)

            # 3-second live feed buffer without capturing
            start_time = time.time()
            while time.time() - start_time < live_feed_buffer:

                # read in frame
                ret, live_frame = cap.read()
                if ret:
                    # Show the live feed on the left side and the last captured image on the right
                    combined = np.hstack(
                        (live_frame, last_captured_image if last_captured_image is not None else live_frame))
                    cv.imshow('Webcam - Live Feed and Captured Image', combined)

                # cancel early when q is typed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    print("Calibration canceled.")
                    cap.release()
                    cv.destroyAllWindows()
                    return

        else:
            # Show the live frame without detecting a valid pattern
            combined = np.hstack(
                (frame, last_captured_image if last_captured_image is not None else frame))
            cv.imshow('Webcam - Live Feed and Captured Image', combined)

        # Press 'q' to quit before collecting all images
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Calibration canceled.")
            break

    # Release the webcam
    cap.release()
    cv.destroyAllWindows()

    if len(objpoints) == num_images:
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save the calibration data in json
        calibration_data = {
            "ret": ret,
            "mtx": mtx.tolist(),
            "dist": dist.tolist(),
            "rvecs": [r.tolist() for r in rvecs],
            "tvecs": [t.tolist() for t in tvecs]
        }

        # write the data to the file
        with open(save_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print(f"Calibration data saved to {save_file}")

        # Display the captured images in a 3x5 grid
        grid_rows = 3
        grid_cols = 5
        grid_img = np.zeros(
            (frame.shape[0] * grid_rows, frame.shape[1] * grid_cols, 3), dtype=np.uint8)

        for i, img in enumerate(captured_images):
            row = i // grid_cols
            col = i % grid_cols
            grid_img[row*frame.shape[0]:(row+1)*frame.shape[0],
                     col*frame.shape[1]:(col+1)*frame.shape[1]] = img

        # Get current screen resolution, possibly can be changed by getting the screen res
        screen_res = 1280, 720

        # Resize the grid_img to fit within the screen resolution
        scale_width = screen_res[0] / grid_img.shape[1]
        scale_height = screen_res[1] / grid_img.shape[0]
        scale = min(scale_width, scale_height)

        window_width = int(grid_img.shape[1] * scale)
        window_height = int(grid_img.shape[0] * scale)

        # Resize the grid image to fit the window while maintaining ratio
        grid_img_resized = cv.resize(
            grid_img, (window_width, window_height), interpolation=cv.INTER_AREA)

        # Show the resized grid of all captured images
        cv.imshow('Captured Images Grid', grid_img_resized)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Calibration was not completed: not enough valid images.")


def undistort_image(image, calibration_file="calibration_data.json"):
    """
    Undistorts an image using calibration data from a json file.

    Parameters:
    - image: the input image to undistort
    - calibration_file: path to the JSON file containing the calibration data

    Returns:
    - undistorted_image: The undistorted image
    """
    # Load calibration data
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    # Extract camera matrix and distortion data
    mtx = np.array(calibration_data["mtx"])
    dist = np.array(calibration_data["dist"])

    # Undistort the image
    undistorted_image = cv.undistort(image, mtx, dist)

    # return data
    return undistorted_image


def undistort_video(calibration_file="./calibration_data.json"):
    """
    Undistorts each frame of a video and saves annotated frames.

    Parameters:
    - video_file: Path to the input video file
    - calibration_file: Path to JSON containing camera calibration data
    - output_directory: Directory to save undistorted frames
    - distance_log: Dictionary with {frame_number: distance_mm} logged manually

    Returns:
    - None (Frames are saved to output_directory)
    """
    # Load calibration data
    with open(calibration_file, 'r') as f:
        calib = json.load(f)
    camera_matrix = np.array(calib["mtx"])
    dist_coeffs = np.array(calib["dist"])

    # Open video file
    cap = cv.VideoCapture(0)

    frames_saved = 0
    save_frame = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort frame
        undistorted_frame = cv.undistort(frame, camera_matrix, dist_coeffs)

        cv.imshow('Undistorted Frame', undistorted_frame)

        # Check for key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('\r'):
            save_frame = True

        # Save the frame if the Return button was pressed
        if save_frame:
            frames_saved += 1
            output_path = f"./data/undistorted_frame_{frames_saved}.png"
            if cv.imwrite(output_path, undistorted_frame):
                print(f"Saved {output_path}")
                save_frame = False

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    undistort_video()
    # calibrate_camera_live()
