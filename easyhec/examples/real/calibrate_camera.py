"""
Chessboard camera calibration script for OpenCV cameras.

This script uses a 9x6 chessboard pattern printed on an A4 page (210mm x 297mm)
to calibrate camera intrinsics. The calibration results are saved as JSON.

Usage:
    python calibrate_camera.py --camera-id 0 --output camera_intrinsic.json

Controls:
    - Press SPACEBAR to capture a calibration image
    - Press 'q' to finish calibration and compute intrinsics
    - Requires at least 10-15 images for good calibration
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class CalibrationArgs:
    """Arguments for camera calibration."""
    
    camera_id: int = 0
    """OpenCV camera device ID (usually 0 for first camera)"""
    
    output: str = "camera_intrinsic.json"
    """Output path for the calibrated intrinsics JSON file"""
    
    width: int = 1280 
    """Camera width"""
    
    height: int = 720 
    """Camera height"""
    
    chessboard_cols: int = 9
    """Number of inner corners in the chessboard pattern (columns)"""
    
    chessboard_rows: int = 6
    """Number of inner corners in the chessboard pattern (rows)"""
    
    square_size_mm: float = 23.33
    """Size of each square in millimeters. For A4 (210mm x 297mm) with 9x6 pattern: 210/9 = 23.33mm"""
    
    min_images: int = 10
    """Minimum number of calibration images required"""


def create_chessboard_object_points(
    rows: int, cols: int, square_size: float
) -> np.ndarray:
    """
    Create 3D object points for chessboard corners.
    
    Args:
        rows: Number of inner corners (rows)
        cols: Number of inner corners (columns)
        square_size: Size of each square in millimeters
    
    Returns:
        Array of shape (rows*cols, 3) with object points in mm
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size  # Convert to millimeters
    return objp


def find_chessboard_corners(
    img: np.ndarray, pattern_size: Tuple[int, int]
) -> Tuple[bool, np.ndarray]:
    """
    Find chessboard corners in an image.
    
    Args:
        img: Input image (grayscale)
        pattern_size: Tuple of (cols, rows) - number of inner corners
    
    Returns:
        Tuple of (success, corners) where corners is the detected corner positions
    """
    ret, corners = cv2.findChessboardCorners(
        img,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ret:
        # Refine corner positions for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def calibrate_camera(
    object_points_list: List[np.ndarray],
    image_points_list: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calibrate camera using collected chessboard images.
    
    Args:
        object_points_list: List of 3D object points for each image
        image_points_list: List of 2D image points (corners) for each image
        image_size: Tuple of (width, height) of the images
    
    Returns:
        Tuple of (camera_matrix, dist_coeffs, reprojection_error)
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None,
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(object_points_list)):
        imgpoints2, _ = cv2.projectPoints(
            object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points_list[i], imgpoints2, cv2.NORM_L2) / len(
            imgpoints2
        )
        total_error += error
    
    mean_error = total_error / len(object_points_list)
    
    return camera_matrix, dist_coeffs, mean_error


def main(args: CalibrationArgs):
    """Main calibration function."""
    print(f"Starting camera calibration with camera ID {args.camera_id}")
    print(f"Chessboard pattern: {args.chessboard_cols}x{args.chessboard_rows} inner corners")
    print(f"Square size: {args.square_size_mm} mm")
    print(f"Camera resolution: {args.width}x{args.height}")
    print("\nControls:")
    print("  - Press SPACEBAR to capture a calibration image")
    print("  - Press 'q' to finish calibration and compute intrinsics")
    print(f"  - Need at least {args.min_images} images for calibration\n")

    # Initialize camera
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")

    # Create object points for the chessboard
    objp = create_chessboard_object_points(
        args.chessboard_rows, args.chessboard_cols, args.square_size_mm
    )
    pattern_size = (args.chessboard_cols, args.chessboard_rows)

    # Storage for calibration data
    object_points_list: List[np.ndarray] = []
    image_points_list: List[np.ndarray] = []
    captured_images: List[np.ndarray] = []

    print("Starting camera feed...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find chessboard corners
        found, corners = find_chessboard_corners(gray, pattern_size)

        # Draw the results
        display_frame = frame.copy()
        if found:
            cv2.drawChessboardCorners(
                display_frame, pattern_size, corners, found
            )
            status_text = f"Chessboard found! ({len(object_points_list)} images captured)"
            color = (0, 255, 0)
        else:
            status_text = f"Searching for chessboard... ({len(object_points_list)} images captured)"
            color = (0, 0, 255)

        # Add status text
        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            display_frame,
            "Press SPACEBAR to capture, 'q' to finish",
            (10, args.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Camera Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # Spacebar
            if found:
                object_points_list.append(objp)
                image_points_list.append(corners)
                captured_images.append(frame.copy())
                print(f"Captured image {len(object_points_list)}")
            else:
                print("Chessboard not found! Please adjust the chessboard position.")

    cap.release()
    cv2.destroyAllWindows()

    # Check if we have enough images
    num_images = len(object_points_list)
    print(f"\nCaptured {num_images} calibration images")
    
    if num_images < args.min_images:
        raise RuntimeError(
            f"Not enough calibration images! Got {num_images}, need at least {args.min_images}"
        )

    # Perform calibration
    print("\nComputing camera calibration...")
    image_size = (args.width, args.height)
    camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(
        object_points_list, image_points_list, image_size
    )

    # Extract intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print(f"\nCalibration Results:")
    print(f"  Focal length (fx, fy): ({fx:.2f}, {fy:.2f})")
    print(f"  Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
    print(f"  Reprojection error: {reprojection_error:.4f} pixels")
    print(f"  (Lower is better, typically < 0.5 is good)")

    # Save intrinsics to JSON
    intrinsics_data = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": args.width,
        "height": args.height,
        "reprojection_error": float(reprojection_error),
        "num_images": num_images,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(intrinsics_data, f, indent=2)
    
    print(f"\nIntrinsics saved to: {output_path}")
    print(f"\nYou can now use this file with so100.py:")
    print(f"  python so100.py --camera-intrinsics-path {output_path}")


if __name__ == "__main__":
    main(tyro.cli(CalibrationArgs))

