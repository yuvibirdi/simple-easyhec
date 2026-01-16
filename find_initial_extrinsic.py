#!/usr/bin/env python3
"""
Interactive tool to find a good initial extrinsic guess for SO100 calibration.

This script loads your captured images and lets you manually adjust the camera position
until the rendered robot roughly matches the real robot in the images.

Usage:
    conda activate sam
    python find_initial_extrinsic.py --output-dir results/so100_fixed/stone_home
"""

import numpy as np
import cv2
import json
from pathlib import Path
import argparse
from transforms3d.euler import euler2mat, mat2euler
from urchin import URDF
from easyhec import ROBOT_DEFINITIONS_DIR
from easyhec.utils.utils_3d import merge_meshes
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv
from easyhec.utils import visualization
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/so100_fixed/stone_home", help="Directory with calibration data")
    parser.add_argument("--camera-name", default="base_camera", help="Camera name")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    # Load data
    print("Loading calibration data...")
    link_poses_dataset = np.load(output_path / "link_poses_dataset.npy")
    image_dataset = np.load(output_path / "image_dataset.npy", allow_pickle=True).reshape(-1)[0]
    images = image_dataset[args.camera_name]

    # Load intrinsics
    intrinsic_json = Path("camera_intrinsic.json")
    if intrinsic_json.exists():
        with open(intrinsic_json, 'r') as f:
            data = json.load(f)
        intrinsic = np.array([
            [data['fx'], 0, data['cx']],
            [0, data['fy'], data['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        intrinsic = np.load(output_path / args.camera_name / "camera_intrinsic.npy")

    # Load robot meshes
    robot_def_path = ROBOT_DEFINITIONS_DIR / "so100"
    robot_urdf = URDF.load(str(robot_def_path / "so100.urdf"))
    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        meshes.append(merge_meshes(link_meshes))

    print("\n" + "="*80)
    print("INITIAL EXTRINSIC FINDER")
    print("="*80)
    print("\nBased on your visualizations, the camera appears to be:")
    print("  - Looking DOWN at the robot from above")
    print("  - Positioned to the upper-left of the robot base")
    print("  - About 40-60cm above the table")
    print("\nLet me suggest some better initial guesses to try...\n")

    # Different initial guesses to try based on common camera placements
    suggested_guesses = {
        "top_left_high": {
            "position": np.array([-0.3, 0.15, 0.6]),  # 30cm back, 15cm left, 60cm up
            "rotation": euler2mat(0, np.pi/3, np.pi/6),  # pitch down 60deg, yaw left 30deg
            "description": "Camera high above, looking down-left"
        },
        "top_left_low": {
            "position": np.array([-0.25, 0.1, 0.4]),  # 25cm back, 10cm left, 40cm up
            "rotation": euler2mat(0, np.pi/4, np.pi/8),  # pitch down 45deg, yaw left 22deg
            "description": "Camera medium height, looking down-left"
        },
        "top_center": {
            "position": np.array([-0.35, 0.0, 0.5]),  # 35cm back, centered, 50cm up
            "rotation": euler2mat(0, np.pi/3, 0),  # pitch down 60deg, no yaw
            "description": "Camera centered above robot"
        },
        "eagles_eye": {
            "position": np.array([-0.2, 0.0, 0.7]),  # 20cm back, centered, 70cm up (very high)
            "rotation": euler2mat(0, np.pi/2.5, 0),  # pitch down steep, no yaw
            "description": "Very high eagle's eye view"
        },
    }

    print("Suggested initial extrinsic guesses to try:\n")
    for i, (name, config) in enumerate(suggested_guesses.items(), 1):
        print(f"{i}. {name}: {config['description']}")
        print(f"   Position (ROS): {config['position']}")
        print(f"   Rotation: pitch={np.rad2deg(mat2euler(config['rotation'])[1]):.1f}°, "
              f"yaw={np.rad2deg(mat2euler(config['rotation'])[2]):.1f}°")
        print()

    print("Generating test visualizations for each guess...")
    print("This will create PNG files showing how well each guess matches your images.\n")

    # Create test visualizations for each guess
    test_output_dir = output_path / "initial_guess_tests"
    test_output_dir.mkdir(exist_ok=True)

    for name, config in suggested_guesses.items():
        print(f"Testing {name}...")

        # Create extrinsic matrix in ROS convention
        extrinsic_ros = np.eye(4)
        extrinsic_ros[:3, :3] = config['rotation']
        extrinsic_ros[:3, 3] = config['position']

        # Convert to OpenCV convention for rendering
        extrinsic_opencv = ros2opencv(extrinsic_ros)

        # Create visualization
        viz_path = str(test_output_dir / f"test_{name}.png")
        visualization.visualize_extrinsic_results(
            images=images[:3],  # Only visualize first 3 poses
            link_poses_dataset=link_poses_dataset[:3],
            meshes=meshes,
            intrinsic=intrinsic,
            extrinsics=np.stack([extrinsic_opencv]),
            masks=None,
            labels=[f"Test: {name}"],
            output_dir=str(test_output_dir),
        )

        # Rename the generated files
        for i in range(min(3, len(images))):
            src = test_output_dir / f"{i}.png"
            dst = test_output_dir / f"test_{name}_pose{i}.png"
            if src.exists():
                src.rename(dst)

        print(f"  ✓ Saved to {test_output_dir / f'test_{name}_pose*.png'}")

        # Save the extrinsic matrices
        np.save(test_output_dir / f"extrinsic_opencv_{name}.npy", extrinsic_opencv)
        np.save(test_output_dir / f"extrinsic_ros_{name}.npy", extrinsic_ros)

        # Print code to use this guess
        print(f"\n  To use this guess, add to so100_fixed.py:")
        print(f"  initial_extrinsic_guess[:3, :3] = euler2mat(0, {mat2euler(config['rotation'])[1]}, {mat2euler(config['rotation'])[2]})")
        print(f"  initial_extrinsic_guess[:3, 3] = np.array({list(config['position'])})")
        print()

    print("="*80)
    print(f"✅ Test visualizations saved to {test_output_dir}/")
    print("\nNext steps:")
    print("1. Look at the test_*.png images in the output directory")
    print("2. Find which initial guess has the robot overlay closest to reality")
    print("3. Update the initial_extrinsic_guess in so100_fixed.py with that configuration")
    print("4. Re-run the calibration")
    print("\nIf none of these guesses are close, you may need to:")
    print("  - Measure the actual camera position relative to robot base")
    print("  - Use a ruler/tape measure to get rough position (x, y, z)")
    print("  - Estimate the camera tilt angle")

if __name__ == "__main__":
    main()
