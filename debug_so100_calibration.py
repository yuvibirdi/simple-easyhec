#!/usr/bin/env python3
"""
Debug script for SO100 calibration issues.
This script helps diagnose why the loss is extremely high (56k vs expected <1k).

Run with: conda activate sam && python debug_so100_calibration.py
"""

import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def check_saved_data(output_dir="results/so100/default"):
    """Check the quality of saved calibration data."""
    output_path = Path(output_dir)

    print("=" * 80)
    print("CHECKING SAVED CALIBRATION DATA")
    print("=" * 80)

    # Check if data exists
    link_poses_path = output_path / "link_poses_dataset.npy"
    image_dataset_path = output_path / "image_dataset.npy"

    if not link_poses_path.exists():
        print(f"❌ Link poses not found at {link_poses_path}")
        return False
    if not image_dataset_path.exists():
        print(f"❌ Image dataset not found at {image_dataset_path}")
        return False

    print(f"✓ Found link poses at {link_poses_path}")
    print(f"✓ Found image dataset at {image_dataset_path}")

    # Load data
    link_poses = np.load(link_poses_path)
    image_dataset = np.load(image_dataset_path, allow_pickle=True).reshape(-1)[0]

    print(f"\nLink poses shape: {link_poses.shape}")
    print(f"  -> {link_poses.shape[0]} poses")
    print(f"  -> {link_poses.shape[1]} links")

    for cam_name, images in image_dataset.items():
        print(f"\nCamera '{cam_name}':")
        print(f"  Images shape: {images.shape}")
        print(f"  Resolution: {images.shape[2]}x{images.shape[1]}")
        print(f"  Data type: {images.dtype}")
        print(f"  Value range: [{images.min()}, {images.max()}]")

    # Check masks
    for cam_name in image_dataset.keys():
        mask_path = output_path / cam_name / "mask.npy"
        if mask_path.exists():
            masks = np.load(mask_path)
            print(f"\nMasks for '{cam_name}':")
            print(f"  Shape: {masks.shape}")
            print(f"  Data type: {masks.dtype}")
            print(f"  Value range: [{masks.min()}, {masks.max()}]")

            # Check mask coverage
            for i, mask in enumerate(masks):
                coverage = (mask > 0).sum() / (mask.shape[0] * mask.shape[1])
                print(f"  Mask {i} coverage: {coverage*100:.2f}%")
                if coverage < 0.01:
                    print(f"    ⚠️  WARNING: Very low coverage!")
                if coverage > 0.5:
                    print(f"    ⚠️  WARNING: Very high coverage (>50%), may include background!")
        else:
            print(f"\n❌ No masks found for '{cam_name}' at {mask_path}")

    return True


def visualize_masks_on_images(output_dir="results/so100/default", cam_name="base_camera"):
    """Visualize the masks overlaid on the images."""
    output_path = Path(output_dir)

    image_dataset_path = output_path / "image_dataset.npy"
    mask_path = output_path / cam_name / "mask.npy"

    if not image_dataset_path.exists() or not mask_path.exists():
        print("Cannot visualize: missing data files")
        return

    image_dataset = np.load(image_dataset_path, allow_pickle=True).reshape(-1)[0]
    images = image_dataset[cam_name]
    masks = np.load(mask_path)

    print(f"\n{'=' * 80}")
    print(f"VISUALIZING MASKS FOR {cam_name}")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, min(len(images), 4), figsize=(16, 8))
    if len(images) == 1:
        axes = axes.reshape(2, 1)

    for i in range(min(len(images), 4)):
        # Original image
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis('off')

        # Mask overlay
        overlay = images[i].copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[masks[i] > 0.5] = [255, 0, 0]  # Red for mask
        blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[1, i].imshow(blended)
        axes[1, i].set_title(f"Mask {i} (coverage: {(masks[i] > 0.5).sum() / masks[i].size * 100:.1f}%)")
        axes[1, i].axis('off')

    plt.tight_layout()
    viz_path = output_path / cam_name / "mask_visualization_debug.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {viz_path}")
    plt.close()


def check_intrinsics(output_dir="results/so100/default", cam_name="base_camera", intrinsics_json_path=None):
    """Check if camera intrinsics are reasonable."""
    print(f"\n{'=' * 80}")
    print("CHECKING CAMERA INTRINSICS")
    print(f"{'=' * 80}")

    output_path = Path(output_dir)

    # Try loading from .npy file if it exists
    intrinsic_npy_path = output_path / cam_name / "camera_intrinsic.npy"
    if intrinsic_npy_path.exists():
        intrinsic = np.load(intrinsic_npy_path)
        print(f"Loaded from {intrinsic_npy_path}:")
        print(intrinsic)

    # Try loading from JSON if path provided
    if intrinsics_json_path:
        json_path = Path(intrinsics_json_path)
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"\nLoaded from {json_path}:")
            print(f"  fx: {data['fx']}")
            print(f"  fy: {data['fy']}")
            print(f"  cx: {data['cx']}")
            print(f"  cy: {data['cy']}")

            # Check if intrinsics are reasonable
            if abs(data['fx'] - data['fy']) / data['fx'] > 0.1:
                print("  ⚠️  WARNING: fx and fy differ by >10%, unusual for most cameras")
            if data['cx'] < 100 or data['cx'] > 1180 or data['cy'] < 100 or data['cy'] > 620:
                print("  ⚠️  WARNING: Principal point (cx, cy) seems off-center")


def test_optimization_configs(output_dir="results/so100/default", cam_name="base_camera"):
    """Test the optimization with different configurations."""
    from easyhec.optim.optimize import optimize
    from urchin import URDF
    from easyhec import ROBOT_DEFINITIONS_DIR
    from easyhec.utils.utils_3d import merge_meshes
    import json

    print(f"\n{'=' * 80}")
    print("TESTING OPTIMIZATION CONFIGURATIONS")
    print(f"{'=' * 80}")

    output_path = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    link_poses_dataset = np.load(output_path / "link_poses_dataset.npy")
    image_dataset = np.load(output_path / "image_dataset.npy", allow_pickle=True).reshape(-1)[0]
    images = image_dataset[cam_name]
    masks = np.load(output_path / cam_name / "mask.npy")

    # Load intrinsics - try JSON first, then .npy
    intrinsic = None
    json_path = output_path / cam_name / "camera_intrinsic.json"
    npy_path = output_path / cam_name / "camera_intrinsic.npy"

    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        intrinsic = np.array([
            [data['fx'], 0, data['cx']],
            [0, data['fy'], data['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        print(f"Loaded intrinsics from {json_path}")
    elif npy_path.exists():
        intrinsic = np.load(npy_path)
        print(f"Loaded intrinsics from {npy_path}")
    else:
        print("❌ No intrinsics found!")
        return

    # Load robot URDF and meshes
    robot_def_path = ROBOT_DEFINITIONS_DIR / "so100"
    robot_urdf = URDF.load(str(robot_def_path / "so100.urdf"))
    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        meshes.append(merge_meshes(link_meshes))

    # Initial extrinsic guess from so100.py
    from transforms3d.euler import euler2mat
    from easyhec.utils.camera_conversions import ros2opencv

    initial_extrinsic_guess = np.eye(4)
    initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
    initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.5])
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

    camera_width = images.shape[2]
    camera_height = images.shape[1]

    print(f"\nData loaded:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Link poses: {link_poses_dataset.shape}")
    print(f"  Intrinsic matrix shape: {intrinsic.shape}")

    # Test configurations
    configs = [
        {"name": "Current (batch_size=4)", "batch_size": 4, "iterations": 100},
        {"name": "Paper style (no batching)", "batch_size": None, "iterations": 100},
    ]

    results = []

    for config in configs:
        print(f"\n{'─' * 60}")
        print(f"Testing: {config['name']}")
        print(f"{'─' * 60}")

        torch.cuda.empty_cache()

        result = optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            masks=torch.from_numpy(masks).float().to(device),
            link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
            initial_extrinsic_guess=torch.tensor(initial_extrinsic_guess).float().to(device),
            meshes=meshes,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_mount_poses=None,
            gt_camera_pose=None,
            iterations=config['iterations'],
            early_stopping_steps=50,
            batch_size=config['batch_size'],
            verbose=True,
        )

        results.append({
            "config": config['name'],
            "extrinsic": result.cpu().numpy()
        })

    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    for r in results:
        print(f"\n{r['config']}:")
        print(f"Extrinsic:\n{r['extrinsic']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug SO100 calibration")
    parser.add_argument("--output-dir", default="results/so100/default", help="Output directory")
    parser.add_argument("--camera-name", default="base_camera", help="Camera name")
    parser.add_argument("--intrinsics-json", default=None, help="Path to camera intrinsics JSON")
    parser.add_argument("--test-optimization", action="store_true", help="Run optimization tests")
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")

    args = parser.parse_args()

    # Run diagnostics
    print("🔍 Starting SO100 calibration diagnostics...\n")

    data_ok = check_saved_data(args.output_dir)

    if data_ok:
        visualize_masks_on_images(args.output_dir, args.camera_name)
        check_intrinsics(args.output_dir, args.camera_name, args.intrinsics_json)

        if args.test_optimization or args.all:
            test_optimization_configs(args.output_dir, args.camera_name)

    print("\n✅ Diagnostics complete!")
