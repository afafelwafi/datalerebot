#!/usr/bin/env python3
"""
Generate a dummy LeRobot dataset locally with option to push to Hugging Face Hub.

Usage:
    python generate_dummy_dataset.py --repo-id your-username/dataset-name [--push-to-hub] [--num-frames 10]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from io import BytesIO

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
except ImportError:
    print("Error: lerobot package not found. Please install it first.")
    sys.exit(1)


def download_sample_image(width: int = 640, height: int = 480, timeout: int = 10) -> np.ndarray:
    """Download a random sample image from picsum.photos."""
    try:
        response = requests.get(
            f"https://picsum.photos/{width}/{height}", 
            timeout=timeout
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return np.array(image)
    except Exception as e:
        print(f"Warning: Failed to download image ({e}). Using synthetic image.")
        # Fallback to synthetic image
        return generate_synthetic_image(width, height)


def generate_synthetic_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic RGB image as fallback."""
    # Create a gradient image with some patterns
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = int(255 * i / height)  # Red gradient
            image[i, j, 1] = int(255 * j / width)   # Green gradient
            image[i, j, 2] = int(255 * (i + j) / (height + width))  # Blue gradient
    
    # Add some noise for variety
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image


def create_dataset_features(image_shape: tuple = (480, 640, 3), state_dim: int = 7, action_dim: int = 7) -> dict:
    """Create feature configuration for the dataset."""
    return {
        "observation.images.camera": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)]
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)]
        }
    }


def clean_existing_dataset(repo_id: str) -> None:
    """Remove existing dataset if it exists."""
    dataset_path = HF_LEROBOT_HOME.joinpath(repo_id)
    if dataset_path.exists():
        print(f"Removing existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)


def generate_dummy_frame(frame_idx: int, use_online_images: bool = True) -> dict:
    """Generate a single dummy frame."""
    # Generate dummy image
    if use_online_images:
        dummy_image = download_sample_image()
    else:
        dummy_image = generate_synthetic_image()
    
    # Add frame index to image for visual differentiation
    if frame_idx < 10:  # Only for first 10 frames to avoid cluttering
        # Add a simple frame counter in the top-left corner
        dummy_image[10:30, 10:50] = [255, 255, 255]  # White background
        # This is a simple way to add visual difference between frames
    
    # Generate dummy robot state (7-DOF robot with some realistic joint angles)
    dummy_robot_state = np.random.uniform(-np.pi, np.pi, 7).astype(np.float32)
    
    # Generate dummy action (small incremental changes)
    dummy_action = np.random.uniform(-0.1, 0.1, 7).astype(np.float32)
    
    return {
        "observation.images.camera": dummy_image,
        "observation.state": dummy_robot_state,
        "action": dummy_action
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dummy LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="Repository ID (e.g., username/dataset-name)")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to Hugging Face Hub")
    parser.add_argument("--task-name", default="dummy_task", help="Task name for the dataset")
    parser.add_argument("--offline-images", action="store_true", help="Use synthetic images instead of downloading")
    parser.add_argument("--force-clean", action="store_true", help="Force remove existing dataset without prompt")
    
    args = parser.parse_args()
    
    print(f"Generating dummy dataset: {args.repo_id}")
    print(f"Number of frames: {args.num_frames}")
    print(f"FPS: {args.fps}")
    print(f"Task name: {args.task_name}")
    print(f"Push to hub: {args.push_to_hub}")
    print(f"Use online images: {not args.offline_images}")
    print("-" * 50)
    
    # Clean existing dataset
    dataset_path = HF_LEROBOT_HOME.joinpath(args.repo_id)
    if dataset_path.exists():
        if args.force_clean:
            clean_existing_dataset(args.repo_id)
        else:
            response = input(f"Dataset {args.repo_id} already exists. Remove it? (y/N): ")
            if response.lower() in ['y', 'yes']:
                clean_existing_dataset(args.repo_id)
            else:
                print("Exiting without creating dataset.")
                return
    
    # Create dataset
    try:
        print("Creating dataset...")
        features = create_dataset_features()
        dataset = LeRobotDataset.create(
            args.repo_id,
            fps=args.fps,
            features=features,
            use_videos=True
        )
        print(f"Dataset created at: {dataset.root}")
        
        # Generate frames
        print(f"Generating {args.num_frames} frames...")
        for i in range(args.num_frames):
            print(f"  Frame {i+1}/{args.num_frames}", end="\r")
            
            frame = generate_dummy_frame(i, use_online_images=not args.offline_images)
            dataset.add_frame(frame, task=args.task_name)
        
        print(f"\nGenerated {args.num_frames} frames successfully")
        
        # Save episode
        print("Saving episode...")
        dataset.save_episode()
        print("Episode saved successfully")
        
        # Push to hub if requested
        if args.push_to_hub:
            print("Pushing to Hugging Face Hub...")
            try:
                dataset.push_to_hub()
                print("Successfully pushed to hub!")
                print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")
            except Exception as e:
                print(f"Error pushing to hub: {e}")
                print("Dataset created locally but not pushed to hub.")
        
        # Print visualization command
        print("\n" + "="*50)
        print("Dataset generation complete!")
        print(f"Local path: {dataset.root}")
        print("\nTo visualize the dataset, run:")
        print(f"python -m lerobot.scripts.visualize_dataset --repo-id {args.repo_id} --episode-index 0")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()