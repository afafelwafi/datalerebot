#!/usr/bin/env python3
"""
Generate a dummy LeRobot dataset locally with option to push to Hugging Face Hub.

Usage:
    generator = DummyDatasetGenerator()
    generator.generate_dataset(
        repo_id="your-username/dataset-name",
        num_frames=10,
        push_to_hub=False
    )
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from io import BytesIO

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


class DummyDatasetGenerator:
    """A class to generate dummy LeRobot datasets."""
    
    def __init__(self):
        self.default_width = 640
        self.default_height = 480
        self.default_state_dim = 7
        self.default_action_dim = 7
        self.default_fps = 10
        self.default_task_name = "dummy_task"
    
    def download_sample_image(self, width: int = None, height: int = None, timeout: int = 10) -> np.ndarray:
        """Download a random sample image from picsum.photos."""
        width = width or self.default_width
        height = height or self.default_height
        
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
            return self.generate_synthetic_image(width, height)

    def generate_synthetic_image(self, width: int = None, height: int = None) -> np.ndarray:
        """Generate a synthetic RGB image as fallback."""
        width = width or self.default_width
        height = height or self.default_height
        
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

    def create_dataset_features(self, image_shape: tuple = None, state_dim: int = None, action_dim: int = None) -> dict:
        """Create feature configuration for the dataset."""
        image_shape = image_shape or (self.default_height, self.default_width, 3)
        state_dim = state_dim or self.default_state_dim
        action_dim = action_dim or self.default_action_dim
        
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

    def clean_existing_dataset(self, repo_id: str) -> None:
        """Remove existing dataset if it exists."""
        dataset_path = HF_LEROBOT_HOME.joinpath(repo_id)
        if dataset_path.exists():
            print(f"Removing existing dataset at {dataset_path}")
            shutil.rmtree(dataset_path)

    def generate_dummy_frame(self, frame_idx: int, use_online_images: bool = True) -> dict:
        """Generate a single dummy frame."""
        # Generate dummy image
        if use_online_images:
            dummy_image = self.download_sample_image()
        else:
            dummy_image = self.generate_synthetic_image()
        
        # Add frame index to image for visual differentiation
        if frame_idx < 10:  # Only for first 10 frames to avoid cluttering
            # Add a simple frame counter in the top-left corner
            dummy_image[10:30, 10:50] = [255, 255, 255]  # White background
            # This is a simple way to add visual difference between frames
        
        # Generate dummy robot state (7-DOF robot with some realistic joint angles)
        dummy_robot_state = np.random.uniform(-np.pi, np.pi, self.default_state_dim).astype(np.float32)
        
        # Generate dummy action (small incremental changes)
        dummy_action = np.random.uniform(-0.1, 0.1, self.default_action_dim).astype(np.float32)
        
        return {
            "observation.images.camera": dummy_image,
            "observation.state": dummy_robot_state,
            "action": dummy_action
        }

    def generate_dataset(self, 
                        repo_id: str,
                        num_frames: int = 10,
                        fps: int = None,
                        push_to_hub: bool = False,
                        task_name: str = None,
                        offline_images: bool = False,
                        force_clean: bool = False,
                        image_shape: tuple = None,
                        state_dim: int = None,
                        action_dim: int = None) -> LeRobotDataset:
        """
        Generate a dummy LeRobot dataset.
        
        Args:
            repo_id: Repository ID (e.g., username/dataset-name)
            num_frames: Number of frames to generate
            fps: Frames per second
            push_to_hub: Push dataset to Hugging Face Hub
            task_name: Task name for the dataset
            offline_images: Use synthetic images instead of downloading
            force_clean: Force remove existing dataset without prompt
            image_shape: Shape of images (height, width, channels)
            state_dim: Dimension of robot state
            action_dim: Dimension of robot action
            
        Returns:
            LeRobotDataset: The created dataset
        """
        fps = fps or self.default_fps
        task_name = task_name or self.default_task_name
        
        print(f"Generating dummy dataset: {repo_id}")
        print(f"Number of frames: {num_frames}")
        print(f"FPS: {fps}")
        print(f"Task name: {task_name}")
        print(f"Push to hub: {push_to_hub}")
        print(f"Use online images: {not offline_images}")
        print("-" * 50)
        
        # Clean existing dataset
        dataset_path = HF_LEROBOT_HOME.joinpath(repo_id)
        if dataset_path.exists():
            if force_clean:
                self.clean_existing_dataset(repo_id)
            else:
                response = input(f"Dataset {repo_id} already exists. Remove it? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    self.clean_existing_dataset(repo_id)
                else:
                    print("Exiting without creating dataset.")
                    return None
        
        # Create dataset
        try:
            print("Creating dataset...")
            features = self.create_dataset_features(image_shape, state_dim, action_dim)
            dataset = LeRobotDataset.create(
                repo_id,
                fps=fps,
                features=features,
                use_videos=True
            )
            print(f"Dataset created at: {dataset.root}")
            
            # Generate frames
            print(f"Generating {num_frames} frames...")
            for i in range(num_frames):
                print(f"  Frame {i+1}/{num_frames}", end="\r")
                
                frame = self.generate_dummy_frame(i, use_online_images=not offline_images)
                dataset.add_frame(frame, task=task_name)
            
            print(f"\nGenerated {num_frames} frames successfully")
            
            # Save episode
            print("Saving episode...")
            dataset.save_episode()
            print("Episode saved successfully")
            
            # Push to hub if requested
            if push_to_hub:
                print("Pushing to Hugging Face Hub...")
                try:
                    dataset.push_to_hub()
                    print("Successfully pushed to hub!")
                    print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
                except Exception as e:
                    print(f"Error pushing to hub: {e}")
                    print("Dataset created locally but not pushed to hub.")
            
            # Print visualization command
            print("\n" + "="*50)
            print("Dataset generation complete!")
            print(f"Local path: {dataset.root}")
            print("\nTo visualize the dataset, run:")
            print(f"python -m lerobot.scripts.visualize_dataset --repo-id {repo_id} --episode-index 0")
            
            return dataset
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise e


# Example usage
if __name__ == "__main__":
    generator = DummyDatasetGenerator()
    dataset = generator.generate_dataset(
        repo_id="test/dummy-dataset",
        num_frames=10,
        push_to_hub=False
    )