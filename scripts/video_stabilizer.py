#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import logging

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from vidstab import VidStab
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoStabilizer:
    """Class to handle video stabilization for LeRobotDataset"""
    
    def __init__(self, smoothing_window: int = 30, max_shift_percent: float = 0.2):
        """
        Initialize the video stabilizer
        
        Args:
            smoothing_window: Window size for smoothing transformations
            max_shift_percent: Maximum shift as percentage of frame size
        """
        self.smoothing_window = smoothing_window
        self.max_shift_percent = max_shift_percent
        self.stabilizer = VidStab()
    
    def convert_av1_to_h264(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert AV1 video to H264 format for compatibility with vidstab
        
        Args:
            input_path: Path to input AV1 video
            output_path: Path to output H264 video
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {input_path} to H264: {e}")
            return False
    
    def stabilize_video(self, input_path: Path, output_path: Path) -> bool:
        """
        Stabilize a video file
        
        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            
        Returns:
            True if stabilization successful, False otherwise
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_h264 = Path(temp_dir) / "temp_h264.mp4"
                temp_stabilized = Path(temp_dir) / "temp_stabilized.avi"
                
                # Convert to H264 if needed
                if not self.convert_av1_to_h264(input_path, temp_h264):
                    return False
                
                # Stabilize video
                self.stabilizer.stabilize(
                    input_path=str(temp_h264),
                    output_path=str(temp_stabilized),
                    smoothing_window=self.smoothing_window,
                )
                
                # Convert back to MP4
                cmd = [
                    'ffmpeg', '-y', '-i', str(temp_stabilized),
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
            return True
        except Exception as e:
            logger.error(f"Failed to stabilize {input_path}: {e}")
            return False


def stabilize_dataset(
    input_repo_id: str,
    output_dir: str,
    episodes: Optional[List[int]] = None,
    camera_keys: Optional[List[str]] = None,
    output_repo_id: Optional[str] = None,
    push_to_hub: bool = False,
    private: bool = False,
    smoothing_window: int = 30,
    max_shift_percent: float = 0.2
) -> None:
    """
    Stabilize videos in a LeRobotDataset and create a new dataset
    
    Args:
        input_repo_id: Repository ID or local path of input dataset
        output_dir: Directory to save the stabilized dataset
        episodes: List of episode indices to process (None for all)
        camera_keys: List of camera keys to stabilize (None for all image observations)
        output_repo_id: Repository ID for pushing to hub
        push_to_hub: Whether to push the result to Hugging Face Hub
        private: Whether the hub repository should be private
        smoothing_window: Window size for smoothing transformations
        max_shift_percent: Maximum shift as percentage of frame size
    """
    
    # Load the input dataset
    logger.info(f"Loading dataset from {input_repo_id}")
    dataset = LeRobotDataset(
        repo_id=input_repo_id,
        episodes=episodes,
        download_videos=True,
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy dataset structure
    logger.info("Copying dataset structure...")
    
    # Copy metadata and other files
    for item in dataset.meta.root.iterdir():
        if item.name != "videos":
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
            else:
                shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
    
    # Create videos directory structure
    videos_dir = output_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Determine camera keys to process
    if camera_keys is None:
        # Find all image observation keys
        sample = dataset[0]
        camera_keys = [key for key in sample.keys() if key.startswith("observation.images.")]
    
    logger.info(f"Processing camera keys: {camera_keys}")
    
    # Initialize stabilizer
    stabilizer = VideoStabilizer(
        smoothing_window=smoothing_window,
        max_shift_percent=max_shift_percent
    )
    
    # Process each episode and camera
    episodes_to_process = episodes if episodes is not None else range(len(dataset.meta.episodes))
    
    for episode_idx in episodes_to_process:
        logger.info(f"Processing episode {episode_idx}")
        
        for camera_key in camera_keys:
            logger.info(f"  Processing camera: {camera_key}")
            
            # Get original video path
            try:
                original_video_path = dataset.meta.root / dataset.meta.get_video_file_path(episode_idx, camera_key)
                
                # Create output video path
                relative_video_path = dataset.meta.get_video_file_path(episode_idx, camera_key)
                output_video_path = videos_dir / relative_video_path
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Stabilize video
                if stabilizer.stabilize_video(original_video_path, output_video_path):
                    logger.info(f"    Stabilized: {relative_video_path}")
                else:
                    logger.warning(f"    Failed to stabilize: {relative_video_path}")
                    # Copy original if stabilization fails
                    shutil.copy2(original_video_path, output_video_path)
                    
            except Exception as e:
                logger.error(f"    Error processing {camera_key} for episode {episode_idx}: {e}")
    
    logger.info(f"Dataset stabilization complete. Saved to: {output_path}")
    
    # Push to hub if requested
    if push_to_hub and output_repo_id:
        logger.info(f"Pushing dataset to hub: {output_repo_id}")
        try:
            api = HfApi()
            api.create_repo(
                repo_id=output_repo_id,
                exist_ok=True,
                private=private,
                repo_type="dataset"
            )
            
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=output_repo_id,
                repo_type="dataset"
            )
            
            logger.info(f"Successfully pushed to: https://huggingface.co/datasets/{output_repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Stabilize videos in a LeRobotDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stabilize all episodes and cameras
  python stabilize_dataset.py --input_repo_id "lerobot/dummy_dataset" --output_dir "./stabilized"
  
  # Stabilize specific episodes and cameras
  python stabilize_dataset.py \\
    --input_repo_id "lerobot/dummy_dataset" \\
    --output_dir "./stabilized" \\
    --episodes 0,1,2 \\
    --camera_keys "observation.images.right_view,observation.images.left_view"
  
  # Stabilize and push to hub
  python stabilize_dataset.py \\
    --input_repo_id "lerobot/dummy_dataset" \\
    --output_dir "./stabilized" \\
    --output_repo_id "your_username/stabilized_dataset" \\
    --push_to_hub
        """
    )
    
    parser.add_argument(
        "--input_repo_id",
        type=str,
        required=True,
        help="Repository ID or local path of input dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the stabilized dataset"
    )
    
    parser.add_argument(
        "--episodes",
        type=str,
        help="Comma-separated list of episode indices to process (e.g., '0,1,2')"
    )
    
    parser.add_argument(
        "--camera_keys",
        type=str,
        help="Comma-separated list of camera keys to stabilize (e.g., 'observation.images.right_view,observation.images.left_view')"
    )
    
    parser.add_argument(
        "--output_repo_id",
        type=str,
        help="Repository ID for pushing to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the stabilized dataset to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the hub repository private"
    )
    
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=30,
        help="Window size for smoothing transformations (default: 30)"
    )
    
    parser.add_argument(
        "--max_shift_percent",
        type=float,
        default=0.2,
        help="Maximum shift as percentage of frame size (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Parse episodes
    episodes = None
    if args.episodes:
        episodes = [int(x.strip()) for x in args.episodes.split(",")]
    
    # Parse camera keys
    camera_keys = None
    if args.camera_keys:
        camera_keys = [x.strip() for x in args.camera_keys.split(",")]
    
    # Validate push to hub arguments
    if args.push_to_hub and not args.output_repo_id:
        parser.error("--output_repo_id is required when --push_to_hub is specified")
    
    # Run stabilization
    stabilize_dataset(
        input_repo_id=args.input_repo_id,
        output_dir=args.output_dir,
        episodes=episodes,
        camera_keys=camera_keys,
        output_repo_id=args.output_repo_id,
        push_to_hub=args.push_to_hub,
        private=args.private,
        smoothing_window=args.smoothing_window,
        max_shift_percent=args.max_shift_percent
    )


if __name__ == "__main__":
    main()