#!/usr/bin/env python3

import logging
import subprocess
import tempfile
from pathlib import Path

from vidstab import VidStab

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from .base import BaseDatasetProcessor
from .config import VideoStabilizationConfig
from .utils import (
    extract_camera_keys,
)

class VideoStabilizer(BaseDatasetProcessor):
    """Stabilizes videos in the dataset"""

    def __init__(self, config: VideoStabilizationConfig):
        super().__init__(config)
        self.stabilizer = VidStab()

    def convert_av1_to_h264(self, input_path: Path, output_path: Path) -> bool:
        """Convert AV1 video to H264 format for compatibility"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to convert {input_path} to H264: {e}")
            return False

    def stabilize_video(self, input_path: Path, output_path: Path) -> bool:
        """Stabilize a video file"""
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
                    smoothing_window=self.config.smoothing_window,
                )

                # Convert back to MP4
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    'ffmpeg', '-y', '-i', str(temp_stabilized),
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            return True
        except Exception as e:
            self.logger.error(f"Failed to stabilize {input_path}: {e}")
            return False

    def process_dataset(self, dataset: LeRobotDataset) -> LeRobotDataset:
        """Process dataset to stabilize videos"""
        self.logger.info("Starting video stabilization")

        episodes = self.config.episodes or range(len(dataset.meta.episodes))
        camera_keys = self.config.camera_keys or extract_camera_keys(dataset)

        output_dir = Path(self.config.output_dir)

        for episode_idx in episodes:
            self.logger.info(f"Processing episode {episode_idx}")

            for camera_key in camera_keys:
                try:
                    # Get video path
                    video_path = dataset.meta.get_video_file_path(episode_idx, camera_key)
                    input_video_path = dataset.meta.root / video_path

                    if output_dir:
                        output_video_path = output_dir / dataset.repo_id / video_path
                        success = self.stabilize_video(input_video_path, output_video_path)
                        if success:
                            self.logger.info(f"  Stabilized: {video_path}")
                        else:
                            self.logger.warning(f"  Failed to stabilize: {video_path}")

                except Exception as e:
                    self.logger.error(f"Error processing {camera_key} for episode {episode_idx}: {e}")
            # Save dataset 
            self.save_or_push_dataset(dataset)
        return dataset
