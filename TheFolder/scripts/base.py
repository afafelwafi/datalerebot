#!/usr/bin/env python3

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import itertools

import torch
import numpy as np
import cv2
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from vidstab import VidStab
from huggingface_hub import HfApi

from .config import (
    ViewpointNormalizationConfig,
    TaskEnhancementConfig, 
    ColorCorrectionConfig,
    VideoStabilizationConfig,
    ProcessingConfig
)
from .utils import (
    setup_logger,
    extract_camera_keys,
    sample_camera_images,
    sample_episode_frames,
    get_episode_description,
    update_meta_files,
    push_dataset_to_hb,
    filter_videos_on_episodes_and_camerakeys,
    copytree_skip_existing,
    correct_color_cast_gray_world,
    correct_yellow_blue_lab
)


class BaseDatasetProcessor:
    """Base class for dataset processors"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
    def process_dataset(self, dataset: LeRobotDataset) -> LeRobotDataset:
        """Process the dataset - to be implemented by subclasses"""
        raise NotImplementedError
        
    def save_or_push_dataset(self, dataset: LeRobotDataset) -> None:
        """Save dataset locally or push to hub based on configuration"""
        # Save locally
        if self.config.output_dir:
            self.logger.info(f"Saving dataset to {self.config.output_dir}")
            # Copy dataset structure and update metadata
            update_meta_files(dataset,  self.config.output_dir , self.logger)
            copytree_skip_existing(dataset.meta.root, f"{self.config.output_dir}/{dataset.repo_id}")
            
        # Push to hub
        if self.config.push_to_hub and self.config.output_repo_id:
            self.logger.info(f"Pushing to hub: {self.config.output_repo_id}")
            try:
                push_dataset_to_hb(self.config.output_repo_id,  self.config.output_dir)
            except Exception as e:
                self.logger.error(f"Failed to push to hub: {e}")

