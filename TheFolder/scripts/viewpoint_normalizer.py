import os
import torch
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image


# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .vlm_interface import ViewpointClassifier
from .config import ViewpointNormalizationConfig
from .utils import setup_logger, extract_camera_keys, sample_camera_images, copytree_skip_existing
from .base import BaseDatasetProcessor

class CameraViewpointNormalizer(BaseDatasetProcessor):
    """Main class for normalizing camera viewpoints in LeRobot datasets"""
    
    def __init__(self, config: ViewpointNormalizationConfig):
        super().__init__(config)
        self.classifier = ViewpointClassifier(config)
        self.viewpoint_mapping = {}  # Store original -> normalized mappings
        self.classification_results = {}

    
    def process_dataset(self, dataset: LeRobotDataset) -> LeRobotDataset:
        """
        Process the dataset by analyzing and normalizing camera viewpoints
        
        Args:
            dataset: LeRobotDataset to analyze
            
        Returns:
            The processed dataset (same instance, modifications applied)
        """
        self.classification_results = self.normalize_dataset_cameras(dataset)
        self.viewpoint_mapping = {key :  self.classification_results [key]["viewpoint"] for key in dataset.meta.camera_keys}
        dataset.meta.camera_keys = list(self.viewpoint_mapping.values())

        # Save results if requested
        if self.config.output_dir:
            self._save_normalization_results(dataset)
        
        return dataset
    
    def normalize_dataset_cameras(self,dataset: LeRobotDataset) -> Dict[str, str]:
        """
        Analyze and normalize camera viewpoints for the entire dataset
        
        Returns:
            Dictionary mapping original camera names to standardized names
        """
        camera_keys = extract_camera_keys(dataset)
        self.logger.info(f"Found {len(camera_keys)} camera keys: {camera_keys}")
        
        # Classify each camera viewpoint
        for i, camera_key in tqdm(enumerate(camera_keys), desc="Classifying camera viewpoints"):
            try:
                viewpoint = self._classify_camera_viewpoint(dataset,camera_key)
                
                # Store classification results
                self.classification_results[camera_key] = {
                    "viewpoint": viewpoint,
                }
                
                self.logger.info(f"Camera '{camera_key}' classified as '{viewpoint}'")
                
            except Exception as e:
                self.logger.error(f"Error classifying camera {camera_key}: {str(e)}")
                self.classification_results[camera_key] = {
                    "viewpoint": f"other_{i}",
                    "error": str(e)
                }
                
        return self.classification_results
    
    
    def _classify_camera_viewpoint(self,dataset:LeRobotDataset, camera_key: str) -> str:
        """Classify the viewpoint of a specific camera"""
        # Sample images from this camera
        images = sample_camera_images(dataset, camera_key, self.config)
        
        if not images:
            self.logger.warning(f"No images found for camera {camera_key}")
            return "other"
        
        # Classify the viewpoint
        viewpoint,confidence = self.classifier.classify_viewpoint(images, camera_key)
        
        return viewpoint
    
    
    def _save_normalization_results(self,dataset:LeRobotDataset):
        """Save normalization results and mappings"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # move the files to the new direction
        input_folder = dataset.meta.root/"videos"
        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                keep_file_path = os.path.join(path, name)
                if os.path.isfile(keep_file_path) and any([key in str(keep_file_path) for key in self.viewpoint_mapping]):
                    for key, value in self.viewpoint_mapping.items():
                        output_path = str(keep_file_path).replace(key,f"observation.images.{value}")
                        print(output_path)

                    
                   




        
        # Save detailed classification results
        # classification_file = output_path / "camera_classification_results.json"
        # with open(classification_file, "w") as f:
        #     json.dump(self.classification_results, f, indent=2, default=str)





# import os
# import torch
# from typing import Optional, Dict, Any, List, Tuple, Set
# from dataclasses import dataclass
# from pathlib import Path
# import json
# import logging
# from tqdm import tqdm
# import numpy as np
# from PIL import Image


# # LeRobot imports
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# from .vlm_interface import ViewpointClassifier
# from .config import ViewpointNormalizationConfig
# from .utils import setup_logger, extract_camera_keys, sample_camera_images

# class CameraViewpointNormalizer:
#     """Main class for normalizing camera viewpoints in LeRobot datasets"""
    
#     def __init__(self, config: ViewpointNormalizationConfig, repo_id: str):
#         self.config = config
#         self.classifier = ViewpointClassifier(config)
#         self.logger = setup_logger("CameraViewpointNormalizer")
#         self.viewpoint_mapping = {}  # Store original -> normalized mappings
#         self.dataset = LeRobotDataset(repo_id=repo_id)
#         self.classification_results = {}

    
#     def normalize_dataset_cameras(self) -> Dict[str, str]:
#         """
#         Analyze and normalize camera viewpoints for the entire dataset
        
#         Args:
#             dataset: LeRobotDataset to analyze
            
#         Returns:
#             Dictionary mapping original camera names to standardized names
#         """
#         camera_keys = extract_camera_keys(self.dataset)
#         self.logger.info(f"Found {len(camera_keys)} camera keys: {camera_keys}")
        
#         # Classify each camera viewpoint
#         for camera_key in tqdm(camera_keys, desc="Classifying camera viewpoints"):
#             try:
#                 viewpoint= self._classify_camera_viewpoint(camera_key)
                
#                 # Store classification results
#                 self.classification_results[camera_key] = {
#                     "viewpoint": viewpoint,
#                     "original_name": camera_key
#                 }
                
#                 self.logger.info(f"Camera '{camera_key}' classified as '{viewpoint}'")
                
#             except Exception as e:
#                 self.logger.error(f"Error classifying camera {camera_key}: {str(e)}")
#                 self.classification_results[camera_key] = {
#                     "viewpoint": "other",
#                     "original_name": camera_key,
#                     "error": str(e)
#                 }
                
#         # Save results if requested
#         if self.config.output_dir and not self.config.dry_run:
#             self._save_normalization_results()
        
#         return self.classification_results
    
    
#     def _classify_camera_viewpoint(self, camera_key: str) -> Tuple[str, float]:
#         """Classify the viewpoint of a specific camera"""
#         # Sample images from this camera
#         images = sample_camera_images(self.dataset, camera_key,self.config)
        
#         if not images:
#             self.logger.warning(f"No images found for camera {camera_key}")
#             return "other", 0.0
        
#         # Classify the viewpoint
#         viewpoint = self.classifier.classify_viewpoint(images, camera_key)
        
#         return viewpoint
    
    
#     def _save_normalization_results(self):
#         """Save normalization results and mappings"""
#         output_path = Path(self.config.output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         # Save detailed classification results
#         classification_file = output_path / "camera_classification_results.json"
#         with open(classification_file, "w") as f:
#             json.dump(self.classification_results, f, indent=2, default=str)

#         # Create summary report
#         self._create_summary_report(output_path)
        
#         self.logger.info(f"Normalization results saved to {output_path}")
    
#     def _create_summary_report(self, output_path: Path):
#         """Create a human-readable summary report"""
#         report_lines = [
#             "Camera Viewpoint Normalization Summary",
#             "=" * 50,
#             "",
#             f"Total cameras analyzed: {len(self.classification_results)}",
#             "",
#             "Classification Results:",
#             "-" * 25
#         ]
        
#         # Group by viewpoint
#         viewpoint_groups = {}
#         for camera_key, result in self.classification_results.items():
#             viewpoint = result["viewpoint"]
#             if viewpoint not in viewpoint_groups:
#                 viewpoint_groups[viewpoint] = []
#             viewpoint_groups[viewpoint].append((camera_key, result))
        
#         for viewpoint, cameras in viewpoint_groups.items():
#             report_lines.append(f"\n{viewpoint.upper()} viewpoint ({len(cameras)} cameras):")
#             for camera_key, result in cameras:
#                 standardized = self.viewpoint_mapping.get(camera_key, "N/A")
#                 report_lines.append(f"  {camera_key} -> {standardized}")
        
#         report_lines.extend([
#             "",
#             "Standardized Mapping:",
#             "-" * 20
#         ])
        
#         for original, standardized in self.viewpoint_mapping.items():
#             report_lines.append(f"  {original} -> {standardized}")
        
#         # Save report
#         report_file = output_path / "normalization_summary.txt"
#         with open(report_file, "w") as f:
#             f.write("\n".join(report_lines))


# # Utility functions
# def normalize_dataset_cameras(
#     dataset_path: str,
#     model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
#     output_dir: Optional[str] = None,
#     dry_run: bool = False,
#     **kwargs
# ) -> Dict[str, str]:
#     """
#     Convenience function to normalize camera viewpoints in a LeRobot dataset
    
#     Args:
#         dataset_path: Path to the LeRobot dataset (repo_id)
#         model_name: Name of the vision-language model to use
#         output_dir: Directory to save normalization results
#         dry_run: If True, only analyze without saving
#         **kwargs: Additional configuration parameters
    
#     Returns:
#         Dictionary mapping original camera names to standardized names
#     """
    
#     # Create configuration
#     config = ViewpointNormalizationConfig(
#         model_name=model_name,
#         output_dir=output_dir,
#         dry_run=dry_run,
#     )
    
#     # Create normalizer and process dataset
#     normalizer = CameraViewpointNormalizer(config,dataset_path)
#     viewpoint_mapping = normalizer.normalize_dataset_cameras()
    
#     return viewpoint_mapping

