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
import copy
import re

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import hf_transform_to_torch

# Vision-Language Model imports
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor,
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModel
)
from qwen_vl_utils import process_vision_info
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


@dataclass
class ViewpointNormalizationConfig:
    """Configuration for camera viewpoint normalization"""
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_type: str = "qwen2.5-vl"  # Options: "qwen2.5-vl", "llava-next"
    device: str = "auto"
    sample_frames: int = 5  # Number of frames to sample for viewpoint classification
    frame_sampling: str = "uniform"  # "uniform", "random"
    output_dir: Optional[str] = None
    dry_run: bool = False
    confidence_threshold: float = 0.8  # Minimum confidence for automatic classification
    standardized_names: Dict[str, str] = None  # Custom mapping for standardized names
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.standardized_names is None:
            self.standardized_names = {
                "top": "OBS_IMAGE_1",
                "overhead": "OBS_IMAGE_1", 
                "bird_eye": "OBS_IMAGE_1",
                "wrist": "OBS_IMAGE_2",
                "gripper": "OBS_IMAGE_2",
                "hand": "OBS_IMAGE_2",
                "side": "OBS_IMAGE_3",
                "front": "OBS_IMAGE_3",
                "lateral": "OBS_IMAGE_3"
            }


class ViewpointClassifier:
    """VLM-based viewpoint classifier for camera normalization"""
    
    VIEWPOINT_CATEGORIES = {
        "top": "A top-down or overhead view looking down at the workspace from above",
        "wrist": "A view from the robot's wrist/gripper/end-effector showing what the robot hand sees",
        "side": "A side view of the robot and workspace from the side or front perspective",
        "other": "Any other viewpoint that doesn't fit the above categories"
    }
    
    def __init__(self, config: ViewpointNormalizationConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load the specified vision-language model"""
        if self.config.model_type == "qwen2.5-vl":
            self._load_qwen_model()
        elif self.config.model_type == "llava-next":
            self._load_llava_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _load_qwen_model(self):
        """Load Qwen2.5-VL model"""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device if self.config.device != "cpu" else None
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(self.config.model_name)
        
    def _load_llava_model(self):
        """Load LLaVA-Next model"""            
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device if self.config.device != "cpu" else None
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.config.model_name)
    
    def classify_viewpoint(self, images: List[Image.Image], camera_name: str) -> Tuple[str, float]:
        """
        Classify the viewpoint of camera images
        
        Args:
            images: List of sample images from the camera
            camera_name: Original camera name for context
            
        Returns:
            Tuple of (viewpoint_category, confidence)
        """
        prompt = self._build_classification_prompt(camera_name)
        
        if self.config.model_type == "qwen2.5-vl":
            return self._classify_qwen(images, prompt)
        elif self.config.model_type == "llava-next":
            return self._classify_llava(images, prompt)
        else:
            raise NotImplementedError(f"Classification not implemented for {self.config.model_type}")
    
    def _build_classification_prompt(self, camera_name: str) -> str:
        """Build prompt for viewpoint classification"""
        categories_text = "\n".join([
            f"- {cat}: {desc}" 
            for cat, desc in self.VIEWPOINT_CATEGORIES.items()
        ])
        
        prompt = f"""Analyze these robotics camera images and classify the viewpoint type.

Camera name: {camera_name}

Viewpoint categories:
{categories_text}

Based on the robot's position, workspace visibility, and camera angle in these images, what type of viewpoint is this?

Answer with just ONE word from: top, wrist, side, other

Answer:"""
        
        return prompt
    
    def _classify_qwen(self, images: List[Image.Image], prompt: str) -> Tuple[str, float]:
        """Classify using Qwen2.5-VL model"""
        # Prepare messages with images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"type": "image", "image": img} for img in images]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.config.device)
        
        # Generate with logit inspection for confidence
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract viewpoint and confidence
        viewpoint, confidence = self._parse_classification_output(output_text, outputs.scores)
        return viewpoint, confidence
    
    def _classify_llava(self, images: List[Image.Image], prompt: str) -> Tuple[str, float]:
        """Classify using LLaVA-Next model"""
        # Use first image for LLaVA (can be adapted for multi-image)
        image = images[0] if images else None
        
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
        # Extract only the generated part
        generated_text = generated_text[len(prompt):].strip()
        
        # Extract viewpoint and confidence
        viewpoint, confidence = self._parse_classification_output(generated_text, outputs.scores)
        return viewpoint, confidence
    
    def _parse_classification_output(self, output_text: str, scores=None) -> Tuple[str, float]:
        """Parse the classification output and estimate confidence"""
        # Clean and extract viewpoint
        output_text = output_text.lower().strip()
        
        # Find the viewpoint in the output
        viewpoint = "other"  # default
        confidence_scores = {}
        
        # Check for each category in the output
        for category in self.VIEWPOINT_CATEGORIES.keys():
            if category in output_text:
                # If we find an exact match at word boundaries, prioritize it
                if re.search(rf'\b{category}\b', output_text):
                    viewpoint = category
                    break
                else:
                    confidence_scores[category] = 0.8
        
        # If no exact match but we have partial matches
        if viewpoint == "other" and confidence_scores:
            viewpoint = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
        
        # Estimate confidence based on scores if available
        confidence = 0.5  # default
        if scores is not None and len(scores) > 0:
            try:
                # Get the confidence from the first generated token's probabilities
                first_token_logits = scores[0][0]  # First generated token
                probs = torch.softmax(first_token_logits, dim=-1)
                max_prob = torch.max(probs).item()
                confidence = min(max_prob * 1.5, 1.0)  # Scale up but cap at 1.0
            except:
                confidence = 0.9 if viewpoint != "other" else 0.5
        else:
            confidence = 0.9 if viewpoint != "other" else 0.5
        
        return viewpoint, confidence


class CameraViewpointNormalizer:
    """Main class for normalizing camera viewpoints in LeRobot datasets"""
    
    def __init__(self, config: ViewpointNormalizationConfig):
        self.config = config
        self.classifier = ViewpointClassifier(config)
        self.logger = self._setup_logger()
        self.viewpoint_mapping = {}  # Store original -> normalized mappings
        self.classification_results = {}  # Store detailed classification results
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def normalize_dataset_cameras(self, dataset: LeRobotDataset) -> Dict[str, str]:
        """
        Analyze and normalize camera viewpoints for the entire dataset
        
        Args:
            dataset: LeRobotDataset to analyze
            
        Returns:
            Dictionary mapping original camera names to standardized names
        """
        camera_keys = self._extract_camera_keys(dataset)
        self.logger.info(f"Found {len(camera_keys)} camera keys: {camera_keys}")
        
        # Classify each camera viewpoint
        for camera_key in tqdm(camera_keys, desc="Classifying camera viewpoints"):
            try:
                viewpoint, confidence = self._classify_camera_viewpoint(dataset, camera_key)
                
                # Store classification results
                self.classification_results[camera_key] = {
                    "viewpoint": viewpoint,
                    "confidence": confidence,
                    "original_name": camera_key
                }
                
                self.logger.info(f"Camera '{camera_key}' classified as '{viewpoint}' (confidence: {confidence:.3f})")
                
            except Exception as e:
                self.logger.error(f"Error classifying camera {camera_key}: {str(e)}")
                self.classification_results[camera_key] = {
                    "viewpoint": "other",
                    "confidence": 0.0,
                    "original_name": camera_key,
                    "error": str(e)
                }
        
        # Generate standardized mapping
        self.viewpoint_mapping = self._generate_standardized_mapping()
        
        # Save results if requested
        if self.config.output_dir and not self.config.dry_run:
            self._save_normalization_results()
        
        return self.viewpoint_mapping
    
    def _extract_camera_keys(self, dataset: LeRobotDataset) -> List[str]:
        """Extract all camera-related keys from the dataset"""
        camera_keys = []
        
        # Use dataset metadata if available
        if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'camera_keys') and dataset.meta.camera_keys:
            camera_keys.extend(dataset.meta.camera_keys)
        else:
            # Fallback: try to get from dataset metadata
            try:
                # Get dataset metadata
                meta = LeRobotDatasetMetadata(dataset.repo_id)
                if hasattr(meta, 'camera_keys') and meta.camera_keys:
                    camera_keys.extend(meta.camera_keys)
            except:
                pass
        
        # Also search through a sample to find image-like keys if no camera_keys found
        if not camera_keys and len(dataset) > 0:
            sample = dataset[0]
            camera_keys = self._find_image_keys_in_sample(sample)
        
        return camera_keys
    
    def _find_image_keys_in_sample(self, sample: Dict) -> List[str]:
        """Find image keys in a sample by looking for tensor data with image-like shapes"""
        image_keys = []
        
        def search_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, torch.Tensor):
                        # Check if tensor has image-like dimensions
                        if len(value.shape) >= 3 and (value.shape[0] == 3 or value.shape[-1] == 3):
                            # Looks like an image tensor (CHW or HWC format)
                            image_keys.append(current_key)
                    elif isinstance(value, dict):
                        search_recursive(value, current_key)
        
        search_recursive(sample)
        return image_keys
    
    def _classify_camera_viewpoint(self, dataset: LeRobotDataset, camera_key: str) -> Tuple[str, float]:
        """Classify the viewpoint of a specific camera"""
        # Sample images from this camera
        images = self._sample_camera_images(dataset, camera_key)
        
        if not images:
            self.logger.warning(f"No images found for camera {camera_key}")
            return "other", 0.0
        
        # Classify the viewpoint
        viewpoint, confidence = self.classifier.classify_viewpoint(images, camera_key)
        
        return viewpoint, confidence
    
    def _sample_camera_images(self, dataset: LeRobotDataset, camera_key: str) -> List[Image.Image]:
        """Sample images from a specific camera across the dataset"""
        images = []
        total_frames = len(dataset)
        
        if total_frames == 0:
            return images
        
        # Determine sampling indices
        if self.config.frame_sampling == "uniform":
            if total_frames <= self.config.sample_frames:
                indices = list(range(total_frames))
            else:
                step = max(1, total_frames // self.config.sample_frames)
                indices = [i * step for i in range(self.config.sample_frames)]
                # Ensure we don't exceed bounds
                indices = [min(idx, total_frames - 1) for idx in indices]
        elif self.config.frame_sampling == "random":
            indices = np.random.choice(total_frames, 
                                     min(self.config.sample_frames, total_frames), 
                                     replace=False)
        else:
            indices = list(range(min(self.config.sample_frames, total_frames)))
        
        # Extract images
        for idx in indices:
            try:
                frame_data = dataset[idx]
                
                # Get the image from the camera key
                img_data = self._get_nested_value(frame_data, camera_key)
                
                if img_data is not None:
                    # Convert to PIL Image
                    pil_image = self._tensor_to_pil(img_data)
                    if pil_image is not None:
                        images.append(pil_image)
                        
            except Exception as e:
                self.logger.warning(f"Error sampling frame {idx} for camera {camera_key}: {str(e)}")
                continue
        
        return images[:self.config.sample_frames]
    
    def _get_nested_value(self, data: Dict, key: str):
        """Get value from nested dictionary using dot notation"""
        current = data
        
        try:
            current = current[key]
            return current
        except (KeyError, TypeError):
            return None
    
    def _tensor_to_pil(self, tensor_data) -> Optional[Image.Image]:
        """Convert tensor data to PIL Image"""
        try:
            if isinstance(tensor_data, torch.Tensor):
                # Convert to numpy
                img_array = tensor_data.cpu().numpy()
                
                # Handle different tensor formats
                if img_array.ndim == 3:
                    # CHW format (channels first)
                    if img_array.shape[0] <= 4:  # Likely channels first
                        img_array = np.transpose(img_array, (1, 2, 0))
                    # Now should be HWC format
                
                # Normalize to 0-255 if needed
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
                
                # Handle single channel (grayscale)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 1:
                    img_array = np.repeat(img_array, 3, axis=-1)
                
                # Create PIL Image
                return Image.fromarray(img_array)
                
            elif isinstance(tensor_data, np.ndarray):
                # Handle numpy array similarly
                if tensor_data.dtype != np.uint8:
                    if tensor_data.max() <= 1.0:
                        tensor_data = (tensor_data * 255).astype(np.uint8)
                    else:
                        tensor_data = tensor_data.astype(np.uint8)
                
                return Image.fromarray(tensor_data)
                
            elif isinstance(tensor_data, Image.Image):
                return tensor_data
            else:
                return None
                
        except Exception as e:
            print(f"Error converting tensor to PIL: {e}")
            return None
    
    def _generate_standardized_mapping(self) -> Dict[str, str]:
        """Generate mapping from original camera names to standardized names"""
        mapping = {}
        viewpoint_counts = {"top": 0, "wrist": 0, "side": 0, "other": 0}
        
        # Sort cameras by confidence (highest first) within each viewpoint category
        cameras_by_viewpoint = {}
        for camera_key, result in self.classification_results.items():
            viewpoint = result["viewpoint"]
            if viewpoint not in cameras_by_viewpoint:
                cameras_by_viewpoint[viewpoint] = []
            cameras_by_viewpoint[viewpoint].append((camera_key, result))
        
        # Sort by confidence within each viewpoint
        for viewpoint in cameras_by_viewpoint:
            cameras_by_viewpoint[viewpoint].sort(key=lambda x: x[1]["confidence"], reverse=True)
        
        # Assign standardized names prioritizing top -> wrist -> side
        priority_order = ["top", "wrist", "side", "other"]
        
        for viewpoint in priority_order:
            if viewpoint not in cameras_by_viewpoint:
                continue
                
            for camera_key, result in cameras_by_viewpoint[viewpoint]:
                confidence = result["confidence"]
                
                # Only auto-assign if confidence is high enough
                if confidence >= self.config.confidence_threshold:
                    if viewpoint in self.config.standardized_names:
                        base_name = self.config.standardized_names[viewpoint]
                        
                        # Handle multiple cameras of the same viewpoint
                        if viewpoint_counts[viewpoint] == 0:
                            standardized_name = base_name
                        else:
                            # Append number for additional cameras: OBS_IMAGE_1_2, OBS_IMAGE_1_3, etc.
                            standardized_name = f"{base_name}_{viewpoint_counts[viewpoint] + 1}"
                        
                        mapping[camera_key] = standardized_name
                        viewpoint_counts[viewpoint] += 1
                    else:
                        # Keep original name if no standardized mapping available
                        mapping[camera_key] = camera_key
                else:
                    # Low confidence - keep original name and flag for manual review
                    mapping[camera_key] = f"{camera_key}_MANUAL_REVIEW_NEEDED"
                    self.logger.warning(f"Low confidence ({confidence:.3f}) for {camera_key}, manual review needed")
        
        return mapping
    
    def _save_normalization_results(self):
        """Save normalization results and mappings"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed classification results
        classification_file = output_path / "camera_classification_results.json"
        with open(classification_file, "w") as f:
            json.dump(self.classification_results, f, indent=2, default=str)
        
        # Save viewpoint mapping
        mapping_file = output_path / "camera_viewpoint_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(self.viewpoint_mapping, f, indent=2)
        
        # Save configuration
        config_file = output_path / "normalization_config.json"
        config_dict = {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "sample_frames": self.config.sample_frames,
            "confidence_threshold": self.config.confidence_threshold,
            "standardized_names": self.config.standardized_names
        }
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Create summary report
        self._create_summary_report(output_path)
        
        self.logger.info(f"Normalization results saved to {output_path}")
    
    def _create_summary_report(self, output_path: Path):
        """Create a human-readable summary report"""
        report_lines = [
            "Camera Viewpoint Normalization Summary",
            "=" * 50,
            "",
            f"Total cameras analyzed: {len(self.classification_results)}",
            f"Confidence threshold: {self.config.confidence_threshold}",
            "",
            "Classification Results:",
            "-" * 25
        ]
        
        # Group by viewpoint
        viewpoint_groups = {}
        for camera_key, result in self.classification_results.items():
            viewpoint = result["viewpoint"]
            if viewpoint not in viewpoint_groups:
                viewpoint_groups[viewpoint] = []
            viewpoint_groups[viewpoint].append((camera_key, result))
        
        for viewpoint, cameras in viewpoint_groups.items():
            report_lines.append(f"\n{viewpoint.upper()} viewpoint ({len(cameras)} cameras):")
            for camera_key, result in cameras:
                confidence = result["confidence"]
                standardized = self.viewpoint_mapping.get(camera_key, "N/A")
                report_lines.append(f"  {camera_key} -> {standardized} (confidence: {confidence:.3f})")
        
        report_lines.extend([
            "",
            "Standardized Mapping:",
            "-" * 20
        ])
        
        for original, standardized in self.viewpoint_mapping.items():
            report_lines.append(f"  {original} -> {standardized}")
        
        # Save report
        report_file = output_path / "normalization_summary.txt"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))


# Utility functions
def normalize_dataset_cameras(
    dataset_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    model_type: str = "qwen2.5-vl",
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    **kwargs
) -> Dict[str, str]:
    """
    Convenience function to normalize camera viewpoints in a LeRobot dataset
    
    Args:
        dataset_path: Path to the LeRobot dataset (repo_id)
        model_name: Name of the vision-language model to use
        model_type: Type of model ("qwen2.5-vl", "llava-next")
        output_dir: Directory to save normalization results
        dry_run: If True, only analyze without saving
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary mapping original camera names to standardized names
    """
    # Load the dataset
    dataset = LeRobotDataset(dataset_path)
    
    # Create configuration
    config = ViewpointNormalizationConfig(
        model_name=model_name,
        model_type=model_type,
        output_dir=output_dir,
        dry_run=dry_run,
        **kwargs
    )
    
    # Create normalizer and process dataset
    normalizer = CameraViewpointNormalizer(config)
    viewpoint_mapping = normalizer.normalize_dataset_cameras(dataset)
    
    return viewpoint_mapping


def apply_camera_normalization(dataset: LeRobotDataset, mapping: Dict[str, str]) -> LeRobotDataset:
    """
    Apply camera normalization mapping to a dataset
    
    Args:
        dataset: Original LeRobotDataset
        mapping: Dictionary mapping original camera names to standardized names
        
    Returns:
        Dataset with normalized camera names
    """
    # This is a conceptual implementation - actual implementation would depend
    # on how LeRobot handles feature renaming
    
    # For now, we'll store the mapping as metadata
    if not hasattr(dataset, '_camera_normalization_mapping'):
        dataset._camera_normalization_mapping = mapping
    
    # In practice, you would need to:
    # 1. Create a new dataset with renamed features
    # 2. Update metadata camera_keys
    # 3. Possibly create a wrapper that translates keys on access
    
    print("Camera normalization mapping applied:")
    for original, normalized in mapping.items():
        print(f"  {original} -> {normalized}")
    
    return dataset


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Normalize camera viewpoints in LeRobot dataset")
#     parser.add_argument("dataset_path", help="Path to the LeRobot dataset (repo_id)")
#     parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct", 
#                        help="Vision-language model to use")
#     parser.add_argument("--model-type", default="qwen2.5-vl", 
#                        choices=["qwen2.5-vl", "llava-next"],
#                        help="Type of model")
#     parser.add_argument("--output-dir", help="Output directory for normalization results")
#     parser.add_argument("--dry-run", action="store_true", 
#                        help="Analyze cameras without saving results")
#     parser.add_argument("--sample-frames", type=int, default=5,
#                        help="Number of frames to sample per camera")
#     parser.add_argument("--confidence-threshold", type=float, default=0.8,
#                        help="Minimum confidence for automatic classification")
    
#     args = parser.parse_args()
    
#     # Run normalization
#     mapping = normalize_dataset_cameras(
#         dataset_path=args.dataset_path,
#         model_name=args.model_name,
#         model_type=args.model_type,
#         output_dir=args.output_dir,
#         dry_run=args.dry_run,
#         sample_frames=args.sample_frames,
#         confidence_threshold=args.confidence_threshold
#     )
    
#     print("\nCamera Viewpoint Normalization Complete!")
#     print("Final Mapping:")
#     for original, normalized in mapping.items():
#         print(f"  {original} -> {normalized}")