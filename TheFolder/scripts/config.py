from dataclasses import dataclass
from typing import Optional, Dict, List
import torch


@dataclass
class ViewpointNormalizationConfig:
    """Configuration for camera viewpoint normalization"""
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "auto"
    sample_frames: int = 8  # Number of frames to sample for viewpoint classification
    frame_sampling: str = "random"  # "uniform", "random"
    confidence_threshold: float = 0.8  # Minimum confidence for automatic classification
    standardized_names: Dict[str, str] = None  # Custom mapping for standardized names
    episodes: Optional[List[int]] = None
    camera_keys: Optional[List[str]] = None
    output_dir: Optional[str] = None
    output_repo_id: Optional[str] = None
    push_to_hub: bool = False
    private: bool = False
    dry_run: bool = False

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


@dataclass
class TaskEnhancementConfig:
    """Configuration for task description enhancement"""
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_length: int = 30
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Changed from str to torch.dtype
    custom_prompt: Optional[str] = None
    push_to_hub: bool = False
    sample_frames: int = 3  # Number of frames to sample from episode
    frame_sampling: str = "uniform"  # "uniform", "start", "middle", "end"
    output_dir: Optional[str] = None 
    dry_run: bool = False

    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VideoStabilizationConfig:
    """Configuration for video stabilization"""
    smoothing_window: int = 30
    max_shift_percent: float = 0.2
    episodes: Optional[List[int]] = None
    camera_keys: Optional[List[str]] = None
    output_dir: Optional[str] = None
    output_repo_id: Optional[str] = None
    push_to_hub: bool = False
    private: bool = False
    dry_run: bool = False

@dataclass
class ColorCorrectionConfig:
    """Configuration for color correction"""
    target_b_mean: int = 128
    smoothing_factor: float = 0.05
    gray_world_enabled: bool = False
    temp_frame_dir: str = "temp_frames"
    ffmpeg_command: str = 'ffmpeg'
    episodes: Optional[List[int]] = None
    camera_keys: Optional[List[str]] = None
    output_dir: Optional[str] = None
    output_repo_id: Optional[str] = None
    push_to_hub: bool = False
    private: bool = False
    dry_run: bool = False



@dataclass
class ProcessingConfig:
    """Main configuration for dataset processing"""
    output_dir: Optional[str] = None
    output_repo_id: Optional[str] = None
    push_to_hub: bool = False
    private: bool = False
    dry_run: bool = False
    episodes: Optional[List[int]] = None  # Specific episodes to process
    
    # Enable/disable modules
    enable_viewpoint_normalization: bool = True
    enable_task_enhancement: bool = True
    enable_color_correction: bool = False
    enable_video_stabilization: bool = False
    
    # Sub-configurations
    viewpoint_config: Optional[ViewpointNormalizationConfig] = None
    task_config: Optional[TaskEnhancementConfig] = None
    color_config: Optional[ColorCorrectionConfig] = None
    stabilization_config: Optional[VideoStabilizationConfig] = None
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.viewpoint_config is None:
            self.viewpoint_config = ViewpointNormalizationConfig()
        if self.task_config is None:
            self.task_config = TaskEnhancementConfig()
        if self.color_config is None:
            self.color_config = ColorCorrectionConfig()
        if self.stabilization_config is None:
            self.stabilization_config = VideoStabilizationConfig()
        
        # Sync common settings across all configs
        configs = [self.viewpoint_config, self.task_config, self.color_config, self.stabilization_config]
        
        for config in configs:
            if self.output_dir:
                config.output_dir = self.output_dir
            if self.output_repo_id:
                config.output_repo_id = self.output_repo_id
            config.push_to_hub = self.push_to_hub
            config.private = self.private
            config.dry_run = self.dry_run
            
            # Sync episodes for video-related configs
            if hasattr(config, 'episodes') and self.episodes:
                config.episodes = self.episodes




    