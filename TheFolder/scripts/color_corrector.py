import os
import subprocess
import shutil
import tempfile
import cv2
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from .config import ColorCorrectionConfig
from .utils import correct_color_cast_gray_world, correct_yellow_blue_lab, filter_videos_on_episodes_and_camerakeys, push_dataset_to_hb, copytree_skip_existing, extract_camera_keys
from .base import BaseDatasetProcessor
from pathlib import Path


class ColorCorrector(BaseDatasetProcessor):
    """Corrects color cast in dataset videos"""
    
    def __init__(self, config: ColorCorrectionConfig):
        super().__init__(config)
        
    def correct_video_colors(self, input_path: Path, output_path: Path) -> bool:
        """Correct colors in a video file"""
        try:
            # Create temporary directory for frame extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = Path(temp_dir) / "frames"
                frames_dir.mkdir()
                
                # Extract frames
                extract_cmd = [
                    self.config.ffmpeg_command, '-i', str(input_path),
                    str(frames_dir / "frame_%04d.png")
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True)
                
                # Process each frame
                prev_b_shift = 0.0
                frame_files = sorted(frames_dir.glob("*.png"))
                
                for frame_path in frame_files:
                    frame = cv2.imread(str(frame_path))
                    
                    if self.config.gray_world_enabled:
                        corrected_frame = correct_color_cast_gray_world(frame)
                    else:
                        corrected_frame, prev_b_shift = correct_yellow_blue_lab(
                            frame, 
                            self.config.target_b_mean,
                            self.config.smoothing_factor,
                            prev_b_shift
                        )
                    
                    cv2.imwrite(str(frame_path), corrected_frame)
                
                # Reconstruct video
                output_path.parent.mkdir(parents=True, exist_ok=True)
                reconstruct_cmd = [
                    self.config.ffmpeg_command, '-y',
                    '-i', str(frames_dir / "frame_%04d.png"),
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    str(output_path)
                ]
                subprocess.run(reconstruct_cmd, check=True, capture_output=True)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to correct colors in {input_path}: {e}")
            return False
    
    def process_dataset(self, dataset: LeRobotDataset) -> LeRobotDataset:
        """Process dataset to correct color cast"""
        self.logger.info("Starting color correction")
        
        episodes = self.config.episodes or range(len(dataset.meta.episodes))
        camera_keys = self.config.camera_keys or extract_camera_keys(dataset)
        
        # Get videos to process
        videos_to_process = filter_videos_on_episodes_and_camerakeys(
            dataset, episodes, camera_keys
        )
        
        output_dir = Path(self.config.output_dir)
        
        for video_path in videos_to_process:
            self.logger.info(f"Processing video: {video_path}")
            
            if output_dir:
                output_video_path = output_dir/ dataset.repo_id / video_path
                
                success = self.correct_video_colors(
                    dataset.meta.root / video_path,
                    output_video_path
                )
                if success:
                    self.logger.info(f"  Successfully corrected: {video_path}")
                else:
                    self.logger.warning(f"  Failed to correct: {video_path}")
            else:
                self.logger.info(f"Would correct {video_path}")
        # Save dataset 
        self.save_or_push_dataset(dataset)
        return dataset



# def correct_dataset_video_colors_ffmpeg(repo_id, output_dir=None,push_to_hub=False,new_repo_id=None):
#     """Use FFmpeg-based approach for better AV1 compatibility"""
#     color_config = ColorCorrectionConfig()
#     corrector = VideoColorCorrectorFFmpeg(config=color_config, repo_id=repo_id,push_to_hub=push_to_hub,output_dir=output_dir,new_repo_id=new_repo_id)
#     corrector.correct_videos_colors()