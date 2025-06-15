#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime
import cv2
import numpy as np
import subprocess
import shutil
import tempfile
from tqdm import tqdm

# Import the correction tools
from video_stabilizer import VideoStabilizer, stabilize_dataset
from task_enhancer import enhance_lerobot_dataset, TaskEnhancementConfig
from metadata_corrector import RobotDatasetCorrector

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoColorCorrector:
    """Video color correction utilities"""
    
    @staticmethod
    def correct_color_cast_gray_world(frame):
        """Gray World color correction"""
        frame_float = frame.astype(np.float64)
        
        b_mean = np.mean(frame_float[:, :, 0])
        g_mean = np.mean(frame_float[:, :, 1])
        r_mean = np.mean(frame_float[:, :, 2])
        
        gray_mean = (b_mean + g_mean + r_mean) / 3.0
        
        b_factor = gray_mean / b_mean if b_mean > 0 else 1.0
        g_factor = gray_mean / g_mean if g_mean > 0 else 1.0
        r_factor = gray_mean / r_mean if r_mean > 0 else 1.0
        
        frame_float[:, :, 0] *= b_factor
        frame_float[:, :, 1] *= g_factor
        frame_float[:, :, 2] *= r_factor
        
        return np.clip(frame_float, 0, 255).astype(np.uint8)
    
    @staticmethod
    def correct_yellow_blue_lab(frame, target_b_mean=128, smoothing_factor=0.05, prev_b_shift=0.0):
        """LAB b* channel correction"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        current_b_mean = np.mean(b)
        required_shift = target_b_mean - current_b_mean
        current_b_shift = prev_b_shift * (1 - smoothing_factor) + required_shift * smoothing_factor
        
        b_corrected = b.astype(np.float64) + current_b_shift
        b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
        
        lab_corrected = cv2.merge([l, a, b_corrected])
        frame_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        return frame_corrected, current_b_shift
    
    @staticmethod
    def process_video_via_frames(input_video_path, output_video_path, correction_method="lab"):
        """Process video by exporting frames, processing them, then reassembling with ffmpeg"""
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix="color_correction_")
        
        try:
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video {input_video_path}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing {total_frames} frames at {fps} FPS...")
            
            # Process frames
            previous_b_shift = 0.0
            lab_smoothing_alpha = 0.05
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Processing frame {frame_count}/{total_frames}...")
                
                # Apply color correction based on method
                if correction_method == "lab":
                    corrected_frame, previous_b_shift = VideoColorCorrector.correct_yellow_blue_lab(
                        frame.copy(),
                        target_b_mean=128,
                        smoothing_factor=lab_smoothing_alpha,
                        prev_b_shift=previous_b_shift
                    )
                elif correction_method == "gray_world":
                    corrected_frame = VideoColorCorrector.correct_color_cast_gray_world(frame.copy())
                else:
                    corrected_frame = frame.copy()
                
                # Save frame
                frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, corrected_frame)
            
            cap.release()
            logger.info(f"Processed {frame_count} frames")
            
            # Use ffmpeg to reassemble video
            logger.info("Reassembling video with ffmpeg...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_video_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info("Video reassembly successful!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            logger.error(f"Error during color correction: {e}")
            return False
        finally:
            # Clean up temporary frames
            if os.path.exists(temp_dir):
                logger.debug("Cleaning up temporary files...")
                shutil.rmtree(temp_dir)
    
    @staticmethod
    def process_video_opencv(input_video_path, output_video_path, correction_method="lab"):
        """Try direct OpenCV method for color correction"""
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {input_video_path}")
            return False
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try to create a working VideoWriter
        success = False
        codecs_to_try = [('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
        
        for codec, ext in codecs_to_try:
            test_output = output_video_path.rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(test_output, fourcc, fps, (frame_width, frame_height))
            
            # Test write
            dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            if out.write(dummy_frame) and out.isOpened():
                logger.info(f"OpenCV method working with codec: {codec}")
                out.release()
                success = True
                output_video_path = test_output
                break
            else:
                out.release()
        
        if not success:
            cap.release()
            logger.warning("OpenCV VideoWriter failed, falling back to ffmpeg method")
            return VideoColorCorrector.process_video_via_frames(input_video_path, output_video_path, correction_method)
        
        # Process with OpenCV
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        previous_b_shift = 0.0
        lab_smoothing_alpha = 0.05
        frame_count = 0
        
        logger.info(f"Processing {total_frames} frames with OpenCV...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"Processing frame {frame_count}/{total_frames}...")
            
            # Apply color correction
            if correction_method == "lab":
                corrected_frame, previous_b_shift = VideoColorCorrector.correct_yellow_blue_lab(
                    frame.copy(),
                    target_b_mean=128,
                    smoothing_factor=lab_smoothing_alpha,
                    prev_b_shift=previous_b_shift
                )
            elif correction_method == "gray_world":
                corrected_frame = VideoColorCorrector.correct_color_cast_gray_world(frame.copy())
            else:
                corrected_frame = frame.copy()
            
            out.write(corrected_frame)
        
        cap.release()
        out.release()
        logger.info(f"Processed {frame_count} frames successfully")
        return True
    
   

class LeRobotDatasetProcessor:
    """Main processor for LeRobot dataset corrections and enhancements"""
    
    def __init__(self, repo_id: str, output_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.output_dir = output_dir or f"./processed_{repo_id.replace('/', '_')}"
        self.dataset = None
        self.corrector = None
        
    def load_dataset(self) -> bool:
        """Load the dataset"""
        try:
            logger.info(f"Loading dataset: {self.repo_id}")
            self.dataset = LeRobotDataset(self.repo_id)
            self.corrector = RobotDatasetCorrector(self.repo_id)
            logger.info(f"✓ Dataset loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset for issues"""
        if not self.corrector:
            return {}
        
        if not self.corrector.load_dataset():
            return {}
        
        return self.corrector.analyze_dataset()
 
    def stabilize_videos(
        self,
        episodes: Optional[List[int]] = None,
        camera_keys: Optional[List[str]] = None,
        smoothing_window: int = 30,
        max_shift_percent: float = 0.2,
        push_to_hub: bool = False,
        output_repo_id: Optional[str] = None
    ) -> bool:
        """Stabilize videos in the dataset"""
        try:
            logger.info("Starting video stabilization...")
            stabilize_dataset(
                input_repo_id=self.repo_id,
                output_dir=os.path.join(self.output_dir, "stabilized"),
                episodes=episodes,
                camera_keys=camera_keys,
                smoothing_window=smoothing_window,
                max_shift_percent=max_shift_percent,
                push_to_hub=push_to_hub,
                output_repo_id=output_repo_id
            )
            logger.info("✓ Video stabilization completed")
            return True
        except Exception as e:
            logger.error(f"Video stabilization failed: {e}")
            return False
    
    def enhance_task_descriptions(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        model_type: str = "qwen2.5-vl",
        episodes: Optional[List[int]] = None,
        max_length: int = 30,
        sample_frames: int = 3,
        frame_sampling: str = "uniform",
        dry_run: bool = False
    ) -> bool:
        """Enhance task descriptions using vision-language models"""
        try:
            logger.info("Starting task description enhancement...")
            
            enhanced_dataset = enhance_lerobot_dataset(
                dataset_path=self.repo_id,
                model_name=model_name,
                model_type=model_type,
                episodes=episodes,
                output_dir=os.path.join(self.output_dir, "enhanced_tasks"),
                dry_run=dry_run,
                max_length=max_length,
                sample_frames=sample_frames,
                frame_sampling=frame_sampling
            )
            
            logger.info("✓ Task description enhancement completed")
            return True
        except Exception as e:
            logger.error(f"Task enhancement failed: {e}")
            return False
    
    def correct_metadata(
        self,
        update_episodes: bool = False,
        correct_fps: bool = False,
        target_fps: int = 30,
        correct_robot_type: bool = False,
        robot_type: Optional[str] = None
    ) -> bool:
        """Correct dataset metadata"""
        try:
            logger.info("Starting metadata corrections...")
            
            if not self.corrector.load_dataset():
                return False
            
            success = True
            
            if update_episodes:
                logger.info("Updating metadata for deleted episodes...")
                success &= self.corrector.update_metadata_for_deleted_episodes()
            
            if correct_fps:
                logger.info(f"Correcting FPS to {target_fps}...")
                success &= self.corrector.correct_camera_fps(target_fps)
            
            if correct_robot_type and robot_type:
                logger.info(f"Setting robot type to {robot_type}...")
                # This would need modification in the corrector to accept robot_type parameter
                success &= self.corrector.correct_robot_type()
            
            if success:
                logger.info("✓ Metadata corrections completed")
            else:
                logger.warning("Some metadata corrections failed")
            
            return success
        except Exception as e:
            logger.error(f"Metadata correction failed: {e}")
            return False
    
    def run_full_pipeline(
        self,
        stabilize: bool = False,
        enhance_tasks: bool = False,
        correct_meta: bool = False,
        correct_colors: bool = False,
        config: Dict[str, Any] = None
    ) -> Dict[str, bool]:
        """Run the full processing pipeline"""
        config = config or {}
        results = {
            "load_dataset": False,
            "stabilize_videos": False,
            "enhance_tasks": False,
            "correct_metadata": False,
            "correct_colors": False
        }
        
        # Load dataset
        results["load_dataset"] = self.load_dataset()
        if not results["load_dataset"]:
            return results
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Video color correction
        from correct_colors import correct_videos_colors
        if correct_colors:
            color_config = config.get("colors", {})
            all_videos = self.dataset.get_episodes_file_paths()
            process_videos_flag = correct_videos_colors( all_videos,color_config['episodes'],color_config['camera_keys'])
            results["correct_colors"] = process_videos_flag if process_videos_flag  else True
        
        # Video stabilization
        if stabilize:
            stabilize_config = config.get("stabilize", {})
            results["stabilize_videos"] = self.stabilize_videos(**stabilize_config)
        
        # Task enhancement
        if enhance_tasks:
            enhance_config = config.get("enhance", {})
            results["enhance_tasks"] = self.enhance_task_descriptions(**enhance_config)
        
        # Metadata correction
        if correct_meta:
            meta_config = config.get("metadata", {})
            results["correct_metadata"] = self.correct_metadata(**meta_config)
        
        # Save processing report
        self.save_processing_report(results, config)
        
        return results
    
    def save_processing_report(self, results: Dict[str, bool], config: Dict[str, Any]) -> None:
        """Save a report of the processing results"""
        report = {
            "repo_id": self.repo_id,
            "output_dir": self.output_dir,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "config": config,
            "success_rate": sum(results.values()) / len(results)
        }
        
        report_path = Path(self.output_dir) / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved to: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LeRobot Dataset Processing Tool with Color Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all corrections including color correction
  python main.py --repo-id "username/dataset" --all
  
  # Only correct video colors
  python main.py --repo-id "username/dataset" --correct-colors
  
  # Correct colors with specific method
  python main.py --repo-id "username/dataset" --correct-colors --color-method gray_world
  
  # Only stabilize videos
  python main.py --repo-id "username/dataset" --stabilize
  
  # Enhance task descriptions with custom model
  python main.py --repo-id "username/dataset" --enhance-tasks --model-name "Qwen/Qwen2.5-VL-7B-Instruct"
  
  # Correct metadata issues
  python main.py --repo-id "username/dataset" --correct-metadata --update-episodes --correct-fps
        """
    )
    
    # Required arguments
    parser.add_argument("--repo-id", required=True, help="Repository ID (e.g., username/dataset-name)")
    parser.add_argument("--output-dir", help="Output directory for processed dataset")
    
    # Processing options
    parser.add_argument("--all", action="store_true", help="Run all processing steps")
    parser.add_argument("--correct-colors", action="store_true", help="Correct video color cast")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize videos")
    parser.add_argument("--enhance-tasks", action="store_true", help="Enhance task descriptions")
    parser.add_argument("--correct-metadata", action="store_true", help="Correct metadata issues")
    
    # Color correction parameters
    parser.add_argument("--color-method", default="lab", choices=["lab", "gray_world"], 
                       help="Color correction method")
    parser.add_argument("--use-ffmpeg", action="store_true", help="Use ffmpeg method instead of OpenCV")
    parser.add_argument("--replace-original", action="store_true", help="Replace original videos")
    
    # Video stabilization parameters
    parser.add_argument("--episodes", nargs="+", type=int, help="Specific episodes to process")
    parser.add_argument("--camera-keys", nargs="+", help="Specific camera keys to process")
    parser.add_argument("--smoothing-window", type=int, default=30, help="Smoothing window for stabilization")
    parser.add_argument("--max-shift-percent", type=float, default=0.2, help="Maximum shift percentage")
    
    # Task enhancement parameters
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Vision-language model")
    parser.add_argument("--model-type", default="qwen2.5-vl", choices=["qwen2.5-vl", "llava-next"], help="Model type")
    parser.add_argument("--max-length", type=int, default=30, help="Max length for task descriptions")
    parser.add_argument("--sample-frames", type=int, default=3, help="Number of frames to sample")
    parser.add_argument("--frame-sampling", default="uniform", 
                       choices=["uniform", "start", "middle", "end"], help="Frame sampling strategy")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for task enhancement")
    
    # Metadata correction parameters
    parser.add_argument("--update-episodes", action="store_true", help="Update episode metadata")
    parser.add_argument("--correct-fps", action="store_true", help="Correct FPS")
    parser.add_argument("--target-fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--correct-robot-type", action="store_true", help="Correct robot type")
    parser.add_argument("--robot-type", help="Specific robot type to set")
    
    # Hub parameters
    parser.add_argument("--push-to-hub", action="store_true", help="Push results to Hugging Face Hub")
    parser.add_argument("--output-repo-id", help="Output repository ID for hub")
    
    # Other options
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze dataset without processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = LeRobotDatasetProcessor(args.repo_id, args.output_dir)
    
    # Analyze only mode
    if args.analyze_only:
        logger.info("Analyzing dataset...")
        analysis = processor.analyze_dataset()
        if analysis:
            print("\n" + "="*60)
            print("DATASET ANALYSIS RESULTS")
            print("="*60)
            print(json.dumps(analysis, indent=2))
        return
    
    # Determine what to process
    process_colors = args.all or args.correct_colors
    process_stabilize = args.all or args.stabilize
    process_enhance = args.all or args.enhance_tasks
    process_metadata = args.all or args.correct_metadata
    
    if not any([process_colors, process_stabilize, process_enhance, process_metadata]):
        logger.error("No processing options selected. Use --all or specific options.")
        parser.print_help()
        return
    
    # Build configuration
    config = {}
    
    if process_colors:
        config["colors"] = {
            "episodes": args.episodes,
            "camera_keys": args.camera_keys,
            "correction_method": args.color_method,
            "use_opencv_first": not args.use_ffmpeg,
            "replace_original": args.replace_original
        }
    
    if process_stabilize:
        config["stabilize"] = {
            "episodes": args.episodes,
            "camera_keys": args.camera_keys,
            "smoothing_window": args.smoothing_window,
            "max_shift_percent": args.max_shift_percent,
            "push_to_hub": args.push_to_hub,
            "output_repo_id": args.output_repo_id
        }
    
    if process_enhance:
        config["enhance"] = {
            "model_name": args.model_name,
            "model_type": args.model_type,
            "episodes": args.episodes,
            "max_length": args.max_length,
            "sample_frames": args.sample_frames,
            "frame_sampling": args.frame_sampling,
            "dry_run": args.dry_run
        }
    
    if process_metadata:
        config["metadata"] = {
            "update_episodes": args.update_episodes,
            "correct_fps": args.correct_fps,
            "target_fps": args.target_fps,
            "correct_robot_type": args.correct_robot_type,
            "robot_type": args.robot_type
        }
    
    # Run processing pipeline
    logger.info(f"Starting processing pipeline for {args.repo_id}")
    results = processor.run_full_pipeline(
        stabilize=process_stabilize,
        enhance_tasks=process_enhance,
        correct_meta=process_metadata,
        correct_colors=process_colors,
        config=config
    )
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {step.replace('_', ' ').title()}: {'Success' if success else 'Failed'}")
    
    success_rate = sum(results.values()) / len(results)
    print(f"\nOverall Success Rate: {success_rate:.1%}")
    
    if processor.output_dir:
        print(f"Output saved to: {processor.output_dir}")


if __name__ == "__main__":
    main()