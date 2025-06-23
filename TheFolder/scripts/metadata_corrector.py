#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
from datetime import datetime

import numpy as np
import torch


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata



class RobotDatasetCorrector:
    """Interactive corrector for robot datasets."""
    
    # Common robot types mapping
    ROBOT_TYPES = {
        "1": {"name": "aloha", "description": "ALOHA Mobile Manipulation"},
        "2": {"name": "xarm", "description": "xArm Robotic Arm"},
        "3": {"name": "franka", "description": "Franka Emika Panda"},
        "4": {"name": "ur5", "description": "Universal Robots UR5"},
        "5": {"name": "kuka", "description": "KUKA Robot"},
        "6": {"name": "baxter", "description": "Rethink Robotics Baxter"},
        "7": {"name": "sawyer", "description": "Rethink Robotics Sawyer"},
        "8": {"name": "fetch", "description": "Fetch Robotics"},
        "9": {"name": "generic", "description": "Generic Robot"},
        "10": {"name": "custom", "description": "Custom Robot (specify name)"}
    }
    
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.dataset = None
        self.metadata = None
        
    def load_dataset(self) -> bool:
        """Load the dataset and metadata."""
        try:
            # First load metadata to check basic info
            self.metadata = LeRobotDatasetMetadata(self.repo_id)
            print(f"✓ Metadata loaded: {self.repo_id}")
            print(f"  Total episodes: {self.metadata.total_episodes}")
            print(f"  Total frames: {self.metadata.total_frames}")
            print(f"  FPS: {self.metadata.fps}")
            print(f"  Robot type: {self.metadata.robot_type}")
            
            # Then load the actual dataset
            self.dataset = LeRobotDataset(self.repo_id)
            print(f"✓ Dataset loaded successfully")
            print(f"  Available episodes: {self.dataset.num_episodes}")
            print(f"  Available frames: {self.dataset.num_frames}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset for potential issues."""
        analysis = {
            "episodes": {
                "total_expected": self.metadata.total_episodes,
                "total_available": self.dataset.num_episodes,
                "missing_count": 0,
                "episode_lengths": []
            },
            "fps": {
                "current": self.metadata.fps,
                "needs_correction": False,
                "recommended": 30
            },
            "robot_type": {
                "current": self.metadata.robot_type,
                "needs_correction": False
            },
            "camera_info": {
                "camera_keys": self.metadata.camera_keys,
                "features": {}
            }
        }
        
        # Check episode availability
        if self.dataset.num_episodes < self.metadata.total_episodes:
            analysis["episodes"]["missing_count"] = self.metadata.total_episodes - self.dataset.num_episodes
        
        # Analyze episode lengths
        for episode_idx in range(self.dataset.num_episodes):
            from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
            to_idx = self.dataset.episode_data_index["to"][episode_idx].item()
            episode_length = to_idx - from_idx
            analysis["episodes"]["episode_lengths"].append(episode_length)
        
        # Check FPS
        if self.metadata.fps > 30:
            analysis["fps"]["needs_correction"] = True
            analysis["fps"]["undersample_factor"] = self.metadata.fps / 30
        
        # Check robot type
        current_robot = self.metadata.robot_type
        if current_robot in ['unknown', 'generic', None, ''] or not current_robot:
            analysis["robot_type"]["needs_correction"] = True
        
        # Analyze camera features
        for camera_key in self.metadata.camera_keys:
            if camera_key in self.dataset.features:
                analysis["camera_info"]["features"][camera_key] = self.dataset.features[camera_key]
        
        return analysis
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print dataset analysis results."""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        # Episodes analysis
        expected = analysis['episodes']['total_expected']
        available = analysis['episodes']['total_available']
        print(f"📊 Episodes: {available}/{expected} available")
        
        if analysis['episodes']['missing_count'] > 0:
            print(f"⚠️  Missing episodes: {analysis['episodes']['missing_count']}")
            print("   → Metadata may need updating")
        else:
            print("✓ All episodes are available")
        
        # Episode length statistics
        lengths = analysis['episodes']['episode_lengths']
        if lengths:
            print(f"   Episode lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # FPS analysis
        print(f"\n🎥 Camera FPS: {analysis['fps']['current']}")
        if analysis['fps']['needs_correction']:
            factor = analysis['fps']['undersample_factor']
            print(f"⚠️  FPS is above 30 (current: {analysis['fps']['current']})")
            print(f"   → Recommended: undersample by factor {factor:.1f} to reach 30fps")
        else:
            print("✓ FPS is within acceptable range")
        
        # Robot type analysis
        print(f"\n🤖 Robot Type: {analysis['robot_type']['current']}")
        if analysis['robot_type']['needs_correction']:
            print("⚠️  Robot type is not specified or generic")
            print("   → Consider setting a specific robot type")
        else:
            print("✓ Robot type is specified")
        
        # Camera info
        print(f"\n📷 Camera Keys: {analysis['camera_info']['camera_keys']}")
        for camera_key, feature_info in analysis['camera_info']['features'].items():
            shape = feature_info.get('shape', 'unknown')
            print(f"   {camera_key}: {shape}")
    
    def get_dataset_info_path(self) -> Path:
        """Get the path to dataset_info.json file."""
        # Try to find the dataset locally first
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        local_path = HF_LEROBOT_HOME / self.repo_id
        
        if local_path.exists():
            return local_path / "meta"/ "info.json"
        else:
            # If not local, we'll need to work with the hub version
            return None
    
    def update_metadata_for_deleted_episodes(self) -> bool:
        """Update metadata after episodes have been deleted."""
        try:
            print("\n" + "="*50)
            print("UPDATING METADATA FOR DELETED EPISODES")
            print("="*50)
            
            # Get dataset info file path
            info_path = self.get_dataset_info_path()
            if not info_path or not info_path.exists():
                print("⚠️  Dataset info file not found locally")
                print("   This correction requires a local dataset copy")
                return False
            
            # Read current dataset info
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Update episode counts
            current_episodes = self.dataset.num_episodes
            current_frames = self.dataset.num_frames
            
            print(f"Current episodes in dataset: {current_episodes}")
            print(f"Current frames in dataset: {current_frames}")
            print(f"Metadata shows episodes: {self.metadata.total_episodes}")
            print(f"Metadata shows frames: {self.metadata.total_frames}")
            
            if current_episodes == self.metadata.total_episodes:
                print("✓ Metadata already matches dataset, no update needed")
                return True
            
            # Create backup
            backup_path = info_path.with_suffix('.json.backup')
            shutil.copy2(info_path, backup_path)
            print(f"✓ Backup created: {backup_path}")
            
            # Update metadata
            if 'splits' in dataset_info and 'train' in dataset_info['splits']:
                dataset_info['splits']['train']['num_examples'] = current_frames
            
            # Update total episodes/frames if these fields exist
            if 'total_episodes' in dataset_info:
                dataset_info['total_episodes'] = current_episodes
            if 'total_frames' in dataset_info:
                dataset_info['total_frames'] = current_frames
            
            # Add update timestamp
            dataset_info['last_updated'] = datetime.now().isoformat()
            dataset_info['corrected_by'] = 'robot_dataset_corrector'
            
            # Write updated info
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            print("✓ Metadata updated successfully")
            return True
            
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False
    
    def correct_camera_fps(self, target_fps: int = 30) -> bool:
        """Correct camera FPS by undersampling if necessary."""
        try:
            print(f"\n" + "="*50)
            print(f"CORRECTING CAMERA FPS TO {target_fps}")
            print("="*50)
            
            current_fps = self.metadata.fps
            print(f"Current FPS: {current_fps}")
            
            if current_fps <= target_fps:
                print(f"✓ FPS is already at or below {target_fps}, no correction needed")
                return True
            
            undersample_factor = current_fps / target_fps
            step_size = int(np.round(undersample_factor))
            
            print(f"Undersample factor: {undersample_factor:.2f}")
            print(f"Will keep every {step_size} frames")
            
            # Get dataset info path
            info_path = self.get_dataset_info_path()
            if not info_path or not info_path.exists():
                print("⚠️  Dataset info file not found locally")
                print("   This correction requires a local dataset copy")
                return False
            
            print("This correction would:")
            print(f"  - Keep every {step_size} frames")
            print(f"  - Reduce dataset size by ~{(1 - 1/step_size)*100:.1f}%")
            print(f"  - Update FPS from {current_fps} to {target_fps}")
            print("  - Require reprocessing all video files")
            
            confirm = input("\n⚠️  This is a major operation. Proceed? (y/N): ")
            if confirm.lower() not in ['y', 'yes']:
                print("FPS correction cancelled")
                return False
            
            # Create backup
            info_path = self.get_dataset_info_path()
            backup_path = info_path.with_suffix('.json.backup')
            shutil.copy2(info_path, backup_path)
            print(f"✓ Backup created: {backup_path}")
            
            # Update dataset info with new FPS
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            
            dataset_info['fps'] = target_fps
            dataset_info['last_updated'] = datetime.now().isoformat()
            dataset_info['corrected_by'] = 'robot_dataset_corrector'
            dataset_info['original_fps'] = current_fps
            dataset_info['undersample_factor'] = step_size
            
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            print("✓ FPS metadata updated")
            print("⚠️  Note: Video files still need to be reprocessed")
            print("   Consider using LeRobot's video processing tools")
            
            return True
            
        except Exception as e:
            print(f"Error correcting FPS: {e}")
            return False
    
    def correct_robot_type(self) -> bool:
        """Interactively correct robot type."""
        try:
            print("\n" + "="*50)
            print("CORRECTING ROBOT TYPE")
            print("="*50)
            
            current_type = self.metadata.robot_type
            print(f"Current robot type: {current_type}")
            
            print("\nAvailable robot types:")
            for key, robot in self.ROBOT_TYPES.items():
                print(f"  {key}. {robot['name']} - {robot['description']}")
            
            while True:
                choice = input("\nSelect robot type (1-10): ").strip()
                
                if choice in self.ROBOT_TYPES:
                    selected_robot = self.ROBOT_TYPES[choice]
                    robot_name = selected_robot['name']
                    
                    if robot_name == 'custom':
                        robot_name = input("Enter custom robot name: ").strip()
                    
                    # Get dataset info path
                    info_path = self.get_dataset_info_path()
                    if not info_path or not info_path.exists():
                        print("⚠️  Dataset info file not found locally")
                        print("   This correction requires a local dataset copy")
                        return False
                    
                    # Create backup
                    backup_path = info_path.with_suffix('.json.backup')
                    if not backup_path.exists():
                        shutil.copy2(info_path, backup_path)
                        print(f"✓ Backup created: {backup_path}")
                    
                    # Update robot type
                    with open(info_path, 'r') as f:
                        dataset_info = json.load(f)
                    
                    dataset_info['robot_type'] = robot_name
                    dataset_info['last_updated'] = datetime.now().isoformat()
                    dataset_info['corrected_by'] = 'robot_dataset_corrector'
                    
                    with open(info_path, 'w') as f:
                        json.dump(dataset_info, f, indent=2)
                    
                    print(f"✓ Robot type updated to: {robot_name}")
                    return True
                else:
                    print("Invalid choice. Please select 1-10.")
                    
        except Exception as e:
            print(f"Error correcting robot type: {e}")
            return False
    
    def show_dataset_sample(self) -> None:
        """Show a sample of the dataset for verification."""
        try:
            print("\n" + "="*50)
            print("DATASET SAMPLE")
            print("="*50)
            
            if self.dataset.num_frames == 0:
                print("No frames available in dataset")
                return
            
            # Get a sample frame
            sample_frame = self.dataset[0]
            
            print("Available keys in frame:")
            for key in sample_frame.keys():
                value = sample_frame[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)} - {value}")
            
            # Show camera info if available
            if self.metadata.camera_keys:
                camera_key = self.metadata.camera_keys[0]
                if camera_key in sample_frame:
                    camera_tensor = sample_frame[camera_key]
                    print(f"\nCamera '{camera_key}' info:")
                    print(f"  Shape: {camera_tensor.shape}")
                    print(f"  Dtype: {camera_tensor.dtype}")
                    print(f"  Min/Max values: {camera_tensor.min():.3f} / {camera_tensor.max():.3f}")
        
        except Exception as e:
            print(f"Error showing sample: {e}")
    
    def interactive_correction(self) -> None:
        """Main interactive correction interface."""
        if not self.load_dataset():
            return
        
        analysis = self.analyze_dataset()
        self.print_analysis(analysis)
        
        print("\n" + "="*60)
        print("AVAILABLE CORRECTIONS")
        print("="*60)
        print("1. Update metadata for deleted episodes")
        print("2. Correct camera FPS (set to 30fps, undersample if needed)")
        print("3. Correct robot type selection")
        print("4. Show dataset sample")
        print("5. Run all applicable corrections")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    self.update_metadata_for_deleted_episodes()
                elif choice == '2':
                    self.correct_camera_fps()
                elif choice == '3':
                    self.correct_robot_type()
                elif choice == '4':
                    self.show_dataset_sample()
                elif choice == '5':
                    print("Running all applicable corrections...")
                    if analysis['episodes']['missing_count'] > 0:
                        self.update_metadata_for_deleted_episodes()
                    if analysis['fps']['needs_correction']:
                        self.correct_camera_fps()
                    if analysis['robot_type']['needs_correction']:
                        self.correct_robot_type()
                elif choice == '6':
                    print("Exiting...")
                    break
                else:
                    print("Invalid choice. Please select 1-6.")
                    
                # Re-analyze after corrections
                if choice in ['1', '2', '3', '5']:
                    print("\nRe-analyzing dataset...")
                    # Reload metadata to reflect changes
                    self.metadata = LeRobotDatasetMetadata(self.repo_id)
                    analysis = self.analyze_dataset()
                    self.print_analysis(analysis)
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Dataset Corrector")
    parser.add_argument("--repo-id", required=True, help="Repository ID (e.g., username/dataset-name)")
    parser.add_argument("--show-sample", action="store_true", help="Show a sample of the dataset")
    
    args = parser.parse_args()
    
    print("🤖 Robot Dataset Corrector")
    print("="*50)
    print(f"Repository: {args.repo_id}")
    
    corrector = RobotDatasetCorrector(args.repo_id)
    
    if args.show_sample:
        if corrector.load_dataset():
            corrector.show_dataset_sample()
        return
    
    # Run interactive correction
    corrector.interactive_correction()


if __name__ == "__main__":
    main()