import os
import torch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, HF_LEROBOT_HOME

# Vision-Language Model imports (model-agnostic interface)
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq
)
from qwen_vl_utils import process_vision_info
from .config import TaskEnhancementConfig
from .vlm_interface import TaskDescriptionGenerator
from .utils import push_dataset_to_hb, print_dataset_structure, get_episode_description, sample_episode_frames, get_task_index_from_episode_index, update_meta_files, setup_logger


class LeRobotTaskEnhancer:
    """Main class for enhancing LeRobot dataset task descriptions"""
    
    def __init__(self, config: TaskEnhancementConfig, repo_id: str):
        self.config = config
        self.vlm = TaskDescriptionGenerator(config)
        self.logger = setup_logger("Task enhancer")
        self.dataset = LeRobotDataset(repo_id)
        self.tasks_mapping =  {} # Store enhanced task descriptions
        self.videos_path =  HF_LEROBOT_HOME / dataset.repo_id 
    
    def enhance_dataset(self,episodes: Optional[List[int]] = None) -> LeRobotDataset:
        """Enhance task descriptions for the entire dataset or specific episodes"""
        # Debug: Print dataset structure
        print_dataset_structure(self.logger,self.dataset)
      
        if episodes is None:
            # Use the actual number of episodes from metadata
            episodes = list(range(self.dataset.meta.total_episodes))
                
        self.logger.info(f"Enhancing {len(episodes)} episodes...")
        self.logger.info(f"Available tasks: {self.dataset.meta.tasks}")

        episodes_desc = get_episode_description(self.dataset.meta,episodes) 
    
        for episode_idx in tqdm(episodes, desc="Processing episodes"):
            original_desc = episodes_desc[episode_idx]
            
            try:
                enhanced_description = self._enhance_episode( episode_idx,original_desc)
                
                if not self.config.dry_run:
                    # Update the enhanced tasks dictionary
                    self._update_episode_description(episode_idx, enhanced_description)
                    pass
                else:
                    self.logger.info(f"Episode {episode_idx}:")
                    self.logger.info(f"  Original: {original_desc}")
                    self.logger.info(f"  Enhanced: {enhanced_description}")
                    
            except Exception as e:
                self.logger.error(f"Error processing episode {episode_idx}: {str(e)}")
                continue
        
        if self.config.output_dir and not self.config.dry_run:
            self._save_enhanced_dataset()
        
        if self.config.push_to_hub:
            push_dataset_to_hb("Afaf/test",f"{self.config.output_dir}/{self.dataset.repo_id}")
        
        return self.dataset
    
    def _enhance_episode(self, episode_idx: int,original_desc:str) -> str:
        """Enhance task description for a single episode"""
        
        # Sample frames from the episode
        frames = sample_episode_frames(self.dataset, episode_idx,self.config)
        
        # Generate enhanced description
        enhanced_desc = self.vlm.generate_description(frames, original_desc)
        
        return enhanced_desc

    
    def _update_episode_description(self, episode_idx: int, new_description: str):
        """Update the task description for an episode"""
   
        task_index = int(get_task_index_from_episode_index(self.dataset, episode_idx))

        # Update the metadata tasks dictionary
        if task_index not in self.tasks_mapping:
            self.tasks_mapping[task_index] = {"old":self.dataset.meta.tasks[task_index],"new":new_description}
        self.dataset.meta.tasks[task_index] = new_description

        # Update the metadata episodes dict
        self.dataset.meta.episodes[episode_idx]["tasks"] =  [self.tasks_mapping[task_index]["new"]]

        self.logger.info(f"Updated task {task_index} description to: {new_description}")
    
    def _save_enhanced_dataset(self):
        """Save the enhanced dataset metadata and task descriptions"""
        update_meta_files(self.dataset, self.config.output_dir, self.logger)
            
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # save the mapping file 
        with open(output_path / "task_mapping.json", "w") as f:
            json.dump(self.tasks_mapping, f, indent=2)
        

# Example usage and utility functions
def enhance_lerobot_dataset(
    dataset_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    episodes: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    **kwargs
) -> LeRobotDataset:
    """
    Convenience function to enhance a LeRobot dataset
    
    Args:
        dataset_path: Path to the LeRobot dataset (repo_id)
        model_name: Name of the vision-language model to use
        model_type: Type of model ("qwen2.5-vl", "llava-next", etc.)
        episodes: List of episode indices to process (None for all)
        output_dir: Directory to save enhanced dataset
        dry_run: If True, only show what would be changed without modifying
        **kwargs: Additional configuration parameters
    
    Returns:
        Enhanced LeRobotDataset
    """
    
    # Create configuration
    config = TaskEnhancementConfig(
        model_name=model_name,
        output_dir=output_dir,
        dry_run=dry_run,
        **kwargs
    )
    
    # Create enhancer and process dataset
    enhancer = LeRobotTaskEnhancer(config,repo_id=dataset_path)
    enhanced_dataset = enhancer.enhance_dataset(episodes)
    
    return enhanced_dataset


if __name__ == "__main__":
    repo_id = "satvikahuja/mixer_on_off_new_1"
    dataset = enhance_lerobot_dataset(
        dataset_path=repo_id,
        dry_run=False,
        output_dir="test",
        push_to_hub=True,
    )