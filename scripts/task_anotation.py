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
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import hf_transform_to_torch

# Vision-Language Model imports (model-agnostic interface)
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
class TaskEnhancementConfig:
    """Configuration for task description enhancement"""
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_type: str = "qwen2.5-vl"  # Options: "qwen2.5-vl", "llava-next", "custom"
    max_length: int = 30
    device: str = "auto"
    custom_prompt: Optional[str] = None
    sample_frames: int = 3  # Number of frames to sample from episode
    frame_sampling: str = "uniform"  # "uniform", "start", "middle", "end"
    output_dir: Optional[str] = None 
    dry_run: bool = False
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class VisionLanguageModel:
    """Model-agnostic interface for vision-language models"""
    
    def __init__(self, config: TaskEnhancementConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
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
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        
    def _load_llava_model(self):
        """Load LLaVA-Next model"""            
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device if self.config.device != "cpu" else None
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.config.model_name)
    
    def generate_description(self, images: List[Image.Image], current_task: str) -> str:
        """Generate improved task description"""
        prompt = self._build_prompt(current_task)
        
        if self.config.model_type == "qwen2.5-vl":
            return self._generate_qwen(images, prompt)
        elif self.config.model_type == "llava-next":
            return self._generate_llava(images, prompt)
        else:
            raise NotImplementedError(f"Generation not implemented for {self.config.model_type}")
    
    def _build_prompt(self, current_task: str) -> str:
        """Build the prompt for task description generation"""
        if self.config.custom_prompt:
            return self.config.custom_prompt.format(current_task=current_task)
        
        default_prompt = f"""Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max {self.config.max_length} characters). Do not include unnecessary words.
Be concise.
Here are some examples: Pick up the cube and place it in the box, open the drawer and so on.
Start directly with an action verb like "Pick", "Place", "Open", etc.
Similar to the provided examples, what is the main action done by the robot arm?"""
        
        return default_prompt
    
    def _generate_qwen(self, images: List[Image.Image], prompt: str) -> str:
        """Generate using Qwen2.5-VL model"""
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
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return self._clean_output(output_text)
    
    def _generate_llava(self, images: List[Image.Image], prompt: str) -> str:
        """Generate using LLaVA-Next model"""
        # Use first image for LLaVA (can be extended for multi-image)
        image = images[0] if images else None
        
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the generated part
        generated_text = generated_text[len(prompt):].strip()
        
        return self._clean_output(generated_text)
    
    def _clean_output(self, text: str) -> str:
        """Clean and validate the generated output"""
        # Remove extra whitespace and newlines
        text = " ".join(text.split())
        
        # Truncate to max length
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length].rsplit(' ', 1)[0]
        
        # Ensure it starts with a capital letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text


class LeRobotTaskEnhancer:
    """Main class for enhancing LeRobot dataset task descriptions"""
    
    def __init__(self, config: TaskEnhancementConfig):
        self.config = config
        self.vlm = VisionLanguageModel(config)
        self.logger = self._setup_logger()
        self.dataset_metadata = None
        self.enhanced_tasks = {}  # Store enhanced task descriptions
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def enhance_dataset(self, dataset: LeRobotDataset,original_tasks_dict, episodes: Optional[List[int]] = None) -> LeRobotDataset:
        """Enhance task descriptions for the entire dataset or specific episodes"""
        # Load dataset metadata to access task descriptions
        self.dataset_metadata = LeRobotDatasetMetadata(dataset.repo_id)
        
        # Debug: Print dataset structure
        self.logger.info(f"Dataset structure debug:")
        self.logger.info(f"  episode_data_index type: {type(dataset.episode_data_index)}")
        self.logger.info(f"  episode_data_index keys: {dataset.episode_data_index.keys()}")
        self.logger.info(f"  'from' type: {type(dataset.episode_data_index['from'])}")
        self.logger.info(f"  'from' shape/length: {getattr(dataset.episode_data_index['from'], 'shape', len(dataset.episode_data_index['from']))}")
        
        if episodes is None:
            # Use the actual number of episodes from metadata
            episodes = list(range(self.dataset_metadata.total_episodes))
        
        # Don't deep copy the entire dataset - just reference it
        enhanced_dataset = dataset
        
        self.logger.info(f"Enhancing {len(episodes)} episodes...")
        self.logger.info(f"Available tasks: {self.dataset_metadata.tasks}")
        
        for episode_idx in tqdm(episodes, desc="Processing episodes"):
            try:
                enhanced_description = self._enhance_episode(dataset, episode_idx)
                
                if not self.config.dry_run:
                    # Update the enhanced tasks dictionary
                    self._update_episode_description(dataset, episode_idx, enhanced_description)
                    pass
                else:
                    original_desc = self._get_episode_description(dataset, episode_idx)
                    self.logger.info(f"Episode {episode_idx}:")
                    self.logger.info(f"  Original: {original_desc}")
                    self.logger.info(f"  Enhanced: {enhanced_description}")
                    
            except Exception as e:
                self.logger.error(f"Error processing episode {episode_idx}: {str(e)}")
                continue
        
        if self.config.output_dir and not self.config.dry_run:
            self._save_enhanced_dataset(enhanced_dataset,original_tasks_dict)
        
        return enhanced_dataset
    
    def _enhance_episode(self, dataset: LeRobotDataset, episode_idx: int) -> str:
        """Enhance task description for a single episode"""
        # Get original task description
        original_desc = self._get_episode_description(dataset, episode_idx)
        
        # Sample frames from the episode
        frames = self._sample_episode_frames(dataset, episode_idx)
        
        # Generate enhanced description
        enhanced_desc = self.vlm.generate_description(frames, original_desc)
        
        return enhanced_desc
    
    def _get_episode_description(self, dataset: LeRobotDataset, episode_idx: int) -> str:
        """Get the original task description for an episode from metadata"""
        # Get the first frame of the episode to access task_index
        episode_start = int(dataset.episode_data_index["from"][episode_idx])
        
        # Get the task_index from the frame data
        frame_data = dataset.hf_dataset[episode_start]
        task_index = int(frame_data["task_index"])
        
        # Get task description from metadata using task_index
        if self.dataset_metadata and task_index in self.dataset_metadata.tasks:
            return self.dataset_metadata.tasks[task_index]
        else:
            # Fallback to a default description
            self.logger.warning(f"Task index {task_index} not found in metadata for episode {episode_idx}")
            return "Perform robotics task"
    
    def _sample_episode_frames(self, dataset: LeRobotDataset, episode_idx: int) -> List[Image.Image]:
        """Sample frames from an episode"""
        episode_start = int(dataset.episode_data_index["from"][episode_idx])
        episode_end = int(dataset.episode_data_index["to"][episode_idx])
        episode_length = episode_end - episode_start
        
        # Determine frame indices to sample
        if self.config.frame_sampling == "uniform":
            if episode_length <= self.config.sample_frames:
                frame_indices = list(range(episode_start, episode_end))
            else:
                step = episode_length // self.config.sample_frames
                frame_indices = [episode_start + i * step for i in range(self.config.sample_frames)]
        elif self.config.frame_sampling == "start":
            frame_indices = [episode_start + i for i in range(min(self.config.sample_frames, episode_length))]
        elif self.config.frame_sampling == "middle":
            mid_point = episode_start + episode_length // 2
            half_frames = self.config.sample_frames // 2
            start_idx = max(episode_start, mid_point - half_frames)
            end_idx = min(episode_end, mid_point + half_frames)
            frame_indices = list(range(start_idx, end_idx))
        elif self.config.frame_sampling == "end":
            start_idx = max(episode_start, episode_end - self.config.sample_frames)
            frame_indices = list(range(start_idx, episode_end))
        
        # Extract images
        images = []
        for idx in frame_indices:
            # Get image from dataset (assuming there's an observation image)
            obs = dataset[idx]
            
            # Try different possible image keys
            possible_image_keys = ["observation.image", "image", "rgb", "camera_image", "observation.images.top"]
            
            for key in possible_image_keys:
                if self._has_nested_key(obs, key):
                    img_data = self._get_nested_value(obs, key)
                    if isinstance(img_data, torch.Tensor):
                        # Convert tensor to PIL Image
                        if img_data.dim() == 3:  # CHW format
                            img_data = img_data.permute(1, 2, 0)
                        img_data = (img_data.cpu().numpy() * 255).astype(np.uint8)
                        images.append(Image.fromarray(img_data))
                        break
                    elif isinstance(img_data, np.ndarray):
                        images.append(Image.fromarray(img_data))
                        break
                    elif isinstance(img_data, Image.Image):
                        images.append(img_data)
                        break
        
        return images[:self.config.sample_frames]  # Ensure we don't exceed the limit
    
    def _has_nested_key(self, data: Dict, key: str) -> bool:
        """Check if nested key exists in dictionary"""
        keys = key.split('.')
        current = data
        
        try:
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False
            return True
        except:
            return False
    
    def _get_nested_value(self, data: Dict, key: str):
        """Get value from nested dictionary key"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            current = current[k]
        
        return current
    
    def _update_episode_description(self, dataset: LeRobotDataset, episode_idx: int, new_description: str):
        """Update the task description for an episode"""
        episode_start = int(dataset.episode_data_index["from"][episode_idx])
        episode_end = int(dataset.episode_data_index["to"][episode_idx])
        
        # Get the task_index for this episode
        frame_data = dataset.hf_dataset[episode_start]
        task_index = int(frame_data["task_index"])
        
        # Store the enhanced description
        self.enhanced_tasks[task_index] = new_description
        
        # Update the metadata tasks dictionary
        if hasattr(self, 'dataset_metadata') and self.dataset_metadata:
            self.dataset_metadata.tasks[task_index] = new_description
            
        self.logger.info(f"Updated task {task_index} description to: {new_description}")
    
    def _save_enhanced_dataset(self, dataset: LeRobotDataset,original_desc: Dict[int, str]):
        """Save the enhanced dataset metadata and task descriptions"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Instead of saving the entire dataset (which has serialization issues),
        # we'll save the enhanced metadata and task mappings
        
        # Save enhanced metadata with updated task descriptions
        # dataset.repo_id = "Afaf/test"  # Example repo_id, replace with actual
        # dataset.push_to_hub("Afaf/test",token="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")
        
        # Save task mapping for easy reference
        task_mapping = {}
        for task_index, enhanced_desc in self.enhanced_tasks.items():
            original_desc = original_desc[task_index]
            task_mapping[task_index] = {
                "original": original_desc,
                "enhanced": enhanced_desc
            }
        
        with open(output_path / "task_mapping.json", "w") as f:
            json.dump(task_mapping, f, indent=2)
        
        self.logger.info(f"Enhanced dataset metadata and task mappings saved to {output_path}")
        self.logger.info(f"Enhanced {len(self.enhanced_tasks)} unique tasks")




# Example usage and utility functions
def enhance_lerobot_dataset(
    dataset_path: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    model_type: str = "qwen2.5-vl",
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
    # Load the dataset
    dataset = LeRobotDataset(dataset_path)
    original_tasks_dict = dataset.meta.tasks
    
    # Create configuration
    config = TaskEnhancementConfig(
        model_name=model_name,
        model_type=model_type,
        output_dir=output_dir,
        dry_run=dry_run,
        **kwargs
    )
    
    # Create enhancer and process dataset
    enhancer = LeRobotTaskEnhancer(config)
    enhanced_dataset = enhancer.enhance_dataset(dataset, original_tasks_dict,episodes)
    
    return enhanced_dataset


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance LeRobot dataset task descriptions")
    parser.add_argument("dataset_path", help="Path to the LeRobot dataset (repo_id)")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct", 
                       help="Vision-language model to use")
    parser.add_argument("--model-type", default="qwen2.5-vl", 
                       choices=["qwen2.5-vl", "llava-next"],
                       help="Type of model")
    parser.add_argument("--output-dir", help="Output directory for enhanced dataset")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show changes without modifying dataset")
    parser.add_argument("--episodes", nargs="+", type=int, 
                       help="Specific episodes to process")
    parser.add_argument("--max-length", type=int, default=30,
                       help="Maximum length of generated descriptions")
    parser.add_argument("--sample-frames", type=int, default=3,
                       help="Number of frames to sample per episode")
    
    args = parser.parse_args()
    
    # Example usage with the provided dataset
    if not hasattr(args, 'dataset_path') or not args.dataset_path:
        # Use the example dataset from your description
        args.dataset_path = "sixpigs1/so100_pick_cube_in_box"
    
    # Run enhancement
    enhanced_dataset = enhance_lerobot_dataset(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        model_type=args.model_type,
        episodes=args.episodes,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_length=args.max_length,
        sample_frames=args.sample_frames
    )
    
    
    print(f"Enhanced dataset processing complete!")