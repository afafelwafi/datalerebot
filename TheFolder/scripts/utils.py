import os
import subprocess
import json
import tempfile
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Any
import logging
from huggingface_hub import create_repo,HfApi, upload_folder
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import shutil
from typing import List, Tuple, Optional, Union
from .config import ColorCorrectionConfig
import itertools
import cv2


def push_dataset_to_hb(repo_id:str,local_folder_path):
    """ Push dataset to the hub"""
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        upload_folder(repo_id=repo_id, folder_path=local_folder_path, repo_type="dataset")
    except:
        username = get_hf_username()
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        upload_folder(repo_id=f"{username}/{repo_id}", folder_path=local_folder_path, repo_type="dataset")
   
def print_dataset_structure(logger, dataset):
    logger.info(f"Dataset structure debug:")
    logger.info(f"  episode_data_index type: {type(dataset.episode_data_index)}")
    logger.info(f"  episode_data_index keys: {dataset.episode_data_index.keys()}")
    logger.info(f"  'from' type: {type(dataset.episode_data_index['from'])}")
    logger.info(f"  'from' shape/length: {getattr(dataset.episode_data_index['from'], 'shape', len(dataset.episode_data_index['from']))}")
    

def setup_logger(name: str) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def extract_camera_keys(dataset) -> List[str]:
    """Extract all camera-related keys from the dataset"""
    camera_keys = []
    
    # Try to get from dataset metadata
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'camera_keys') and dataset.meta.camera_keys:
        camera_keys.extend(dataset.meta.camera_keys)
    
    # Fallback: search through a sample
    if not camera_keys and len(dataset) > 0:
        sample = dataset[0]
        camera_keys = find_image_keys_in_sample(sample)
    
    return camera_keys


def find_image_keys_in_sample(sample: Dict) -> List[str]:
    """Find image keys in a sample by looking for tensor data with image-like shapes"""
    image_keys = []
    
    def search_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, torch.Tensor):
                    # Check if tensor has image-like dimensions
                    if len(value.shape) >= 3 and (value.shape[0] == 3 or value.shape[-1] == 3):
                        image_keys.append(current_key)
                elif isinstance(value, dict):
                    search_recursive(value, current_key)
    
    search_recursive(sample)
    return image_keys


def has_nested_key( data: Dict, key: str) -> bool:
        """Check if nested key exists in dictionary"""
        current = data
        
        try:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
            return True
        except:
            return False
    
def get_nested_value(data: Dict, key: str):
    """Get value from nested dictionary key"""
    current = data[key]
    return current


def tensor_to_pil(tensor_data) -> Optional[Image.Image]:
    """Convert tensor data to PIL Image"""
    try:
        if isinstance(tensor_data, torch.Tensor):
            img_array = tensor_data.cpu().numpy()
            
            # Handle different tensor formats
            if img_array.ndim == 3:
                # CHW format (channels first)
                if img_array.shape[0] <= 4:  # Likely channels first
                    img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if img_array.dtype in [np.float32, np.float64]:
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
            
            return Image.fromarray(img_array)
            
        elif isinstance(tensor_data, np.ndarray):
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


def sample_camera_images(dataset, camera_key,config) :
    """Sample images from a specific camera across the dataset"""
    images = []
    total_frames = len(dataset)
    
    if total_frames == 0:
        return images
    
    # Determine sampling indices
    if config.frame_sampling == "uniform":
        if total_frames <= config.sample_frames:
            indices = list(range(total_frames))
        else:
            step = max(1, total_frames // config.sample_frames)
            indices = [min(i * step, total_frames - 1) for i in range(config.sample_frames)]
    elif config.frame_sampling == "random":
        indices = list(np.random.choice(total_frames, min(config.sample_frames, total_frames), replace=False))
    # Extract images
    for idx in indices:
        try:
            frame_data = dataset[int(idx)]
            img_data = get_nested_value(frame_data, camera_key)
            
            if img_data is not None:
                pil_image = tensor_to_pil(img_data)
                if pil_image is not None:
                    images.append(pil_image)
        except Exception as e:
            print(f"Error sampling frame {idx} for camera {camera_key}: {str(e)}")
            continue
    
    return images[:config.sample_frames]

    
    return images
def apply_ffmpeg_on_all_videos(videos_directory):
    """
    Apply FFmpeg processing to all video files in a directory.
    Replaces original files with processed versions.
    
    Args:
        videos_directory (str): Path to directory containing video files
    """
    # Convert to Path object for easier handling
    video_dir = Path(videos_directory)
    
    # Check if directory exists
    if not video_dir.exists():
        print(f"Error: Directory {videos_directory} does not exist")
        return
    
    # Find all video files (common extensions)
    video_extensions = ['.mp4']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))  # Include uppercase
    
    if not video_files:
        print(f"No video files found in {videos_directory}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video file
    for video_path in tqdm(video_files, desc="Processing videos"):
        print(f"Processing: {video_path}")
        
        # Create temporary output file in system temp directory
        with tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False) as temp_file:
            temp_output_path = temp_file.name
        
        command = [
            'ffmpeg', '-i', str(video_path),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-y',  # Overwrite output files without asking
            temp_output_path
        ]
        
        try:
            result = subprocess.run(command, check=True)
            
            # Replace original file with processed version
            import shutil
            shutil.move(temp_output_path, str(video_path))
            print(f"✓ Successfully replaced: {video_path.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error processing {video_path.name}:")
            print(f"  Return code: {e.returncode}")
            # Clean up temp file
            try:
                os.unlink(temp_output_path)
            except:
                pass
                
        except Exception as e:
            print(f"✗ Unexpected error processing {video_path.name}: {e}")
            # Clean up temp file
            try:
                os.unlink(temp_output_path)
            except:
                pass
        break

def get_episode_description(dataset_meta,episodes) -> dict:
    """Get the original task description for each episode from metadata"""

    return {ep["episode_index"]:ep["tasks"][0] for ep in list(dataset_meta.episodes.values()) if ep["episode_index"] in episodes}
        
def sample_episode_frames(dataset, episode_idx,config):
    """Sample frames from an episode"""
    episode_start = int(dataset.episode_data_index["from"][episode_idx])
    episode_end = int(dataset.episode_data_index["to"][episode_idx])
    episode_length = episode_end - episode_start
    
    # Determine frame indices to sample
    if config.frame_sampling == "uniform":
        frame_indices = list(range(episode_start, episode_end)) if episode_length <= config.sample_frames else [episode_start + i * (episode_length // config.sample_frames) for i in range(config.sample_frames)]
    elif config.frame_sampling == "start":
        frame_indices = [episode_start + i for i in range(min(config.sample_frames, episode_length))]
    elif config.frame_sampling == "middle":
        frame_indices = list(range(max(episode_start, episode_start + episode_length // 2 - config.sample_frames // 2), min(episode_end, episode_start + episode_length // 2 + config.sample_frames // 2))) if config.frame_sampling == "middle" else frame_indices
    elif config.frame_sampling == "end":
        frame_indices = list(range(max(episode_start, episode_end - config.sample_frames), episode_end))
    
    # get all camera keys
    camera_keys = dataset.meta.camera_keys

    # Extract images
    images = []
    for idx in frame_indices:
        # Get image from dataset (assuming there's an observation image)
        obs = dataset[idx]
        # Try different possible image keys
        for key in camera_keys:
            if has_nested_key(obs, key):
                img_data = get_nested_value(obs, key)
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
    
    return images[:config.sample_frames]  # Ensure we don't exceed the limit

def get_task_index_from_episode_index(dataset,episode_idx):

    episode_start = int(dataset.episode_data_index["from"][episode_idx])
    episode_end = int(dataset.episode_data_index["to"][episode_idx])
    
    # Get the task_index for this episode
    frame_data = dataset.hf_dataset[episode_start]
    task_index = int(frame_data["task_index"])
    return task_index
        
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj


def update_meta_files(dataset,output_dir,logger):
    # Define the file path
    src_folder = dataset.meta.root
    dist_folder = f"{output_dir}/{dataset.repo_id}/meta"
    os.makedirs(dist_folder,exist_ok=True)
    # Read the dict
    for file_path in os.listdir(dist_folder):
        logger.info(f"Modifying: {file_path}")        
        try:
            # Modify the dict as needed
            meta_type = file_path.split("/")[-1].split(".")[0]
            data = getattr(dataset.meta, meta_type)
            logger.info(f"Modifying  with :",data)

            # Save back to file
            with open(f"{dist_folder}/{file_path}", "w") as f:
                f.write(json.dump(convert_ndarray(data), f,indent=1))
        except Exception as e:
            logger.error(f"Error modifying {file_path}: {e}")
    

def correct_color_cast_gray_world(frame):
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


def correct_yellow_blue_lab(frame, target_b_mean=128, smoothing_factor=0.05, prev_b_shift=0.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    current_b_mean = np.mean(b)
    required_shift = target_b_mean - current_b_mean
    current_b_shift = prev_b_shift * (1 - smoothing_factor) + required_shift * smoothing_factor

    b_corrected = np.clip(b.astype(np.float64) + current_b_shift, 0, 255).astype(np.uint8)
    lab_corrected = cv2.merge([l, a, b_corrected])
    frame_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return frame_corrected, current_b_shift


def filter_videos_on_episodes_and_camerakeys(dataset, episodes, camera_keys):
    keep_videos = [ dataset.meta.get_video_file_path(ep, camera_key)
        for ep, camera_key in list(itertools.product(episodes, camera_keys))
    ]
    return keep_videos

def get_hf_username():
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']
    return username

def copytree_skip_existing(src, dst):
    """
    Copy files from src to dst, skip files that already exist in dst.
    """
    for root, dirs, files in os.walk(src):
        # Compute relative path
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        # Create destination directories if they don't exist
        os.makedirs(dst_path, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)

            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {dst_file}")
            else:
                print(f"Skipped (already exists): {dst_file}")
