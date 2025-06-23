"""Vision Language Model interface for various VLM models."""

import torch
from typing import List, Tuple, Optional
from PIL import Image
import re
import numpy as np

# Vision-Language Model imports
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration

)
from qwen_vl_utils import process_vision_info

from .config import ViewpointNormalizationConfig, TaskEnhancementConfig


class VisionLanguageModel:
    """Unified interface for vision-language models"""
    
    def __init__(self,config: TaskEnhancementConfig | ViewpointNormalizationConfig):
        self.model_name = config.model_name
        self.device = config.device 
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified vision-language model"""
        self._load_qwen_model()
       
    def _load_qwen_model(self):
        """Load Qwen2.5-VL model"""
        print(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device != "cpu" else None
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
    
    def generate_text(self, images: List[Image.Image], prompt: str, max_tokens: int = 50) -> str:
        """Generate text from images and prompt"""
        return self._generate_qwen(images, prompt, max_tokens)
      
    def _generate_qwen(self, images: List[Image.Image], prompt: str, max_tokens: int) -> str:
        """Generate using Qwen2.5-VL model"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"type": "image", "image": img} for img in images]
            }
        ]
        
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
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
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
        
        return output_text.strip()
    
    


class ViewpointClassifier:
    """VLM-based viewpoint classifier for camera normalization"""
    
    VIEWPOINT_CATEGORIES = {
        "top": "A top-down view looking directly down at the workspace (images.top)",
        # "front": "A front-facing view of the robot and workspace (images.front)",
        "left": "A view from the left side of the workspace (images.left)",
        "right": "A view from the right side of the workspace (images.right)",
        "wrist.left": "View from the left wrist-mounted camera (images.wrist.left)",
        "wrist.right": "View from the right wrist-mounted camera (images.wrist.right)",
        "wrist.top": "View from the top wrist-mounted camera (images.wrist.top)",
        "wrist.bottom": "View from the bottom wrist-mounted camera (images.wrist.bottom)",
        # "other": "Any other viewpoint that doesn't fit the standard locations"
    }

    
    def __init__(self, config: ViewpointNormalizationConfig):
        self.config = config
        self.vlm = VisionLanguageModel(config)
        
    def classify_viewpoint(self, images: List[Image.Image], camera_name: str) -> Tuple[str, float]:
        """Classify the viewpoint of camera images"""
        prompt = self._build_classification_prompt(camera_name)
        output_text = self.vlm.generate_text(images, prompt, max_tokens=30)
        
        # Parse the classification output
        viewpoint = self._parse_classification_output(output_text)
        return viewpoint
    
    def _build_classification_prompt(self, camera_name: str) -> str:
        """Build enhanced prompt for viewpoint classification"""
        
        prompt = f"""Analyze this robotics camera image to classify the viewpoint accurately.

Camera source: {camera_name}

VISUAL ANALYSIS CRITERIA:

TOP VIEW characteristics:
- Looking DOWN at the workspace surface
- Robot arms/hands visible from above
- Workspace appears flat and spread out
- Objects cast shadows downward
- Tabletop or work surface dominates the view

LEFT SIDE VIEW characteristics:
- Looking FROM THE LEFT at the workspace
- Robot visible from the side profile
- Vertical elements (robot base, arms) clearly visible
- Can see height/depth of workspace
- Horizontal workspace edge visible

RIGHT SIDE VIEW characteristics:
- Looking FROM THE RIGHT at the workspace
- Robot visible from opposite side profile
- Vertical elements visible from right perspective
- Mirror image perspective compared to left view

WRIST CAMERA characteristics:
- Close-up view from robot's end-effector
- Very close to objects being manipulated
- Limited field of view
- May show gripper/end-effector parts in frame

CLASSIFICATION TASK:
Based on the visual characteristics above, classify this image as ONE of these exact options:
- top
- left  
- right
- wrist.left
- wrist.right
- wrist.top
- wrist.bottom

CRITICAL INSTRUCTIONS:
1. Carefully examine the viewing angle and perspective
2. Look for distinguishing features that separate top-down from side views
3. Consider the robot's position relative to the camera
4. Output ONLY the single most accurate category
5. Do not include explanations or additional text

ANSWER:"""
        
        return prompt
    
    def _parse_classification_output(self, output_text: str) -> Tuple[str, float]:
        """Parse the classification output with improved matching"""
        output_text = output_text.lower().strip()
        
        # Remove common prefixes/suffixes that might interfere
        output_text = re.sub(r'^(the\s+|category:\s*|answer:\s*|viewpoint:\s*)', '', output_text)
        output_text = re.sub(r'\s*(view|viewpoint|camera)$', '', output_text)
        
        viewpoint = "other"
        confidence = 0.5
        
        # Check for exact matches first (highest priority)
        exact_matches = {
            "wrist.left": r'\bwrist\.left\b',
            "wrist.right": r'\bwrist\.right\b', 
            "wrist.top": r'\bwrist\.top\b',
            "wrist.bottom": r'\bwrist\.bottom\b',
            "top": r'\btop\b',
            "left": r'\bleft\b',
            "right": r'\bright\b'
        }
        
        for category, pattern in exact_matches.items():
            if re.search(pattern, output_text):
                viewpoint = category
                confidence = 0.9
                break
        
        # If no exact match, try partial matches with lower confidence
        if viewpoint == "other":
            partial_matches = {
                "top": ["top-down", "overhead", "bird", "down"],
                "left": ["left side", "from left", "left-side"],
                "right": ["right side", "from right", "right-side"],
                "wrist.left": ["wrist left", "left wrist"],
                "wrist.right": ["wrist right", "right wrist"],
                "wrist.top": ["wrist top", "top wrist"],
                "wrist.bottom": ["wrist bottom", "bottom wrist"]
            }
            
            for category, keywords in partial_matches.items():
                if any(keyword in output_text for keyword in keywords):
                    viewpoint = category
                    confidence = 0.7
                    break
        
        return viewpoint, confidence
    
    def classify_with_validation(self, images: List[Image.Image], camera_name: str, 
                               expected_viewpoint: str = None) -> Tuple[str, float, bool]:
        """Classify viewpoint with optional validation against expected result"""
        viewpoint, confidence = self.classify_viewpoint(images, camera_name)
        
        is_correct = expected_viewpoint is None or viewpoint == expected_viewpoint
        
        if not is_correct and expected_viewpoint:
            print(f"WARNING: Classified as '{viewpoint}' but expected '{expected_viewpoint}'")
            print(f"Camera: {camera_name}, Confidence: {confidence:.2f}")
        
        return viewpoint, confidence, is_correct


class TaskDescriptionGenerator:
    """VLM-based task description generator"""
    
    def __init__(self, config: TaskEnhancementConfig):
        self.config = config
        self.vlm = VisionLanguageModel(config)
    
    def generate_description(self, images: List[Image.Image], current_task: str) -> str:
        """Generate improved task description"""
        prompt = self._build_prompt(current_task)
        output_text = self.vlm.generate_text(images, prompt, max_tokens=50)
        return self._clean_output(output_text)
    
    def _build_prompt(self, current_task: str) -> str:
        """Build the prompt for task description generation"""
        if self.config.custom_prompt:
            return self.config.custom_prompt.format(current_task=current_task)
        
        prompt = f"""Here is a current task description: {current_task}. 
Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max {self.config.max_length} characters). 
Do not include unnecessary words. Be concise.

Here are some examples: Pick up the cube and place it in the box, open the drawer and so on.
Start directly with an action verb like "Pick", "Place", "Open", etc.

Similar to the provided examples, what is the main action done by the robot arm?"""
        
        return prompt
    
    def _clean_output(self, text: str) -> str:
        """Clean and validate the generated output"""
        text = " ".join(text.split())
        
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length].rsplit(' ', 1)[0]
        
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text