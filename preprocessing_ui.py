import gradio as gr
from typing import List
import time
import subprocess
import os

def preprocess_dataset(
    repo_id: str,
    output_dir: str,
    tasks: List[str],
    episodes: str,
    camera_keys: str,
    smoothing_window: int,
    max_shift_percent: float,
    model_name: str,
    model_type: str,
    max_length: int,
    sample_frames: int,
    frame_sampling: str,
    dry_run: bool,
    update_episodes: bool,
    correct_fps: bool,
    target_fps: int,
    correct_robot_type: bool,
    robot_type: str,
    push_to_hub: bool,
    output_repo_id: str,
    analyze_only: bool,
    verbose: bool,
    progress=gr.Progress()
) -> str:
    """Handle dataset processing with all available options"""
    try:
        # Initialize progress
        progress(0, desc="Initializing...")
        
        # Construct the base command
        cmd = ["python", "main.py", "--repo-id", repo_id]
        
        # Add output directory if specified
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        
        # Add processing options based on selected tasks
        if "Run Full Pipeline" in tasks:
            cmd.append("--all")
        if "Stabilize Videos" in tasks:
            cmd.append("--stabilize")
        if "Enhance Task Descriptions" in tasks:
            cmd.append("--enhance-tasks")
        if "Correct Metadata" in tasks:
            cmd.append("--correct-metadata")
        
        # Add video stabilization parameters
        if episodes:
            cmd.extend(["--episodes"] + episodes.split())
        if camera_keys:
            cmd.extend(["--camera-keys"] + camera_keys.split())
        cmd.extend(["--smoothing-window", str(smoothing_window)])
        cmd.extend(["--max-shift-percent", str(max_shift_percent)])
        
        # Add task enhancement parameters
        cmd.extend(["--model-name", model_name])
        cmd.extend(["--model-type", model_type])
        cmd.extend(["--max-length", str(max_length)])
        cmd.extend(["--sample-frames", str(sample_frames)])
        cmd.extend(["--frame-sampling", frame_sampling])
        if dry_run:
            cmd.append("--dry-run")
        
        # Add metadata correction parameters
        if update_episodes:
            cmd.append("--update-episodes")
        if correct_fps:
            cmd.append("--correct-fps")
        cmd.extend(["--target-fps", str(target_fps)])
        if correct_robot_type:
            cmd.append("--correct-robot-type")
        if robot_type:
            cmd.extend(["--robot-type", robot_type])
        
        # Add hub parameters
        if push_to_hub:
            cmd.append("--push-to-hub")
        if output_repo_id:
            cmd.extend(["--output-repo-id", output_repo_id])
        
        # Add other options
        if analyze_only:
            cmd.append("--analyze-only")
        if verbose:
            cmd.append("--verbose")
        
        # Execute the command
        progress(0.5, desc="Running processing command...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error during processing:\n{result.stderr}"
        
        # Finalize
        progress(1.0, desc="Finalizing...")
        
        return f"Successfully processed dataset: {repo_id}\nCommand output:\n{result.stdout}"
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Custom CSS for styling
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.main-header {
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.subtitle {
    text-align: center;
    color: #e2e8f0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.dataset-list {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

.video-container {
    position: relative;
    width: 100%;
    height: 300px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}

#component-0 {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(20px);
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="LeRobot Dataset Processing", theme=gr.themes.Soft()) as demo:
    
    # Header section
    gr.HTML("""
        <div style="text-align: center; color: white; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                LeRobot Dataset Processing
            </h1>
            <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                <a href="https://x.com/RemiCadene/status/1825455895561859185" 
                   target="_blank" 
                   style="color: #60a5fa; text-decoration: underline;">
                   create & train your own robots
                </a>
            </p>
        </div>
    """)
    
    # Video background (since Gradio doesn't support video backgrounds, we'll use an HTML component)
    gr.HTML("""
        <div style="position: relative; width: 100%; height: 250px; border-radius: 12px; overflow: hidden; margin-bottom: 2rem;">
            <video style="width: 100%; height: 100%; object-fit: cover;" autoplay muted loop>
                <source src="https://huggingface.co/datasets/cadene/koch_bimanual_folding/resolve/v1.6/videos/observation.images.phone_episode_000037.mp4" type="video/mp4">
                Your browser does not support HTML5 video.
            </video>
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.3);"></div>
        </div>
    """)
    
    # Main input section
    with gr.Column():
        # Required arguments
        with gr.Accordion("Required Arguments", open=True):
            repo_id = gr.Textbox(
                placeholder="username/dataset-name",
                label="Repository ID",
                elem_classes=["dataset-input"]
            )
            output_dir = gr.Textbox(
                placeholder="Output directory path",
                label="Output Directory",
                elem_classes=["path-input"]
            )
        
        # Processing options
        with gr.Accordion("Processing Options", open=True):
            task_checkboxes = gr.CheckboxGroup(
                choices=[
                    "Run Full Pipeline",
                    "Stabilize Videos",
                    "Enhance Task Descriptions",
                    "Correct Metadata"
                ],
                label="Select Tasks",
                value=[]
            )
        
        # Video stabilization parameters
        with gr.Accordion("Video Stabilization Parameters", open=False):
            episodes = gr.Textbox(
                placeholder="Space-separated episode numbers",
                label="Episodes to Process"
            )
            camera_keys = gr.Textbox(
                placeholder="Space-separated camera keys",
                label="Camera Keys to Process"
            )
            smoothing_window = gr.Slider(
                minimum=1,
                maximum=100,
                value=30,
                step=1,
                label="Smoothing Window"
            )
            max_shift_percent = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.01,
                label="Maximum Shift Percentage"
            )
        
        # Task enhancement parameters
        with gr.Accordion("Task Enhancement Parameters", open=False):
            model_name = gr.Textbox(
                value="Qwen/Qwen2.5-VL-3B-Instruct",
                label="Model Name"
            )
            model_type = gr.Dropdown(
                choices=["qwen2.5-vl", "llava-next"],
                value="qwen2.5-vl",
                label="Model Type"
            )
            max_length = gr.Slider(
                minimum=1,
                maximum=100,
                value=30,
                step=1,
                label="Max Length for Task Descriptions"
            )
            sample_frames = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of Frames to Sample"
            )
            frame_sampling = gr.Dropdown(
                choices=["uniform", "start", "middle", "end"],
                value="uniform",
                label="Frame Sampling Strategy"
            )
            dry_run = gr.Checkbox(label="Dry Run", value=False)
        
        # Metadata correction parameters
        with gr.Accordion("Metadata Correction Parameters", open=False):
            update_episodes = gr.Checkbox(label="Update Episode Metadata", value=False)
            correct_fps = gr.Checkbox(label="Correct FPS", value=False)
            target_fps = gr.Slider(
                minimum=1,
                maximum=120,
                value=30,
                step=1,
                label="Target FPS"
            )
            correct_robot_type = gr.Checkbox(label="Correct Robot Type", value=False)
            robot_type = gr.Textbox(
                placeholder="Robot type to set",
                label="Robot Type"
            )
        
        # Hub parameters
        with gr.Accordion("Hub Parameters", open=False):
            push_to_hub = gr.Checkbox(label="Push to Hub", value=False)
            output_repo_id = gr.Textbox(
                placeholder="Output repository ID",
                label="Output Repository ID"
            )
        
        # Other options
        with gr.Accordion("Other Options", open=False):
            analyze_only = gr.Checkbox(label="Analyze Only", value=False)
            verbose = gr.Checkbox(label="Verbose Logging", value=False)
        
        # Process button at the bottom
        go_button = gr.Button("Process", variant="primary", size="lg")
    
    # Output area
    output_text = gr.Textbox(
        label="Processing Result",
        interactive=False,
        visible=True
    )
    
    # Connect the main input and go button
    go_button.click(
        preprocess_dataset,
        inputs=[
            repo_id,
            output_dir,
            task_checkboxes,
            episodes,
            camera_keys,
            smoothing_window,
            max_shift_percent,
            model_name,
            model_type,
            max_length,
            sample_frames,
            frame_sampling,
            dry_run,
            update_episodes,
            correct_fps,
            target_fps,
            correct_robot_type,
            robot_type,
            push_to_hub,
            output_repo_id,
            analyze_only,
            verbose
        ],
        outputs=[output_text]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()