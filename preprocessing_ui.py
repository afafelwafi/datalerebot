import gradio as gr
from typing import List
import time


def preprocess_dataset(dataset_id: str, input_path: str, output_path: str, tasks: List[str], progress=gr.Progress()) -> str:
    """Handle navigation to dataset and preprocessing tasks"""
    # Simulate processing steps
    total_steps = len(tasks) + 2  # +2 for initialization and finalization
    progress(0, desc="Initializing...")
    time.sleep(0.5)  # Simulate initialization
    
    # Process each selected task
    for i, task in enumerate(tasks, 1):
        progress(i/total_steps, desc=f"Processing: {task}")
        time.sleep(1)  # Simulate task processing
    
    # Finalize
    progress(1.0, desc="Finalizing...")
    time.sleep(0.5)  # Simulate finalization
    
    task_str = ", ".join(tasks) if tasks else "No tasks selected"
    return f"Processing dataset: {dataset_id}\nInput path: {input_path}\nOutput path: {output_path}\nSelected tasks: {task_str}"

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
with gr.Blocks(css=custom_css, title="LeRobot Dataset Preprocess", theme=gr.themes.Soft()) as demo:
    
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
    with gr.Row():
        with gr.Column(scale=4):
            dataset_input = gr.Textbox(
                placeholder="enter dataset id (ex: lerobot/droid_100)",
                label="Dataset ID",
                elem_classes=["dataset-input"]
            )
            input_path = gr.Textbox(
                placeholder="Enter input path",
                label="Input Path",
                elem_classes=["path-input"]
            )
            output_path = gr.Textbox(
                placeholder="Enter output path",
                label="Output Path",
                elem_classes=["path-input"]
            )
            
            # Task checkboxes
            gr.Markdown("### Preprocessing Tasks")
            task_checkboxes = gr.CheckboxGroup(
                choices=[
                    "Resize Images",
                    "Normalize Data",
                    "Extract Features",
                    "Clean Annotations",
                    "Data Augmentation"
                ],
                label="Select Tasks",
                value=[]
            )
        
        with gr.Column(scale=1):
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
            dataset_input,
            input_path,
            output_path,
            task_checkboxes
        ],
        outputs=[output_text]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
    )
