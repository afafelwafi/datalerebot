import os
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

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

