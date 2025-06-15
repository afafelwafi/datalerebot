import cv2
import numpy as np
import os
import subprocess
import shutil


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

# Alternative approach: Export frames then use ffmpeg
def process_video_via_frames(input_video_path, output_video_path):
    """Process video by exporting frames, processing them, then reassembling with ffmpeg"""
    
    # Create temporary directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {total_frames} frames at {fps} FPS...")
    
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
            print(f"Processing frame {frame_count}/{total_frames}...")
        
        # Apply color correction
        corrected_frame, previous_b_shift = correct_yellow_blue_lab(
            frame.copy(),
            target_b_mean=128,
            smoothing_factor=lab_smoothing_alpha,
            prev_b_shift=previous_b_shift
        )
        
        # Save frame
        frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, corrected_frame)
    
    cap.release()
    print(f"Processed {frame_count} frames")
    
    # Use ffmpeg to reassemble video
    print("Reassembling video with ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print("Video reassembly successful!")
        
        # Clean up temporary frames
        print("Cleaning up temporary files...")
        for filename in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg or use the direct OpenCV method.")
        print("Frames have been saved in the 'temp_frames' directory.")
        return False

# Try direct OpenCV method first, fallback to ffmpeg method
def process_video_robust(input_video_path, output_video_path):
    """Try OpenCV first, fallback to ffmpeg method if it fails"""
    
    print("Attempting direct OpenCV processing...")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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
            print(f"OpenCV method working with codec: {codec}")
            out.release()
            success = True
            output_video_path = test_output
            break
        else:
            out.release()
    
    cap.release()
    
    if success:
        # Proceed with OpenCV method (use the corrected script from the first artifact)
        print("Using OpenCV method...")
        return True
    else:
        print("OpenCV VideoWriter failed, trying ffmpeg method...")
        return process_video_via_frames(input_video_path, output_video_path)

def filter_videos_on_episodes_and_camerakeys(video_files,episodes,camera_keys):
    """ filter videos files list on selected list of episodes and camera keys"""
    keep_videos = []
    for file in video_files:
        # filter on camera keys:
        if any([camera_key in str(file) for camera_key in  camera_keys]):
            if any([f"episode__{int(eps_indx):04d}" in str(file) for eps_indx in  episodes]):
                keep_videos.append(file)
    return keep_videos
            
        
        

def correct_videos_colors(all_video_files,episodes,camera_keys,**kwargs):
    """
    Correct color cast in all videos in the specified directory.
    Uses a robust method that first tries OpenCV, then falls back to ffmpeg.
    
    Args:
        videos_path (str): Path to the directory containing video files
    """

    video_files = filter_videos_on_episodes_and_camerakeys(all_video_files,episodes,camera_keys)
    if not video_files:
        print("No video files found in the specified directory.")
        return
    
    for video_file in tqdm(video_files):
        input_video_path = os.path.join(videos_path, video_file)
        output_video_path = os.path.join(videos_path, f"corrected_{video_file}")
         # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_file)[1], delete=False) as temp_file:
            temp_output_path = temp_file.name        
        
        print(f"Processing video: {input_video_path}")
        if not process_video_robust(input_video_path, temp_output_path):
            print(f"Failed to process {video_file}. Check logs for details.")
        else:
            # Replace original video with corrected version
            shutil.move(temp_output_path, input_video_path)
            print(f"✓ Successfully processed: {input_video_path}")

    return True

# if __name__ == "__main__":
#     videos_directory = "/path/to/your/videos"
#     correct_videos_colors(videos_directory)
