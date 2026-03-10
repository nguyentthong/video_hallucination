from qwen_vl_utils.vision_process import fetch_video

video_config = {
    "type": "video",
    "video": "raw_data/0026_09uVWWLKNdc.mp4",
    "fps": 2,  # desired sample fps
}

# Get video with metadata
video, video_metadata = fetch_video(
    video_config,
    return_video_metadata=True
)

# Extract the original video FPS
original_fps = video_metadata["fps"]
print(f"Original video FPS: {original_fps}")