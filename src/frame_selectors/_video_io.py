"""
src/frame_selectors/_video_io.py
---------------------------------
Shared video-decoding utilities used by all frame selectors.
"""

from pathlib import Path
from typing import List, Tuple

import av
from PIL import Image


def dense_extract(
    video_path: str | Path,
    candidate_fps: float = 1.0,
) -> Tuple[List[Image.Image], List[float]]:
    """
    Decode frames at ``candidate_fps`` fps.

    Returns
    -------
    frames : list of PIL Images (RGB)
    timestamps : list of floats (seconds)
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate)
    step = max(1, int(native_fps / candidate_fps))

    frames: List[Image.Image] = []
    timestamps: List[float] = []
    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            frames.append(frame.to_image())
            timestamps.append(i / native_fps)
    container.close()
    return frames, timestamps


def uniform_extract(
    video_path: str | Path,
    num_frames: int = 16,
) -> List[Image.Image]:
    """
    Pick ``num_frames`` evenly-spaced frames without decoding the whole video.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or int((stream.duration or 0) * float(stream.average_rate))
    if total == 0:
        total = sum(1 for _ in container.decode(stream))
        container.close()
        container = av.open(str(video_path))
        stream = container.streams.video[0]

    sample_indices = {int(i * total / num_frames) for i in range(num_frames)}
    frames: List[Image.Image] = []
    for i, frame in enumerate(container.decode(stream)):
        if i in sample_indices:
            frames.append(frame.to_image())
        if len(frames) == num_frames:
            break
    container.close()
    return frames
