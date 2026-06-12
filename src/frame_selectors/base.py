"""
src/frame_selectors/base.py
----------------------------
Abstract contract for frame selectors.

A frame selector takes a video path (and optionally a question) and returns
a list of PIL Images to feed into the backbone model.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image


class BaseFrameSelector(ABC):
    """
    Select a fixed number of frames from a video for downstream QA.

    Parameters
    ----------
    num_frames : int
        Target number of frames to return.
    candidate_fps : float
        FPS at which to decode candidate frames before selection.
    """

    def __init__(self, num_frames: int = 16, candidate_fps: float = 1.0) -> None:
        self.num_frames = num_frames
        self.candidate_fps = candidate_fps

    @abstractmethod
    def select(
        self,
        video_path: str,
        question: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Return up to ``self.num_frames`` PIL Images in temporal order.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        question : str | None
            The question being asked.  Query-aware selectors (CLIP, AKS, EFS)
            use this; query-agnostic ones (Uniform) ignore it.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_frames={self.num_frames}, "
            f"candidate_fps={self.candidate_fps})"
        )
