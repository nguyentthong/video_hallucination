"""
src/frame_selectors/uniform.py
--------------------------------
Uniform temporal frame sampling — the simplest baseline.
"""

from typing import List, Optional

from PIL import Image

from .base import BaseFrameSelector
from ._video_io import uniform_extract


class UniformSelector(BaseFrameSelector):
    """
    Pick ``num_frames`` frames at evenly-spaced temporal positions.
    Query-agnostic: ``question`` is ignored.
    """

    def select(
        self,
        video_path: str,
        question: Optional[str] = None,
    ) -> List[Image.Image]:
        return uniform_extract(video_path, self.num_frames)
