"""
src/frame_selectors/clip_retrieval.py
---------------------------------------
CLIP-retrieval frame sampling.

Algorithm
---------
1. Dense-extract candidate frames at ``candidate_fps``.
2. Encode all candidates with CLIP image encoder (batched).
3. Encode ``question`` with CLIP text encoder.
4. Rank by cosine similarity → pick top ``num_frames``.
5. Return in temporal order.
"""

from typing import List, Optional

from PIL import Image

from .base import BaseFrameSelector
from ._video_io import dense_extract
from ._clip_shared import score_frames_clip


class CLIPRetrievalSelector(BaseFrameSelector):
    """
    Select the frames most semantically relevant to the question via CLIP.

    Parameters
    ----------
    num_frames : int
        Number of frames to return.
    candidate_fps : float
        FPS for candidate frame extraction.  Default: 1.0.
    clip_batch_size : int
        Batch size for CLIP image encoding.  Default: 64.
    """

    def __init__(
        self,
        num_frames: int = 16,
        candidate_fps: float = 1.0,
        clip_batch_size: int = 64,
    ) -> None:
        super().__init__(num_frames, candidate_fps)
        self.clip_batch_size = clip_batch_size

    def select(
        self,
        video_path: str,
        question: Optional[str] = None,
    ) -> List[Image.Image]:
        if question is None:
            raise ValueError("CLIPRetrievalSelector requires a question.")

        candidates, _ = dense_extract(video_path, self.candidate_fps)
        if len(candidates) <= self.num_frames:
            return candidates

        scores = score_frames_clip(candidates, question, self.clip_batch_size)
        top_indices = sorted(scores.topk(self.num_frames).indices.tolist())
        return [candidates[i] for i in top_indices]
