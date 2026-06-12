"""
src/frame_selectors/aks.py
---------------------------
Adaptive Keyframe Selection (AKS) frame sampling.

Algorithm
---------
1. Dense-extract candidate frames at ``candidate_fps``.
2. Score each frame via CLIP cosine similarity with the question.
3. Normalise scores to [0, 1].
4. Recursively split segments that are not yet "peaked":
     - if top-k mean >> overall mean AND std is large → keep segment
     - else split in half and recurse (up to ``all_depth`` levels)
5. From each final segment pick top-k frames (k ∝ 1 / 2^depth).
6. Return deduplicated frames in temporal order.
"""

import heapq
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import BaseFrameSelector
from ._video_io import dense_extract
from ._clip_shared import score_frames_clip


def _aks_split(
    dic_scores: list,
    fns: list,
    n: int,
    t1: float,
    t2: float,
    all_depth: int,
) -> Tuple[list, list]:
    """Recursive segment splitter. Returns (segments, frame_index_lists)."""
    split_scores, split_fn = [], []
    no_split_scores, no_split_fn = [], []

    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = np.mean(score)
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        mean_diff = np.mean([score[t] for t in top_n]) - mean

        if mean_diff > t1 and np.std(score) > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            split_scores += [
                {"score": score[:mid], "depth": depth + 1},
                {"score": score[mid:], "depth": depth + 1},
            ]
            split_fn += [fn[:mid], fn[mid:]]
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if split_scores:
        rec_scores, rec_fn = _aks_split(split_scores, split_fn, n, t1, t2, all_depth)
    else:
        rec_scores, rec_fn = [], []

    return no_split_scores + rec_scores, no_split_fn + rec_fn


class AKSSelector(BaseFrameSelector):
    """
    Adaptive Keyframe Selection — concentrates frame budget on
    informationally rich segments of the video.

    Parameters
    ----------
    num_frames : int
        Target number of frames to return.
    candidate_fps : float
        FPS for candidate frame extraction.  Default: 1.0.
    t1 : float
        Mean-diff threshold above which a segment is kept as-is.  Default: 0.8.
    t2 : float
        Std threshold (set very low to effectively only gate on t1).  Default: -100.
    all_depth : int
        Max recursion depth for splitting.  Default: 5.
    clip_batch_size : int
        Batch size for CLIP image encoding.  Default: 64.
    """

    def __init__(
        self,
        num_frames: int = 16,
        candidate_fps: float = 1.0,
        t1: float = 0.8,
        t2: float = -100.0,
        all_depth: int = 5,
        clip_batch_size: int = 64,
    ) -> None:
        super().__init__(num_frames, candidate_fps)
        self.t1 = t1
        self.t2 = t2
        self.all_depth = all_depth
        self.clip_batch_size = clip_batch_size

    def select(
        self,
        video_path: str,
        question: Optional[str] = None,
    ) -> List[Image.Image]:
        if question is None:
            raise ValueError("AKSSelector requires a question.")

        candidates, timestamps = dense_extract(video_path, self.candidate_fps)
        if len(candidates) <= self.num_frames:
            return candidates

        raw_scores = score_frames_clip(candidates, question, self.clip_batch_size).numpy()
        s_min, s_max = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - s_min) / (s_max - s_min + 1e-8)

        frame_indices = list(range(len(candidates)))
        segments, seg_frames = _aks_split(
            [{"score": norm_scores, "depth": 0}],
            [frame_indices],
            self.num_frames,
            self.t1,
            self.t2,
            self.all_depth,
        )

        selected: List[int] = []
        for seg, fidxs in zip(segments, seg_frames):
            k = max(1, int(self.num_frames / 2 ** seg["depth"]))
            topk = heapq.nlargest(k, range(len(seg["score"])), seg["score"].__getitem__)
            selected.extend(fidxs[t] for t in topk)

        selected = sorted(set(selected))
        return [candidates[i] for i in selected]
