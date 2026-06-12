"""
src/frame_selectors/efs.py
---------------------------
Event-anchored Frame Selection (EFS).

Algorithm
---------
1. Dense-extract candidate frames at ``candidate_fps``.
2. Extract DINOv2 features for all frames (batched).
3. Segment video into scenes via weighted similarity + peak detection.
4. If query provided → score frames with CLIP, select best frame/scene
   then diversify with MMR.
5. If no query → select centre frame/scene + Farthest-Point Sampling.

Required extras: ``scipy``, ``scikit-learn``, and ``transformers`` (for DINOv2).
"""

import random
import warnings
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

from .base import BaseFrameSelector
from ._video_io import dense_extract
from ._clip_shared import score_frames_clip

warnings.filterwarnings("ignore")

DINO_MODEL_ID = "facebook/dinov2-base"

_dino_model = None
_dino_proc = None


def _load_dino(device: str = "cuda"):
    global _dino_model, _dino_proc
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        _dino_model = AutoModel.from_pretrained(DINO_MODEL_ID).to(device).eval()
        _dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    return _dino_model, _dino_proc


@torch.inference_mode()
def _extract_dino_features(
    frames: list,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    model, proc = _load_dino(device)
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i: i + batch_size]
        inp = proc(images=batch, return_tensors="pt").to(device)
        out = model(**inp)
        all_feats.append(out.last_hidden_state.mean(dim=1).cpu())
    return torch.cat(all_feats, dim=0).numpy()


def _weighted_similarity(features: np.ndarray, window_size: int = 3) -> np.ndarray:
    nf = torch.nn.functional.normalize(
        torch.from_numpy(features).float(), p=2, dim=1
    )
    half = window_size // 2
    scores = []
    for i in range(len(features)):
        sim, total_w = 0.0, 0.0
        for offset in range(-half, half + 1):
            j = i + offset
            if offset == 0 or j < 0 or j >= len(features):
                continue
            weight = max(0, half + 1 - abs(offset))
            s = torch.cosine_similarity(nf[i: i + 1], nf[j: j + 1]).item()
            sim += weight * s
            total_w += weight
        scores.append(sim / total_w if total_w > 0 else 1.0)
    return np.array(scores)


def _segment_scenes(
    features: np.ndarray,
    target_scenes: int,
    window_size: int = 3,
    min_scene_length: int = 5,
) -> list:
    sim = _weighted_similarity(features, window_size)
    minima, _ = find_peaks(-sim, distance=min_scene_length)
    boundaries = sorted(minima.tolist())

    if not boundaries:
        return [(0, features.shape[0])]

    scenes = [(0, boundaries[0])]
    for i in range(1, len(boundaries)):
        scenes.append((boundaries[i - 1], boundaries[i]))
    scenes.append((boundaries[-1], features.shape[0]))

    # Merge most-similar adjacent scenes until target count reached
    while len(scenes) > target_scenes:
        scene_feats = np.array([np.mean(features[s:e], axis=0) for s, e in scenes])
        sim_mat = sk_cosine_sim(scene_feats)
        best = max(range(len(scenes) - 1), key=lambda i: sim_mat[i, i + 1])
        scenes = (
            scenes[:best]
            + [(scenes[best][0], scenes[best + 1][1])]
            + scenes[best + 2:]
        )
    return scenes


def _fps_fill(features: np.ndarray, seed: list, k: int) -> list:
    """Farthest-Point Sampling to fill ``seed`` up to ``k`` indices."""
    norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    selected = seed.copy()
    while len(selected) < k:
        candidates = [i for i in range(len(features)) if i not in selected]
        if not candidates:
            break
        dists = [
            np.min(1 - np.dot(norm[c], norm[selected].T)) for c in candidates
        ]
        selected.append(candidates[int(np.argmax(dists))])
    return sorted(selected)[:k]


def _select_with_query(
    scenes: list,
    features: np.ndarray,
    itm_scores: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> list:
    """Best-frame-per-scene + MMR diversification."""
    scene_keys = [
        start + int(np.argmax(itm_scores[start:end]))
        for start, end in scenes
        if end > start
    ]
    if len(scene_keys) >= k:
        return sorted(scene_keys[:k])

    n = len(itm_scores)
    remain = sorted(
        [i for i in range(n) if i not in scene_keys],
        key=lambda i: itm_scores[i],
        reverse=True,
    )
    all_sel = scene_keys.copy()
    max_sims = [
        float(np.max(sk_cosine_sim(features[i: i + 1], features[all_sel])[0]))
        if all_sel else 0.0
        for i in remain
    ]
    mu, sigma = np.mean(max_sims), np.std(max_sims)
    strict = float(np.clip(mu - alpha * sigma, 0.0, 1.0))
    loose = float(np.clip(mu + alpha * sigma, 0.0, 1.0))
    threshold, delta = strict, 0.05

    while len(all_sel) < k and threshold <= loose:
        for idx in remain:
            if idx in all_sel or len(all_sel) >= k:
                continue
            max_sim = (
                float(np.max(sk_cosine_sim(features[idx: idx + 1], features[all_sel])[0]))
                if all_sel else 0.0
            )
            if max_sim < threshold:
                all_sel.append(idx)
        if len(all_sel) < k and threshold >= loose:
            break
        threshold = min(threshold + delta, loose)

    return sorted(set(all_sel))[:k]


def _select_visual_only(scenes: list, features: np.ndarray, k: int) -> list:
    scene_keys = sorted(set((s + e) // 2 for s, e in scenes if e > s))
    return _fps_fill(features, scene_keys, k)


class EFSSelector(BaseFrameSelector):
    """
    Event-anchored Frame Selection using DINOv2 scene segmentation.

    Parameters
    ----------
    num_frames : int
        Target frames to return.
    candidate_fps : float
        FPS for candidate extraction.  Default: 1.0.
    device : str
        Device for DINOv2 / CLIP.  Default: ``"cuda"``.
    window_size : int
        Temporal window for similarity smoothing.  Default: 3.
    min_scene_length : int
        Minimum frames per scene.  Default: 5.
    target_scenes : int
        Max scenes after merging.  Default: 10.
    alpha : float
        MMR diversity weight.  Default: 0.5.
    dino_batch_size : int
        Batch size for DINOv2.  Default: 32.
    clip_batch_size : int
        Batch size for CLIP scoring.  Default: 64.
    """

    def __init__(
        self,
        num_frames: int = 16,
        candidate_fps: float = 1.0,
        device: str = "cuda",
        window_size: int = 3,
        min_scene_length: int = 5,
        target_scenes: int = 10,
        alpha: float = 0.5,
        dino_batch_size: int = 32,
        clip_batch_size: int = 64,
    ) -> None:
        super().__init__(num_frames, candidate_fps)
        self.device = device
        self.window_size = window_size
        self.min_scene_length = min_scene_length
        self.target_scenes = target_scenes
        self.alpha = alpha
        self.dino_batch_size = dino_batch_size
        self.clip_batch_size = clip_batch_size

    def select(
        self,
        video_path: str,
        question: Optional[str] = None,
    ) -> List[Image.Image]:
        candidates, _ = dense_extract(video_path, self.candidate_fps)
        if len(candidates) <= self.num_frames:
            return candidates

        features = _extract_dino_features(candidates, self.device, self.dino_batch_size)
        scenes = _segment_scenes(
            features, self.target_scenes, self.window_size, self.min_scene_length
        )

        if question:
            itm_scores = score_frames_clip(
                candidates, question, self.clip_batch_size
            ).numpy()
            sampled = _select_with_query(scenes, features, itm_scores, self.num_frames, self.alpha)
        else:
            sampled = _select_visual_only(scenes, features, self.num_frames)

        sampled = sorted(set(sampled))[: self.num_frames]
        return [candidates[i] for i in sampled]
