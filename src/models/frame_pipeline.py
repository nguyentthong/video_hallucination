"""
src/models/frame_pipeline.py
------------------------------
FramePipelineModel — wraps a frame selector + Qwen3.5-VL backbone.

This model runs a frame selector on each video before calling the backbone,
allowing query-aware frame sampling strategies (CLIP, AKS, EFS) to be
plugged in transparently through the same ``BaseVideoQAModel`` interface.

Routing convention in ``load_model``
--------------------------------------
    "<selector>+<backbone_model_id>"

Examples:
    "uniform+Qwen/Qwen3.5-2B"
    "clip+Qwen/Qwen3.5-2B"
    "aks+Qwen/Qwen3.5-2B"
    "efs+Qwen/Qwen3.5-2B"
"""

from typing import List, Optional

from .base import BaseVideoQAModel


class FramePipelineModel(BaseVideoQAModel):
    """
    Composes a ``BaseFrameSelector`` with a ``Qwen35VLModel`` backbone.

    Parameters
    ----------
    selector_name : str
        One of ``"uniform"``, ``"clip"``, ``"aks"``, ``"efs"``.
    backbone_model_id : str
        HuggingFace repo ID for the Qwen3.5-VL checkpoint.
    prompt_method : str
        Propagated to the backbone (cache namespace).  Default: ``"vanilla"``.
    thinking : bool
        Enable thinking mode on the backbone.  Default: False.
    num_frames : int
        Frames to select per question.  Default: 16.
    candidate_fps : float
        FPS for candidate frame extraction.  Default: 1.0.
    selector_kwargs : dict | None
        Extra kwargs forwarded to the selector constructor.
    backbone_kwargs : dict | None
        Extra kwargs forwarded to ``Qwen35VLModel`` (e.g. ``max_pixels``).
    """

    def __init__(
        self,
        selector_name: str,
        backbone_model_id: str,
        prompt_method: str = "vanilla",
        thinking: bool = False,
        num_frames: int = 16,
        candidate_fps: float = 1.0,
        selector_kwargs: Optional[dict] = None,
        backbone_kwargs: Optional[dict] = None,
    ) -> None:
        # Cache namespace: "<selector>+<backbone>__<prompt_method>"
        super().__init__(
            model_id=f"{selector_name}+{backbone_model_id}",
            prompt_method=prompt_method,
        )
        self.selector_name = selector_name
        self.backbone_model_id = backbone_model_id
        self.thinking = thinking
        self.num_frames = num_frames
        self.candidate_fps = candidate_fps
        self._selector_kwargs = selector_kwargs or {}
        self._backbone_kwargs = backbone_kwargs or {}

        self._selector = None   # lazy
        self._backbone = None   # lazy

    # ------------------------------------------------------------------

    def _get_selector(self):
        if self._selector is None:
            from src.frame_selectors import load_selector
            self._selector = load_selector(
                self.selector_name,
                num_frames=self.num_frames,
                candidate_fps=self.candidate_fps,
                **self._selector_kwargs,
            )
        return self._selector

    def _get_backbone(self):
        if self._backbone is None:
            from .qwen35_vl import Qwen35VLModel
            self._backbone = Qwen35VLModel(
                model_id=self.backbone_model_id,
                prompt_method=self.prompt_method,
                thinking=self.thinking,
                **self._backbone_kwargs,
            )
        return self._backbone

    # ------------------------------------------------------------------
    # BaseVideoQAModel interface
    # ------------------------------------------------------------------

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        selector = self._get_selector()
        backbone = self._get_backbone()

        results = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            frames = selector.select(video_path, question)
            answer = backbone.answer_from_frames(frames, question, max_new_tokens)
            results.append(answer)
        return results
