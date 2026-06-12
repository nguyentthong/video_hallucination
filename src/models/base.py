"""
src/models/base.py
------------------
Abstract contract that every model backend must satisfy.

All backends — local HuggingFace checkpoints, Claude, Gemini, GPT, etc. —
inherit from BaseVideoQAModel and implement `answer_questions`.
The rest of the codebase only ever talks to this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVideoQAModel(ABC):
    """
    A model that can answer a batch of questions about a single video.

    Parameters
    ----------
    model_id : str
        Canonical identifier for the model (HuggingFace repo ID, API model name, …).
    prompt_method : str
        Short label for the prompt template in use.  Used as part of the cache
        namespace so that changing the prompt automatically invalidates old
        cached answers.  Default: "vanilla".
    """

    def __init__(self, model_id: str, prompt_method: str = "vanilla") -> None:
        self.model_id = model_id
        self.prompt_method = prompt_method

    # ------------------------------------------------------------------
    # Core interface — every subclass must implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Given a local video file path and a list of N questions, return a
        list of N answer strings in the same order.

        Parameters
        ----------
        video_path : str
            Absolute or relative path to the video file.
        questions : List[str]
            Questions to answer about the video.
        max_new_tokens : int
            Upper bound on generated tokens per answer (ignored by API
            backends that use their own defaults, but kept for a uniform
            signature).

        Returns
        -------
        List[str]
            One answer string per question, same length as `questions`.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @property
    def cache_namespace(self) -> str:
        """
        Directory-safe string used as the cache sub-folder name.
        Format: ``{sanitized_model_id}__{prompt_method}``

        The double-underscore separator makes it easy to split the two parts
        programmatically later if needed.

        Examples
        --------
        >>> model.cache_namespace
        'Qwen3-VL-8B-Instruct__vanilla'
        'claude-3-7-sonnet-20250219__cot'
        """
        safe_id = self.model_id.replace("/", "_").replace(":", "-")
        return f"{safe_id}__{self.prompt_method}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r}, prompt_method={self.prompt_method!r})"