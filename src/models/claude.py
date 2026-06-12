"""
src/models/claude.py
---------------------
Anthropic Claude API backend.

STATUS: Stub — not yet implemented.  The class is importable and will raise
NotImplementedError with a clear message when called, so the rest of the
pipeline can reference it without breaking.

Implementation notes (for when API access is available)
--------------------------------------------------------
* Use the `anthropic` SDK: ``pip install anthropic``
* Videos must be encoded as base64 image frames (Claude does not accept raw
  video files).  Extract N evenly-spaced frames with OpenCV, encode each as
  JPEG/PNG base64, and pass them as a sequence of `image` content blocks.
* Recommended frame count: 8-32 depending on video length.
* The API key is read from the env var defined in configs/models.yaml
  (default: ``ANTHROPIC_API_KEY``).

Example skeleton (do not delete — fill in when ready)
------------------------------------------------------
    import anthropic, base64, cv2

    client = anthropic.Anthropic(api_key=os.environ[api_key_env])
    frames_b64 = _extract_frames_b64(video_path, n_frames=self.n_frames)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}}
        for f in frames_b64
    ]
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(
        model=self.model_id,
        max_tokens=max_new_tokens,
        messages=[{"role": "user", "content": content}],
    )
    answer = response.content[0].text
"""

import os
from typing import List, Optional

from .base import BaseVideoQAModel


class ClaudeModel(BaseVideoQAModel):
    """
    Anthropic Claude API backend (stub).

    Parameters
    ----------
    model_id : str
        Claude model name, e.g. ``"claude-3-7-sonnet-20250219"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    n_frames : int
        Number of evenly-spaced frames to extract from the video and send as
        images.  Default: 16.
    api_key_env : str
        Name of the environment variable that holds the Anthropic API key.
        Default: ``"ANTHROPIC_API_KEY"``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = 16,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.n_frames = n_frames
        self.api_key_env = api_key_env

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        raise NotImplementedError(
            "ClaudeModel is not yet implemented.  "
            "See the docstring in src/models/claude.py for the implementation plan."
        )