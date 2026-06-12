"""
src/models/litellm.py
---------------------
OpenAI-compatible backend for LiteLLM proxy models.

The benchmark uses model IDs like ``"litellm/gemini-3-flash"``. The registry
strips the ``"litellm/"`` prefix before creating this class, so ``self.model_id``
is the model name exposed by the LiteLLM proxy.
"""

import os
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

from .openai_gpt import OpenAIModel

_DEFAULT_N_FRAMES = 32
_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_API_KEY_ENV = "LITELLM_API_KEY"
_DEFAULT_PROXY_URL_ENV = "LITELLM_PROXY_URL"
_FALLBACK_PROXY_URL_ENV = "LITELLM_BASE_URL"
_DEFAULT_MAX_FRAME_LONGEST_SIDE = 480


class LiteLLMModel(OpenAIModel):
    """
    OpenAI-compatible client for a LiteLLM proxy.

    Authentication and routing come from ``.env``/environment variables by
    default:

    * ``LITELLM_API_KEY`` for the proxy key
    * ``LITELLM_PROXY_URL`` for the proxy OpenAI-compatible base URL

    ``LITELLM_BASE_URL`` is accepted as a fallback alias for the proxy URL.
    Sampled frames are downscaled so their longest side is at most 480 pixels
    before they are sent to the proxy.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = _DEFAULT_N_FRAMES,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        base_url_env: str = _DEFAULT_PROXY_URL_ENV,
        base_url: Optional[str] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_frame_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
    ) -> None:
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        super().__init__(
            model_id=model_id,
            prompt_method=prompt_method,
            n_frames=n_frames,
            api_key_env=api_key_env,
            base_url=base_url,
            max_concurrency=max_concurrency,
            max_frame_longest_side=max_frame_longest_side,
        )
        self.base_url_env = base_url_env

    @property
    def cache_namespace(self) -> str:
        safe_id = f"litellm/{self.model_id}".replace("/", "_").replace(":", "-")
        return f"{safe_id}__{self.prompt_method}"

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        load_dotenv()

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set; "
                f"required for LiteLLM proxy model {self.model_id!r}."
            )

        base_url = self._resolve_base_url()
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def _resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url

        env_names = [self.base_url_env]
        if self.base_url_env == _DEFAULT_PROXY_URL_ENV:
            env_names.append(_FALLBACK_PROXY_URL_ENV)

        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                return value

        raise RuntimeError(
            f"None of the environment variables {env_names!r} are set; "
            f"required for LiteLLM proxy model {self.model_id!r}."
        )
