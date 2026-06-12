from types import SimpleNamespace

import src.models.vllm_openai as vllm_openai_module
from src.models import load_model
from src.models.vllm_openai import (
    VLLMOpenAIModel,
    _extract_frames_b64,
    _extract_response_text,
)


def test_load_model_routes_vllm_prefix_to_vllm_backend():
    model = load_model("vllm/Qwen/Qwen3-VL-32B-Thinking")

    assert isinstance(model, VLLMOpenAIModel)
    assert model.model_id == "Qwen/Qwen3-VL-32B-Thinking"


def test_vllm_build_content_omits_image_detail():
    model = VLLMOpenAIModel(model_id="Qwen/Qwen3-VL-32B-Thinking")

    content = model._build_content("What is happening?", ["abc123"])

    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert "detail" not in content[0]["image_url"]


def test_vllm_extract_frames_uses_pyav_sampler(monkeypatch):
    opened_paths = []

    class FakeContainer:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(frames=5)])

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def decode(self, _stream):
            return [SimpleNamespace(frame_idx=idx) for idx in range(5)]

    def fake_open(path):
        opened_paths.append(path)
        return FakeContainer()

    monkeypatch.setattr(vllm_openai_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(vllm_openai_module.av, "open", fake_open)
    monkeypatch.setattr(
        vllm_openai_module,
        "_encode_frame_b64",
        lambda frame: f"frame-{frame.frame_idx}",
    )

    assert _extract_frames_b64("clip.mp4", 3) == ["frame-0", "frame-2", "frame-4"]
    assert opened_paths == ["clip.mp4"]


def _response_with_message(**message_kwargs):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(**message_kwargs))]
    )


def test_vllm_extract_response_text_uses_reasoning_fallback():
    response = _response_with_message(
        content=None,
        reasoning=" Yes, there is a man on the right side. ",
    )

    assert _extract_response_text(response) == "Yes, there is a man on the right side."


class DumpOnlyMessage:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_vllm_extract_response_text_reads_reasoning_from_model_dump():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=DumpOnlyMessage(
                    {"content": None, "reasoning": " Model dump answer. "}
                )
            )
        ]
    )

    assert _extract_response_text(response) == "Model dump answer."


def test_vllm_extract_response_text_prefers_content_over_reasoning():
    response = _response_with_message(
        content=" Final answer. ",
        reasoning="Thinking text.",
    )

    assert _extract_response_text(response) == "Final answer."


def test_vllm_client_defaults_to_localhost_and_empty_key(monkeypatch):
    captured = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setattr("src.models.vllm_openai.OpenAI", DummyOpenAI)

    model = VLLMOpenAIModel(model_id="Qwen/Qwen3-VL-32B-Thinking")
    model._get_client()

    assert captured == {
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1",
    }


def test_load_model_passes_force_fps_to_local_qwen_backend(monkeypatch):
    from src import models as models_module

    captured = {}

    class DummyLocalModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(models_module, "_load_qwen3vl", lambda: DummyLocalModel)

    model = models_module.load_model(
        "weights/cosmos_reason2",
        prompt_method="vanilla",
        debug_with_n_frames=8,
        force_fps=4,
    )

    assert isinstance(model, DummyLocalModel)
    assert captured["model_id"] == "weights/cosmos_reason2"
    assert captured["debug_with_n_frames"] == 8
    assert captured["force_fps"] == 4
