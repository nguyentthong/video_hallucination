"""
src/frame_selectors/_clip_shared.py
-------------------------------------
Lazy CLIP singleton shared by CLIPRetrievalSelector, AKSSelector, and EFSSelector.

Importing this module is free — CLIP weights are loaded only on the first call
to ``get_clip()``.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

_clip_model = None
_clip_proc = None


def get_clip(device: str = "cuda") -> Tuple:
    """Return (clip_model, clip_processor), loading on first call."""
    global _clip_model, _clip_proc
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = (
            CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float32)
            .to(device)
            .eval()
        )
        _clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return _clip_model, _clip_proc


@torch.inference_mode()
def clip_image_embed(pixel_values: torch.Tensor) -> torch.Tensor:
    """L2-normalised CLIP image embeddings. ``pixel_values`` must be on model device."""
    model, _ = get_clip()
    out = model.vision_model(pixel_values=pixel_values)
    feats = model.visual_projection(out.pooler_output)
    return F.normalize(feats, dim=-1)


@torch.inference_mode()
def clip_text_embed(question: str) -> torch.Tensor:
    """L2-normalised CLIP text embedding (CPU tensor)."""
    model, proc = get_clip()
    txt = proc(text=[question], return_tensors="pt", padding=True)
    out = model.text_model(
        input_ids=txt["input_ids"].to(model.device),
        attention_mask=txt["attention_mask"].to(model.device),
    )
    feats = model.text_projection(out.pooler_output)
    return F.normalize(feats, dim=-1).cpu()


@torch.inference_mode()
def score_frames_clip(
    frames: list,
    question: str,
    batch_size: int = 64,
) -> "torch.Tensor":
    """
    Cosine similarity between each frame and ``question`` via CLIP.

    Returns a 1-D CPU float tensor of length len(frames).
    """
    import torch
    model, proc = get_clip()
    txt_embed = clip_text_embed(question)   # (1, D)
    img_embeds = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i: i + batch_size]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(model.device)
        img_embeds.append(clip_image_embed(pv).cpu())
    img_embeds = torch.cat(img_embeds, dim=0)       # (N, D)
    return (img_embeds @ txt_embed.T).squeeze(-1)   # (N,)
