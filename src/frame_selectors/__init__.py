"""
src/frame_selectors/__init__.py
---------------------------------
Frame selector registry.

Usage
-----
    from src.frame_selectors import load_selector

    selector = load_selector("clip", num_frames=16)
    frames   = selector.select(video_path, question)

Available names
---------------
    "uniform"  → UniformSelector      (no query needed)
    "clip"     → CLIPRetrievalSelector
    "aks"      → AKSSelector
    "efs"      → EFSSelector
"""

from typing import Any

from .base import BaseFrameSelector

__all__ = ["BaseFrameSelector", "load_selector"]

_REGISTRY = {
    "uniform": "src.frame_selectors.uniform.UniformSelector",
    "clip":    "src.frame_selectors.clip_retrieval.CLIPRetrievalSelector",
    "aks":     "src.frame_selectors.aks.AKSSelector",
    "efs":     "src.frame_selectors.efs.EFSSelector",
}


def load_selector(name: str, **kwargs: Any) -> BaseFrameSelector:
    """
    Instantiate a frame selector by name.

    Parameters
    ----------
    name : str
        One of ``"uniform"``, ``"clip"``, ``"aks"``, ``"efs"``.
    **kwargs
        Forwarded to the selector constructor (e.g. ``num_frames``,
        ``candidate_fps``, ``clip_batch_size``, …).

    Returns
    -------
    BaseFrameSelector
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown frame selector {name!r}. "
            f"Choose from: {list(_REGISTRY)}"
        )

    module_path, class_name = _REGISTRY[key].rsplit(".", 1)
    # Convert dotted path → relative import
    parts = module_path.split(".")          # ["src", "frame_selectors", "clip_retrieval"]
    rel = "." + ".".join(parts[2:])        # ".clip_retrieval"
    import importlib
    mod = importlib.import_module(rel, package="src.frame_selectors")
    cls = getattr(mod, class_name)
    return cls(**kwargs)
