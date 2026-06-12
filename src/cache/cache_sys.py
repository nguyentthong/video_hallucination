"""
src/cache/cache_sys.py
-----------------------
Question-level answer cache.

Design
------
* One JSON file per video, stored under ``{cache_root}/{namespace}/``.
* ``namespace`` = ``model.cache_namespace`` = ``{model_id}__{prompt_method}``
  so a prompt change automatically writes to a fresh subdirectory.
* Within each JSON file, answers are keyed by a short hash of the question
  text.  Adding new questions to a video re-uses existing answers and only
  calls the model for the new ones.

Directory layout
----------------
    cache/
      Qwen_Qwen3-VL-8B-Instruct__vanilla/
        0016_NtTb-Cw6JVs.json      # { "<q_hash>": "Yes", ... }
      claude-3-7-sonnet__cot/
        0016_NtTb-Cw6JVs.json
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_hash(question: str) -> str:
    """Short, stable hash of a question string used as the JSON key."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:12]


def get_video_id(video_name: str) -> str:
    """
    Derive a filesystem-safe cache ID from a video filename.

    Strips the file extension so that ``"0016_NtTb-Cw6JVs.mp4"`` and
    ``"0016_NtTb-Cw6JVs"`` resolve to the same cache entry.
    """
    return Path(video_name).stem


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------

class AnswerCache:
    """
    Per-namespace cache that stores one answer string per (video, question).

    Parameters
    ----------
    cache_root : str
        Root directory for all caches.  Default: ``"cache"``.
    namespace : str
        Sub-directory name, typically ``model.cache_namespace``.
        Example: ``"Qwen_Qwen3-VL-8B-Instruct__vanilla"``.
    """

    def __init__(self, namespace: str, cache_root: str = "cache") -> None:
        self.cache_dir = Path(cache_root) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, video_id: str, question: str) -> Optional[str]:
        """
        Return the cached answer for (video_id, question), or None if absent.
        """
        store = self._load(video_id)
        return store.get(_question_hash(question))

    def get_many(
        self, video_id: str, questions: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Batch lookup.  Returns ``{question: answer_or_None}`` for every
        question in the list.
        """
        store = self._load(video_id)
        return {q: store.get(_question_hash(q)) for q in questions}

    def put(self, video_id: str, question: str, answer: str) -> None:
        """Store a single (question, answer) pair."""
        store = self._load(video_id)
        store[_question_hash(question)] = answer
        self._save(video_id, store)

    def put_many(self, video_id: str, qa_pairs: Dict[str, str]) -> None:
        """
        Store multiple (question, answer) pairs in a single write.

        Parameters
        ----------
        qa_pairs : Dict[str, str]
            ``{question_text: answer_text}`` mapping.
        """
        store = self._load(video_id)
        for question, answer in qa_pairs.items():
            store[_question_hash(question)] = answer
        self._save(video_id, store)

    def missing(self, video_id: str, questions: List[str]) -> List[str]:
        """Return the subset of *questions* that have no cached answer yet."""
        store = self._load(video_id)
        return [q for q in questions if _question_hash(q) not in store]

    def resolve(
        self,
        video_id: str,
        questions: List[str],
        model,
        video_path: str,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        High-level helper: return answers for all questions, running the
        model only for those that are not already cached.

        Parameters
        ----------
        video_id : str
            Cache key derived from the video filename.
        questions : List[str]
            All questions to answer.
        model : BaseVideoQAModel
            The model to call for uncached questions.
        video_path : str
            Path to the video file (passed to the model).
        max_new_tokens : int
            Forwarded to ``model.answer_questions``.

        Returns
        -------
        List[str]
            Answers in the same order as *questions*.
        """
        cached = self.get_many(video_id, questions)
        uncached_qs = [q for q, a in cached.items() if a is None]

        if uncached_qs:
            new_answers = model.answer_questions(
                video_path, uncached_qs, max_new_tokens=max_new_tokens
            )
            new_pairs = dict(zip(uncached_qs, new_answers))
            self.put_many(video_id, new_pairs)
            cached.update(new_pairs)

        return [cached[q] for q in questions]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}.json"

    def _load(self, video_id: str) -> Dict[str, str]:
        path = self._cache_path(video_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self, video_id: str, store: Dict[str, str]) -> None:
        path = self._cache_path(video_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_cache(model, cache_root: str = "cache") -> AnswerCache:
    """
    Create an AnswerCache scoped to *model*'s namespace.

    Usage
    -----
    >>> cache = make_cache(model)
    >>> answers = cache.resolve(video_id, questions, model, video_path)
    """
    return AnswerCache(namespace=model.cache_namespace, cache_root=cache_root)