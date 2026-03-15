"""
Budget-Aware Video Understanding Pipeline
==========================================
Components:
  1. FrameExtractor      — sample frames at configurable FPS
  2. SceneSegmenter       — segment video into coherent scenes via DINO
  3. MonitorBuilder       — build U, B, I from scenes using a lightweight VLM
  4. QuestionRetriever    — select evidence segments up to budget
  5. CoreReasoner         — answer question given monitor + frames
  6. BenchmarkEvaluator   — compute accuracy, consistency, groundedness

Requirements:
  pip install torch torchvision transformers accelerate
  pip install decord opencv-python-headless
  pip install sentence-transformers
  pip install vllm  # for fast inference
"""

import os, json, math, re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image


# ===========================================================================
# 1. FRAME EXTRACTOR
# ===========================================================================
@dataclass
class FrameExtractorConfig:
    fps: float = 1.0            # frames per second to sample
    max_frames: int = 1800      # hard cap (30 min @ 1fps)
    resize_short: int = 384     # shortest side


class FrameExtractor:
    """Sample video frames at a fixed FPS."""

    def __init__(self, cfg: FrameExtractorConfig = None):
        self.cfg = cfg or FrameExtractorConfig()

    def extract(self, video_path: str) -> dict:
        """Returns {frames: list[PIL.Image], timestamps: list[float], fps: float, duration: float}."""
        try:
            from decord import VideoReader, cpu
        except ImportError:
            return self._extract_opencv(video_path)

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        native_fps = float(vr.get_avg_fps())
        duration = total / native_fps
        step = max(1, int(native_fps / self.cfg.fps))
        indices = list(range(0, total, step))[: self.cfg.max_frames]
        raw = vr.get_batch(indices).asnumpy()

        frames, timestamps = [], []
        for i, idx in enumerate(indices):
            img = Image.fromarray(raw[i])
            img = self._resize(img)
            frames.append(img)
            timestamps.append(idx / native_fps)

        return {"frames": frames, "timestamps": timestamps,
                "fps": self.cfg.fps, "duration": duration}

    def _extract_opencv(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / native_fps
        step = max(1, int(native_fps / self.cfg.fps))
        frames, timestamps = [], []
        idx = 0
        while cap.isOpened() and len(frames) < self.cfg.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = self._resize(img)
                frames.append(img)
                timestamps.append(idx / native_fps)
            idx += 1
        cap.release()
        return {"frames": frames, "timestamps": timestamps,
                "fps": self.cfg.fps, "duration": duration}

    def _resize(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.cfg.resize_short
        if min(w, h) <= s:
            return img
        scale = s / min(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ===========================================================================
# 2. SCENE SEGMENTER
# ===========================================================================
@dataclass
class SceneSegmenterConfig:
    similarity_threshold: float = 0.85   # cosine sim below this → scene cut
    min_scene_frames: int = 2
    embedding_model: str = "facebook/dinov2-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Scene:
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    frame_indices: list  # indices into the frame list


class SceneSegmenter:
    """Segment frames into scenes using DINOv2 embedding similarity."""

    def __init__(self, cfg: SceneSegmenterConfig = None):
        self.cfg = cfg or SceneSegmenterConfig()
        self._model = None
        self._transform = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self.cfg.embedding_model)
        self._model = AutoModel.from_pretrained(self.cfg.embedding_model).to(self.cfg.device).eval()

    @torch.no_grad()
    def _embed_frames(self, frames: list) -> np.ndarray:
        self._load_model()
        embeddings = []
        bs = 16
        for i in range(0, len(frames), bs):
            batch = frames[i : i + bs]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.cfg.device)
            out = self._model(**inputs).last_hidden_state[:, 0]  # CLS token
            embeddings.append(out.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def segment(self, frames: list, timestamps: list) -> list:
        """Returns list[Scene]."""
        if len(frames) < 2:
            return [Scene(0, len(frames) - 1, timestamps[0], timestamps[-1],
                          list(range(len(frames))))]

        embs = self._embed_frames(frames)
        # Cosine similarity between consecutive frames
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs_n = embs / norms
        sims = np.sum(embs_n[:-1] * embs_n[1:], axis=1)

        # Find scene boundaries
        cuts = [0]
        for i, s in enumerate(sims):
            if s < self.cfg.similarity_threshold:
                cuts.append(i + 1)
        cuts.append(len(frames))

        scenes = []
        for i in range(len(cuts) - 1):
            si, ei = cuts[i], cuts[i + 1] - 1
            if ei - si + 1 < self.cfg.min_scene_frames and scenes:
                # Merge tiny scenes with previous
                scenes[-1] = Scene(
                    scenes[-1].start_idx, ei,
                    scenes[-1].start_time, timestamps[ei],
                    list(range(scenes[-1].start_idx, ei + 1))
                )
            else:
                scenes.append(Scene(si, ei, timestamps[si], timestamps[ei],
                                    list(range(si, ei + 1))))
        return scenes


# ===========================================================================
# 3. STATE MONITOR
# ===========================================================================
@dataclass
class EntityState:
    entity_id: str
    label: str
    states: list          # list of str
    time: float
    frame_idx: int


@dataclass
class Relation:
    subject_id: str
    object_id: str
    relation: str
    time: float


@dataclass
class IdentityLink:
    entity_a: str
    time_a: float
    entity_b: str
    time_b: float
    confidence: float


@dataclass
class Monitor:
    """Spatio-Temporal Monitor M = (U, B, I)."""
    U: list = field(default_factory=list)   # list[EntityState]
    B: list = field(default_factory=list)   # list[Relation]
    I: list = field(default_factory=list)   # list[IdentityLink]

    def to_text(self, max_entries: int = 200) -> str:
        """Serialize monitor to text for the core model."""
        lines = ["=== MONITOR STATE ==="]

        lines.append("\n[Entities & States (U)]")
        for e in self.U[:max_entries]:
            lines.append(f"  t={e.time:.1f}s: {e.label} ({e.entity_id}) — {', '.join(e.states)}")

        lines.append("\n[Relations (B)]")
        for r in self.B[:max_entries]:
            lines.append(f"  t={r.time:.1f}s: {r.subject_id} —[{r.relation}]→ {r.object_id}")

        lines.append("\n[Identity Links (I)]")
        for il in self.I[:max_entries]:
            lines.append(f"  ({il.entity_a}@{il.time_a:.1f}s) ≡ ({il.entity_b}@{il.time_b:.1f}s)"
                         f" [conf={il.confidence:.2f}]")
        return "\n".join(lines)


@dataclass
class MonitorBuilderConfig:
    vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 512
    sample_per_scene: int = 3   # frames to caption per scene


class MonitorBuilder:
    """Build the monitor M=(U,B,I) using a lightweight VLM."""

    ENTITY_PROMPT = (
        "List every distinct object/person visible. For each, give:\n"
        "- id (e.g., person_1, suitcase_0)\n"
        "- label (e.g., woman, red suitcase)\n"
        "- states (e.g., standing, open, leftmost)\n"
        "- relations with other entities (e.g., holding cup, behind person_2)\n"
        "Output JSON: {\"entities\": [{\"id\": ..., \"label\": ..., \"states\": [...], "
        "\"relations\": [{\"target\": ..., \"relation\": ...}]}]}"
    )

    IDENTITY_PROMPT = (
        "You see two frames from different times in a video.\n"
        "Frame A (t={t_a:.1f}s) entities: {entities_a}\n"
        "Frame B (t={t_b:.1f}s) entities: {entities_b}\n"
        "Which entities in Frame A correspond to the same entity in Frame B?\n"
        "Output JSON: {\"links\": [{\"a\": ..., \"b\": ..., \"confidence\": 0.0-1.0}]}"
    )

    def __init__(self, cfg: MonitorBuilderConfig = None):
        self.cfg = cfg or MonitorBuilderConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.cfg.vlm_model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.vlm_model, torch_dtype=torch.bfloat16,
            device_map=self.cfg.device
        )

    def _query_vlm(self, images: list, text: str) -> str:
        self._load_model()
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
        return self._processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)

    def _parse_json(self, text: str) -> dict:
        # Extract JSON from model output
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def build(self, frames: list, timestamps: list, scenes: list) -> Monitor:
        """Build monitor from frames organized into scenes."""
        monitor = Monitor()

        # Phase 1: Extract entities and states per scene
        scene_entities = {}  # scene_idx -> list of entity dicts
        for si, scene in enumerate(scenes):
            # Sample frames from scene
            idxs = scene.frame_indices
            step = max(1, len(idxs) // self.cfg.sample_per_scene)
            sample_idxs = idxs[::step][: self.cfg.sample_per_scene]

            for fidx in sample_idxs:
                result = self._query_vlm([frames[fidx]], self.ENTITY_PROMPT)
                parsed = self._parse_json(result)
                entities = parsed.get("entities", [])

                for e in entities:
                    eid = f"s{si}_{e.get('id', 'unk')}"
                    monitor.U.append(EntityState(
                        entity_id=eid,
                        label=e.get("label", "unknown"),
                        states=e.get("states", []),
                        time=timestamps[fidx],
                        frame_idx=fidx
                    ))
                    for rel in e.get("relations", []):
                        monitor.B.append(Relation(
                            subject_id=eid,
                            object_id=f"s{si}_{rel.get('target', 'unk')}",
                            relation=rel.get("relation", "unknown"),
                            time=timestamps[fidx]
                        ))

                if si not in scene_entities:
                    scene_entities[si] = []
                scene_entities[si].extend(entities)

        # Phase 2: Cross-scene identity linking
        scene_indices = sorted(scene_entities.keys())
        for i in range(len(scene_indices) - 1):
            sa, sb = scene_indices[i], scene_indices[i + 1]
            ea = scene_entities.get(sa, [])
            eb = scene_entities.get(sb, [])
            if not ea or not eb:
                continue

            ea_str = ", ".join(e.get("id", "?") + ":" + e.get("label", "?") for e in ea)
            eb_str = ", ".join(e.get("id", "?") + ":" + e.get("label", "?") for e in eb)

            # Use representative frames from each scene
            fa_idx = scenes[sa].frame_indices[len(scenes[sa].frame_indices) // 2]
            fb_idx = scenes[sb].frame_indices[len(scenes[sb].frame_indices) // 2]

            prompt = self.IDENTITY_PROMPT.format(
                t_a=timestamps[fa_idx], entities_a=ea_str,
                t_b=timestamps[fb_idx], entities_b=eb_str
            )
            result = self._query_vlm([frames[fa_idx], frames[fb_idx]], prompt)
            parsed = self._parse_json(result)

            for link in parsed.get("links", []):
                monitor.I.append(IdentityLink(
                    entity_a=f"s{sa}_{link.get('a', '?')}",
                    time_a=timestamps[fa_idx],
                    entity_b=f"s{sb}_{link.get('b', '?')}",
                    time_b=timestamps[fb_idx],
                    confidence=float(link.get("confidence", 0.5))
                ))

        return monitor


# ===========================================================================
# 4. QUESTION-CONDITIONED RETRIEVER
# ===========================================================================
@dataclass
class RetrieverConfig:
    budget_seconds: float = 120.0       # max evidence duration
    clip_model: str = "google/siglip2-base-patch16-384"
    alpha: float = 0.6                  # weight for visual similarity
    beta: float = 0.4                   # weight for monitor overlap
    expansion_seconds: float = 2.0      # expand each segment by this
    output_fps: float = 2.0             # FPS of output frames
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QuestionRetriever:
    """Select evidence segments up to the time budget."""

    def __init__(self, cfg: RetrieverConfig = None):
        self.cfg = cfg or RetrieverConfig()
        self._model = None

    def _load_clip(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.cfg.clip_model)
        self._model = AutoModel.from_pretrained(self.cfg.clip_model).to(self.cfg.device).eval()

    @torch.no_grad()
    def _score_scenes_visual(self, frames: list, scenes: list, question: str) -> list:
        """Score each scene by CLIP similarity to the question."""
        self._load_clip()
        scores = []
        for scene in scenes:
            mid = scene.frame_indices[len(scene.frame_indices) // 2]
            img = frames[mid]
            inputs = self._processor(text=[question], images=[img],
                                     return_tensors="pt", padding=True).to(self.cfg.device)
            out = self._model(**inputs)
            # SigLIP returns logits_per_image
            if hasattr(out, "logits_per_image"):
                score = out.logits_per_image[0, 0].item()
            else:
                score = 0.0
            scores.append(score)
        return scores

    def _score_scenes_monitor(self, scenes: list, monitor: Monitor,
                               question: str, timestamps: list) -> list:
        """Score scenes by keyword overlap between question and monitor entries."""
        q_words = set(question.lower().split())
        scores = []
        for scene in scenes:
            t_start, t_end = scene.start_time, scene.end_time
            score = 0.0
            for u in monitor.U:
                if t_start <= u.time <= t_end:
                    entity_words = set(u.label.lower().split()) | set(
                        w.lower() for s in u.states for w in s.split()
                    )
                    score += len(q_words & entity_words)
            for b in monitor.B:
                if t_start <= b.time <= t_end:
                    rel_words = set(b.relation.lower().split())
                    score += len(q_words & rel_words)
            scores.append(score)
        # Normalize
        mx = max(scores) if scores and max(scores) > 0 else 1.0
        return [s / mx for s in scores]

    def retrieve(self, frames: list, timestamps: list,
                 scenes: list, monitor: Monitor, question: str) -> dict:
        """
        Select scenes up to budget. Returns:
          {selected_scenes, selected_frames, selected_timestamps, total_duration}
        """
        vis_scores = self._score_scenes_visual(frames, scenes, question)
        mon_scores = self._score_scenes_monitor(scenes, monitor, question, timestamps)

        # Combined score
        combined = []
        for i, scene in enumerate(scenes):
            dur = scene.end_time - scene.start_time + 2 * self.cfg.expansion_seconds
            score = self.cfg.alpha * vis_scores[i] + self.cfg.beta * mon_scores[i]
            combined.append((score, dur, i, scene))

        # Greedy selection by score, respecting budget
        combined.sort(key=lambda x: -x[0])
        selected = []
        total_dur = 0.0
        for score, dur, idx, scene in combined:
            if total_dur + dur <= self.cfg.budget_seconds:
                selected.append((idx, scene))
                total_dur += dur

        # Sort by time
        selected.sort(key=lambda x: x[1].start_time)

        # Collect frames at output FPS
        sel_frames, sel_timestamps = [], []
        for _, scene in selected:
            t_start = max(0, scene.start_time - self.cfg.expansion_seconds)
            t_end = scene.end_time + self.cfg.expansion_seconds
            for fidx in scene.frame_indices:
                ts = timestamps[fidx]
                if t_start <= ts <= t_end:
                    sel_frames.append(frames[fidx])
                    sel_timestamps.append(ts)

        # Subsample to output FPS if needed
        if len(sel_timestamps) > 1:
            min_gap = 1.0 / self.cfg.output_fps
            filtered_f, filtered_t = [sel_frames[0]], [sel_timestamps[0]]
            for f, t in zip(sel_frames[1:], sel_timestamps[1:]):
                if t - filtered_t[-1] >= min_gap:
                    filtered_f.append(f)
                    filtered_t.append(t)
            sel_frames, sel_timestamps = filtered_f, filtered_t

        return {
            "selected_scenes": [s for _, s in selected],
            "selected_frames": sel_frames,
            "selected_timestamps": sel_timestamps,
            "total_duration": total_dur,
        }


# ===========================================================================
# 5. CORE REASONER
# ===========================================================================
@dataclass
class CoreReasonerConfig:
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 1024
    thinking: bool = True    # enable thinking mode


class CoreReasoner:
    """Answer the question using monitor text + retrieved frames."""

    SYSTEM_PROMPT = (
        "You are a precise video understanding assistant. You are given:\n"
        "1. A structured monitor of entities, states, relations, and identity links.\n"
        "2. Key video frames selected as evidence.\n"
        "3. A yes/no question.\n\n"
        "Think step by step. First identify which entities the question is about. "
        "Then trace their states and identities across time using the monitor. "
        "Finally cross-check with the visual evidence in the frames.\n"
        "Answer ONLY 'Yes' or 'No'."
    )

    def __init__(self, cfg: CoreReasonerConfig = None):
        self.cfg = cfg or CoreReasonerConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.cfg.model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model, torch_dtype=torch.bfloat16,
            device_map=self.cfg.device
        )

    def answer(self, monitor: Monitor, frames: list,
               timestamps: list, question: str) -> dict:
        self._load_model()

        # Build prompt content
        content = []
        # Add monitor text
        monitor_text = monitor.to_text(max_entries=100)
        content.append({"type": "text", "text": f"MONITOR:\n{monitor_text}\n\nEVIDENCE FRAMES:"})
        # Add frames with timestamps
        for img, ts in zip(frames, timestamps):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[t={ts:.1f}s]"})
        content.append({"type": "text", "text": f"\nQUESTION: {question}\nAnswer Yes or No."})

        if self.cfg.thinking:
            content[-1]["text"] += "\nThink step by step before answering."

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self._model.device)

        out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
        response = self._processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)

        # Parse answer
        answer = "unknown"
        resp_lower = response.lower().strip()
        if resp_lower.endswith("yes") or resp_lower.endswith("yes."):
            answer = "Yes"
        elif resp_lower.endswith("no") or resp_lower.endswith("no."):
            answer = "No"
        elif "yes" in resp_lower.split()[-5:]:
            answer = "Yes"
        elif "no" in resp_lower.split()[-5:]:
            answer = "No"

        return {"answer": answer, "reasoning": response}


# ===========================================================================
# 6. BENCHMARK EVALUATOR
# ===========================================================================
@dataclass
class BenchmarkSample:
    video_path: str
    question_id: int
    question: str
    answer: str                          # ground truth: "Yes" or "No"
    group_id: int                        # video-level group for consistency
    target_question_id: Optional[int] = None  # which Q this is a sub-Q of
    evidence_start: Optional[float] = None
    evidence_end: Optional[float] = None


class BenchmarkEvaluator:
    """Compute accuracy, consistency, and groundedness."""

    def __init__(self, samples: list):
        self.samples = samples
        self._build_groups()

    def _build_groups(self):
        # Group questions by target for consistency computation
        self.target_to_subs = {}  # target_qid -> [sub_qid, ...]
        for s in self.samples:
            if s.target_question_id is not None:
                tid = s.target_question_id
                if tid not in self.target_to_subs:
                    self.target_to_subs[tid] = []
                self.target_to_subs[tid].append(s.question_id)

    def evaluate(self, predictions: dict) -> dict:
        """
        predictions: {question_id: {"answer": str, "evidence_start": float, "evidence_end": float}}
        Returns metrics dict.
        """
        # Accuracy
        correct = 0
        total = 0
        for s in self.samples:
            pred = predictions.get(s.question_id, {})
            if pred.get("answer", "").lower() == s.answer.lower():
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0

        # Consistency
        cons_all_scores = []
        cons_tc_scores = []
        for target_qid, sub_qids in self.target_to_subs.items():
            all_qids = [target_qid] + sub_qids
            all_correct = []
            for qid in all_qids:
                gt = next((s.answer for s in self.samples if s.question_id == qid), None)
                pred = predictions.get(qid, {}).get("answer", "")
                all_correct.append(pred.lower() == gt.lower() if gt else False)

            # Consistency = fraction of sub-questions correct
            cons = sum(all_correct) / len(all_correct) if all_correct else 0
            cons_all_scores.append(cons)

            # Target-correct consistency
            target_gt = next((s.answer for s in self.samples
                              if s.question_id == target_qid), None)
            target_pred = predictions.get(target_qid, {}).get("answer", "")
            if target_gt and target_pred.lower() == target_gt.lower():
                cons_tc_scores.append(cons)

        cons_all = np.mean(cons_all_scores) if cons_all_scores else 0.0
        cons_tc = np.mean(cons_tc_scores) if cons_tc_scores else 0.0

        # Groundedness (temporal IoU)
        tiou_scores = []
        for s in self.samples:
            if s.evidence_start is None or s.evidence_end is None:
                continue
            pred = predictions.get(s.question_id, {})
            p_start = pred.get("evidence_start")
            p_end = pred.get("evidence_end")
            if p_start is None or p_end is None:
                tiou_scores.append(0.0)
                continue
            intersection = max(0, min(s.evidence_end, p_end) - max(s.evidence_start, p_start))
            union = max(s.evidence_end, p_end) - min(s.evidence_start, p_start)
            tiou = intersection / union if union > 0 else 0.0
            tiou_scores.append(tiou)

        tiou_mean = np.mean(tiou_scores) if tiou_scores else 0.0
        tiou_at_05 = np.mean([1 if t >= 0.5 else 0 for t in tiou_scores]) if tiou_scores else 0.0

        return {
            "accuracy": accuracy,
            "consistency_all": cons_all,
            "consistency_target_correct": cons_tc,
            "groundedness_mean_tiou": tiou_mean,
            "groundedness_tiou@0.5": tiou_at_05,
            "num_samples": total,
        }


# ===========================================================================
# 7. FULL PIPELINE
# ===========================================================================
@dataclass
class PipelineConfig:
    extractor: FrameExtractorConfig = field(default_factory=FrameExtractorConfig)
    segmenter: SceneSegmenterConfig = field(default_factory=SceneSegmenterConfig)
    monitor_builder: MonitorBuilderConfig = field(default_factory=MonitorBuilderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reasoner: CoreReasonerConfig = field(default_factory=CoreReasonerConfig)


class VideoUnderstandingPipeline:
    """End-to-end budget-aware video understanding."""

    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.extractor = FrameExtractor(self.cfg.extractor)
        self.segmenter = SceneSegmenter(self.cfg.segmenter)
        self.monitor_builder = MonitorBuilder(self.cfg.monitor_builder)
        self.retriever = QuestionRetriever(self.cfg.retriever)
        self.reasoner = CoreReasoner(self.cfg.reasoner)

    def __call__(self, video_path: str, question: str) -> dict:
        # Step 1: Extract frames
        data = self.extractor.extract(video_path)
        frames, timestamps = data["frames"], data["timestamps"]

        # Step 2: Segment into scenes
        scenes = self.segmenter.segment(frames, timestamps)

        # Step 3: Build monitor
        monitor = self.monitor_builder.build(frames, timestamps, scenes)

        # Step 4: Retrieve evidence within budget
        evidence = self.retriever.retrieve(
            frames, timestamps, scenes, monitor, question
        )

        # Step 5: Core model reasoning
        result = self.reasoner.answer(
            monitor,
            evidence["selected_frames"],
            evidence["selected_timestamps"],
            question,
        )

        return {
            "answer": result["answer"],
            "reasoning": result["reasoning"],
            "monitor": monitor,
            "evidence_duration": evidence["total_duration"],
            "num_evidence_frames": len(evidence["selected_frames"]),
            "num_scenes": len(scenes),
            "video_duration": data["duration"],
        }


# ===========================================================================
# 8. BENCHMARK RUNNER
# ===========================================================================
def run_benchmark(pipeline: VideoUnderstandingPipeline,
                  benchmark_path: str,
                  output_path: str,
                  budgets: list = None):
    """
    Run the pipeline on a benchmark JSON file.

    benchmark_path: JSON with list of BenchmarkSample dicts
    budgets: list of evidence budgets in seconds, e.g. [15, 30, 60, 120]
    """
    if budgets is None:
        budgets = [15, 30, 60, 120]

    with open(benchmark_path) as f:
        raw = json.load(f)
    samples = [BenchmarkSample(**s) for s in raw]

    results_by_budget = {}
    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Running with budget = {budget}s")
        print(f"{'='*60}")
        pipeline.retriever.cfg.budget_seconds = budget

        predictions = {}
        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}] Q{sample.question_id}: {sample.question[:60]}...")
            try:
                result = pipeline(sample.video_path, sample.question)
                predictions[sample.question_id] = {
                    "answer": result["answer"],
                    "reasoning": result["reasoning"],
                    "evidence_start": (result["monitor"].U[0].time
                                       if result["monitor"].U else None),
                    "evidence_end": (result["monitor"].U[-1].time
                                     if result["monitor"].U else None),
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                predictions[sample.question_id] = {"answer": "unknown"}

        evaluator = BenchmarkEvaluator(samples)
        metrics = evaluator.evaluate(predictions)
        metrics["budget_seconds"] = budget
        results_by_budget[budget] = metrics
        print(f"\n  Results @ {budget}s: Acc={metrics['accuracy']:.3f}, "
              f"Cons@All={metrics['consistency_all']:.3f}, "
              f"Cons@TC={metrics['consistency_target_correct']:.3f}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results_by_budget, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results_by_budget


# ===========================================================================
# 9. TRAINING UTILITIES (SFT + RL)
# ===========================================================================
def generate_sft_data(dataset_path: str, output_path: str,
                      teacher_model: str = "Qwen/Qwen3-VL-32B-Instruct",
                      num_samples: int = 20000):
    """
    Generate chain-of-thought SFT data using a teacher model.
    Input: JSON list of {video_path, question, answer}.
    Output: JSON list of {video_path, question, answer, reasoning}.
    """
    import random
    with open(dataset_path) as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:num_samples]

    # Use vLLM for fast batch inference
    print(f"Generating SFT data with {teacher_model}...")
    print(f"  This requires vLLM: `pip install vllm`")
    print(f"  Launch: `vllm serve {teacher_model} --tensor-parallel-size 4`")

    # Template for the teacher
    sft_template = (
        "Watch this video and answer the question: {question}\n\n"
        "Think step by step:\n"
        "1. Identify the key entities in the question.\n"
        "2. Track their states and identities across the video.\n"
        "3. Check relevant relations and temporal ordering.\n"
        "4. Conclude with Yes or No.\n\n"
        "The correct answer is: {answer}\n"
        "Now provide your detailed reasoning that leads to this answer."
    )

    # Placeholder — in practice you'd call the vLLM API
    sft_data = []
    for sample in data:
        sft_data.append({
            "video_path": sample["video_path"],
            "question": sample["question"],
            "answer": sample["answer"],
            "prompt": sft_template.format(**sample),
            # reasoning would be filled by the teacher model
        })

    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2)
    print(f"SFT template data saved to {output_path} ({len(sft_data)} samples)")


def prepare_rl_data(dataset_path: str, output_path: str):
    """
    Prepare RL training data with verifiable rewards.
    For yes/no questions, reward = 1 if answer matches, else 0.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    rl_data = []
    for sample in data:
        rl_data.append({
            "video_path": sample["video_path"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "reward_type": "binary_match",
            # For GRPO / T-GRPO training
        })

    with open(output_path, "w") as f:
        json.dump(rl_data, f, indent=2)
    print(f"RL data saved to {output_path} ({len(rl_data)} samples)")


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "benchmark", "gen_sft", "gen_rl"],
                        default="run")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "run":
        assert args.video and args.question, "Need --video and --question"
        cfg = PipelineConfig()
        cfg.retriever.budget_seconds = args.budget
        pipeline = VideoUnderstandingPipeline(cfg)
        result = pipeline(args.video, args.question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Evidence: {result['num_evidence_frames']} frames, "
              f"{result['evidence_duration']:.1f}s / {result['video_duration']:.1f}s total")
        print(f"Reasoning:\n{result['reasoning']}")

    elif args.mode == "benchmark":
        assert args.benchmark, "Need --benchmark"
        cfg = PipelineConfig()
        pipeline = VideoUnderstandingPipeline(cfg)
        run_benchmark(pipeline, args.benchmark, args.output)

    elif args.mode == "gen_sft":
        assert args.dataset, "Need --dataset"
        generate_sft_data(args.dataset, args.output)

    elif args.mode == "gen_rl":
        assert args.dataset, "Need --dataset"
        prepare_rl_data(args.dataset, args.output)