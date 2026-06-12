# Methodology

## 1. Problem setting

Let $V$ denote a long-form video (typical length 5–60 minutes) and $q$ a natural-language question about $V$. In this work we restrict $q$ to yes/no questions and write the gold answer $y \in \{\text{yes}, \text{no}\}$. The task is to construct a function $f(V, q) \to \hat{y}$ that minimises mistakes, and in particular minimises *hallucinated* answers — answers that are confidently wrong because the model has fabricated entities, events, counts, or temporal relations not supported by the video.

Monolithic vision-language models (VLMs) — that is, models that consume frames and the question end-to-end and emit an answer in a single forward pass — exhibit characteristic failure modes on long $V$:

1. **Frame-budget pressure.** Open-source and proprietary VLMs cap the number of frames they can attend to (typically 32–128). For videos that are tens of minutes long, this implies sub-Hz sampling, which destroys most short-lived events.
2. **Long-context degradation.** When the language side of the model is asked to reason over the visual encoding of all frames at once, attention dilution and positional drift cause the model to lose track of temporal order and entity identity.
3. **Coreference fragility.** Recurring entities across chunks (the same person, object, or location described differently in different parts of $V$) are not linked, so the model treats them as separate and either invents new actors or merges distinct ones.

Each of these failures presents to the user as a confident hallucination rather than a refusal. Mitigating them requires forcing the model to *commit* to intermediate, inspectable representations rather than producing the answer in a single opaque pass.

## 2. Approach: a three-stage decomposition with structured intermediate states

We propose a pipeline that decomposes $f(V, q)$ into three stages, each with an explicit, inspectable intermediate representation:

$$V \;\;\xrightarrow{\text{A1: extract}}\;\; S \;\;\xrightarrow{\text{B: filter + link}}\;\; (S_q, A) \;\;\xrightarrow{\text{C: answer}}\;\; \hat{y}.$$

- **Stage A1 — Extractor (VLM).** Splits $V$ into fixed-length chunks and emits a structured timeline of events.
- **Stage B — Filter + identity linking (text-only LLM).** Uses $q$ to select the relevant events, and resolves recurring entities across chunks.
- **Stage C — Answerer (VLM).** Conditions on the filtered timeline *and* a sparse set of raw frames to commit to $\hat{y}$ with an explicit evidence trace.

The key design commitment is that **the interfaces between stages are model-agnostic data contracts (typed JSON, plain text), not model-specific representations.** Each stage can be instantiated by any model that satisfies the contract — proprietary or open-source, dense or mixture-of-experts, thinking or non-thinking. This allows us to treat each slot independently as an experimental degree of freedom and to characterise the contribution of each stage to overall accuracy and to hallucination reduction.

We describe each stage in detail below, then discuss the design properties that make the pipeline reproducible and extensible (Section 3) and the implementation details that affect reproducibility (Section 4).

## 2.1 Stage A1 — Visual state extractor

Given $V$, we partition the video into non-overlapping chunks $c_1, \ldots, c_n$ of fixed duration $\tau = 15$ seconds. From each chunk we sample $F_c = 60$ frames at uniform temporal stride and pass them, together with a structured-extraction prompt, to a vision-language model $M_{\text{A1}}$. The model emits a JSON object $s_i$ with the following fields:

| Field            | Type        | Role |
|------------------|-------------|------|
| `event_type`     | string      | Coarse label (e.g., `"open_box"`, `"throw"`, `"speak"`). |
| `description`    | string      | Free-text description of what happens in the chunk. |
| `sub_events`     | list of objects | Atomic decomposition of the chunk into individually-described actions, each with `description`, `actor`, `object`, `outcome`. |
| `outcome`        | string      | Free-text resolution / consequence of the chunk's events. |
| `start_time` / `end_time` | float (seconds) | Anchor each event to absolute video time. |

We concatenate the per-chunk outputs into a single timeline $S = [s_1, \ldots, s_n]$ that serves as the input to Stage B.

**Design rationale.** A natural alternative would be free-text per-chunk captions, which the answerer could read directly. We instead require the extractor to emit *structured* JSON with explicit fields for two reasons:

1. **Filtering tractability.** Stage B selects a subset of $S$ via attribute matching (Section 2.2). Free-text would require the filter to do free-text comparison, which is unreliable on long timelines. Structured fields make the matching task easier and more transparent.
2. **Hallucination localisation.** When the system produces a wrong answer, structured states let us determine whether the failure occurred in extraction, filtering, or answering. With free-text captions the failure surface is undifferentiated.

The fact that we observe Stage A1 to occasionally hallucinate — say, an `event_type` label that is not supported by `description` — is itself a measurable phenomenon and informs our prompt design (we instruct the extractor to ground every field in the visible frames and to leave fields empty rather than confabulate when the chunk does not contain the event).

## 2.2 Stage B — Question-conditioned filter and identity linking

Stage B performs two operations on $(q, S)$, both implemented as structured prompts to a text-only language model $M_{\text{B}}$.

**(a) Filtering.** We compute a question-relevant subset $S_q \subset S$ by asking $M_{\text{B}}$ to return a list of indices into $S$. The prompt instructs the model to assemble two groups of candidates and merge them:

- **Content matches:** events whose `description` or `sub_events.description` mention the question's objects, actors, or actions.
- **Ordinal matches:** events at the question's chronological position when $q$ contains a positional cue such as *first*, *third*, *last*, or refers to an N-th game / round / match. Ordinality is computed in `start_time` order.

The two-group merge is necessary because we observe that single-pass selection causes the language model to satisfy at most one constraint and drop the other — for instance, it returns events matching "fourth game" but ignores the constraint that they involve a "pink ball". Splitting the recall into independently-scored axes and unioning gives more reliable joint matching at modest additional cost.

The prompt also instructs $M_{\text{B}}$ to compare against `description` / `sub_events` text rather than against the coarse `event_type` label. This rule was added after observing Qwen-family filters dropping events whose descriptions clearly matched the question because their `event_type` label did not contain the question's keyword.

We tune the budget $k$ (number of events returned) per-question via a routing rule: on questions containing aggregation or ordinal cues ("how many", "every", an explicit ordinal), we boost $k$ from a default of 10 to 40 so the filter can pull evidence spread across the full timeline rather than clustering locally.

**(b) Identity linking.** Once per video, we build an alias table

$$A = \{\, \text{alias} \mapsto \text{canonical entity} \,\}$$

by passing the full timeline $S$ to $M_{\text{B}}$ with a prompt that asks it to identify recurring entities described differently across chunks (e.g., "the man in the red shirt" in chunk 3 and "the bearded man" in chunk 7) and to assign each one a canonical handle. The result is a header that is prepended to $S_q$ at answer time and tells the answerer to treat the listed surface forms as a single entity. We cache the alias table per video; it does not depend on $q$.

**Design rationale.** Stage B exists for two reasons:

1. **Context compression.** A long video may produce a timeline $S$ of tens of thousands of tokens. Asking the answerer to attend to all of it dilutes its attention over irrelevant material, which is a documented hallucination driver in long-context language models. A 10–40-event subset $S_q$ keeps the answerer focused.
2. **Cross-chunk entity coherence.** Without an explicit identity table, the answerer must rediscover entity identity from the surface forms alone, and (as discussed in Section 1) this is exactly where monolithic VLMs fail.

We deliberately use a text-only LLM for Stage B rather than a VLM — the input is already textual (the structured states), and a text model is faster, cheaper, and more reliable on the structured selection task.

## 2.3 Stage C — Frame-conditioned answerer

Stage C consumes $(q, S_q, A)$ along with $F_a = 64$ raw frames sampled uniformly from $V$ and emits a short reasoning trace of the form

```
Evidence: <one or two short sentences referencing frames or filtered events>
Answer: <yes | no>
```

The answerer is a vision-language model $M_{\text{C}}$ instructed to (i) trust the frames over the text if they disagree, (ii) refuse to invent facts not present in either the filtered text or the frames, and (iii) emit `Evidence` and `Answer` lines in the exact format above so they can be parsed mechanically.

**Design rationale.** Stage C has access to *both* the filtered text $S_q$ and the raw frames because each modality protects against a different failure mode:

- **Frames-only answering** (the monolithic VLM baseline) is subject to the frame-budget and long-context issues from Section 1.
- **Text-only answering** is subject to Stage A1's extraction errors: any hallucination introduced upstream is propagated to $\hat{y}$ with no opportunity for verification.
- **Both together** lets the answerer use $S_q$ as an *index* into the visual evidence — pre-localised events that tell the model where to look — while using the frames themselves as the ground-truth signal that overrides the text on disagreement.

This is the central hallucination-mitigation idea of the pipeline: by separating localisation (cheap, in text) from verification (expensive, on frames) we give each part of the system a constrained job that it can execute reliably, and we give the user an inspectable evidence chain when the system errs.

## 3. Design properties

### 3.1 Model-agnosticism

Each stage is specified by a typed I/O contract: Stage A1 takes (frames, prompt) and emits JSON conforming to the schema in Section 2.1; Stage B takes $(q, S)$ and emits a list of indices plus an alias table; Stage C takes $(q, S_q, A, \text{frames})$ and emits an Evidence/Answer pair. Any model satisfying the contract can fill the slot.

In practice we instantiate the slots with the following backends, all of which we have run end-to-end on our benchmark (see the experiment matrix in §5):

| Tag    | Model                                        | Type      | Slots used                |
|--------|----------------------------------------------|-----------|---------------------------|
| `gfl`  | Gemini 3 Flash                               | proprietary VLM/LLM | A1, B, C        |
| `q3vl` | Qwen3-VL-235B-A22B-Thinking                  | open-source VLM (MoE) | A1, C        |
| `q3t`  | Qwen3-235B-A22B                              | open-source text LLM (MoE) | B       |
| `q35`  | Qwen3.5-27B                                  | open-source dense VLM | C            |

The pipeline is invariant to which slot is filled by which backend. This lets us perform per-slot ablations: hold the design fixed, swap a single slot's model, and observe the change in accuracy. The design contribution is therefore separable from the choice of any specific frontier model.

### 3.2 Stage-keyed caching

All intermediate outputs are cached on disk. The cache is keyed by stage and by configuration so that runs sharing common stages can reuse work without conflating outputs:

- Stage A1 chunks (`chunks.json`, `stage_a_concat.txt`) are keyed by $(V, M_{\text{A1}})$ — runs that share the extractor share the chunks.
- Stage B filter outputs are keyed by $(V, q, M_{\text{B}})$ via a model-name-suffixed subdirectory (`filter_{model_tag}/`), so runs with different filters do not collide.
- Stage C answers are keyed by `prompt_method`, a string tag encoding the (filter, prompt-version, answerer) triple — answers from different configurations land in distinct subdirectories.
- Identity-linked alias tables are keyed by $V$ alone and are reused across all questions for that video.

This caching scheme makes it cheap to ablate any single slot: only the affected stage and downstream stages re-run. It also supports an experimental pattern in which a "primary" model (e.g., a frontier API) handles most calls and a "fallback" model (a local vLLM) automatically retries on transient failures — both writing into the same answer cache so that downstream evaluation sees a single coherent set of answers.

### 3.3 Hallucination as the principal failure mode of interest

Throughout the pipeline we adopt design choices that prioritise **traceability** over end-to-end accuracy. The structured-JSON Stage-A1 output, the explicit filter index list, the alias table, and the `Evidence:` line in Stage C answers all exist so that, when the system produces a wrong answer, the failure can be attributed to a specific stage and a specific intermediate representation. We argue that for long-form video QA — where errors are common and gold-standard frame-level annotations are scarce — this attribution is at least as valuable as raw accuracy: it makes the failures of the pipeline diagnosable in a way that monolithic VLM failures are not.

## 4. Implementation details

We list the hyperparameters and engineering choices that materially affect reproducibility.

- **Chunk granularity.** $\tau = 15$ s, $F_c = 60$ frames per chunk (4 fps). Smaller chunks fragment events that span boundaries; larger chunks force the extractor to summarise too aggressively.
- **Answerer frame budget.** $F_a = 64$ frames sampled uniformly from $V$. We also accept the model's natural sub-sampling when the served context window is the binding constraint.
- **Frame resolution.** Long-side 480 px, JPEG quality 85. Higher resolution increases token cost in proportion to area; 480 px is the lowest setting at which we did not observe significant degradation on the standard benchmark.
- **Aggregation routing.** A regex-based question classifier expands the filter top-$k$ from 10 to 40 when $q$ matches any of `\b(how many|total|more than|less than|first|second|...|last|each|every)\b`. This boosts recall on aggregation/ordinal questions where local clustering is harmful.
- **Token budgets.** Per-stage budgets are sized to allow thinking-style models to emit bounded reasoning before committing to output: $32{,}768$ tokens per chunk extraction, $8{,}192$ for filter, $16{,}384$ for the answerer. We document per-model adjustments where the defaults are insufficient.
- **Concurrency.** Per-question calls are issued in parallel through a thread pool ($\text{concurrency} = 4$ for VLM calls, $8$ for text-only filter). Cache reads and writes are atomic so concurrent workers do not race.
- **Provider routing.** When a slot is filled by a thinking model served via OpenRouter, we pin the request to a provider known to honour the chat-template kwargs that toggle thinking. We document the exact provider preferences in the experiment notes.

## 5. Where this leaves us

The pipeline as described is an artifact, not a result. The questions we use it to answer experimentally are:

1. **Does the decomposition reduce hallucination across multiple frontier and open-source models, or does it merely re-distribute errors?**
2. **Which slot's model choice carries the most weight, and how does that change with video length and question type?**
3. **Where does the pipeline still fail, and what taxonomy of residual hallucinations does the failure analysis reveal?**

The experimental matrix (Section 5 of the full paper) instantiates the pipeline at every cell needed to answer these questions, holding the pipeline design fixed and varying only the slot-level model choices and the difficulty tier of the benchmark.
