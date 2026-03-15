"""
Auto-generate candidate sub-questions for consistency evaluation.
Uses an LLM to decompose target questions into atomic predicates,
then converts each predicate into a yes/no sub-question.

Output: JSON ready for annotator verification.
"""

import json, os
from openai import OpenAI  # or anthropic SDK

DECOMPOSITION_PROMPT = """You are helping build a video understanding benchmark.

Given a yes/no question about a video and its ground-truth answer, your job is to
decompose the question into atomic facts that MUST ALL be true for the answer to be 
correct, then convert each fact into an independent yes/no sub-question.

Rules:
1. Each sub-question must be answerable from the video alone.
2. Each sub-question must be yes/no with a definite answer.
3. Tag each sub-question with its predicate type: 
   IDENTITY | STATE | RELATION | TEMPORAL | COUNT | EXISTENCE
4. The sub-question's answer must be logically NECESSARY for the 
   target answer to hold. If the target is "No", the sub-questions 
   should test the individual facts whose combination would be needed 
   for "Yes" — some of these facts will be true and some false.
5. Generate 2-4 sub-questions. Prefer fewer, higher-quality ones.
6. Do NOT generate sub-questions that are trivially obvious 
   (e.g., "Is this a video?" or "Are there people in the video?").

Output ONLY valid JSON (no markdown, no preamble):
{{
  "decomposition": [
    {{
      "predicate_type": "IDENTITY|STATE|RELATION|TEMPORAL|COUNT|EXISTENCE",
      "atomic_fact": "description of the fact being tested",
      "sub_question": "the yes/no question",
      "expected_answer": "Yes|No",
      "reasoning": "why this fact is necessary for the target answer"
    }}
  ]
}}

Target question: {question}
Ground-truth answer: {answer}
Video context (if available): {context}
"""


def generate_sub_questions(
    questions: list[dict],
    model: str = "gpt-4o",
    api_key: str = None,
) -> list[dict]:
    """
    Generate candidate sub-questions for a list of target questions.
    
    Input: list of dicts with keys: question_id, question, answer, video_title (optional)
    Output: list of dicts with original fields + 'candidate_sub_questions'
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    results = []

    for i, q in enumerate(questions):
        context = q.get("video_title", "No additional context")
        prompt = DECOMPOSITION_PROMPT.format(
            question=q["question"],
            answer=q["answer"],
            context=context,
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            subs = parsed.get("decomposition", [])
        except Exception as e:
            print(f"  [{i+1}] Error for Q{q['question_id']}: {e}")
            subs = []

        results.append({
            **q,
            "candidate_sub_questions": [
                {
                    "sub_question": s["sub_question"],
                    "expected_answer": s["expected_answer"],
                    "predicate_type": s["predicate_type"],
                    "atomic_fact": s["atomic_fact"],
                    "reasoning": s.get("reasoning", ""),
                    # Fields for annotator to fill:
                    "annotator_verdict": None,       # "accept" | "reject" | "edit"
                    "annotator_corrected_answer": None,
                    "annotator_corrected_text": None,
                }
                for s in subs
            ],
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(questions)}")

    return results


def build_annotation_file(generated: list[dict], output_path: str):
    """
    Create annotation file for human reviewers.
    Each sub-question gets a simple accept/reject/edit interface.
    """
    annotation_tasks = []
    sub_id_counter = 10000

    for item in generated:
        for sub in item["candidate_sub_questions"]:
            sub_id_counter += 1
            annotation_tasks.append({
                "task_id": sub_id_counter,
                "video_path": item.get("video_path", ""),
                "video_title": item.get("video_title", ""),
                # Target question context
                "target_question_id": item["question_id"],
                "target_question": item["question"],
                "target_answer": item["answer"],
                # Sub-question to verify
                "sub_question": sub["sub_question"],
                "proposed_answer": sub["expected_answer"],
                "predicate_type": sub["predicate_type"],
                "atomic_fact": sub["atomic_fact"],
                # Annotator fields
                "instructions": (
                    "Watch the video. Then answer these 3 questions:\n"
                    "1. Is the proposed answer correct? (yes/no/ambiguous)\n"
                    "2. Is this sub-question logically necessary for the "
                    "target question? (yes/no)\n"
                    "3. Is the sub-question clear and unambiguous? "
                    "(yes / no — if no, provide edited version)"
                ),
                "answer_correct": None,        # "yes" | "no" | "ambiguous"
                "logically_necessary": None,    # "yes" | "no"
                "clear_and_unambiguous": None,  # "yes" | "no"
                "edited_question": None,        # str if edited
                "corrected_answer": None,       # str if answer was wrong
                "annotator_id": None,
                "annotator_notes": None,
            })

    with open(output_path, "w") as f:
        json.dump(annotation_tasks, f, indent=2)
    print(f"Annotation file saved: {output_path}")
    print(f"  {len(annotation_tasks)} sub-questions to verify")
    print(f"  from {len(generated)} target questions")
    return annotation_tasks


def compile_final_benchmark(
    annotation_path: str,
    original_questions: list[dict],
    output_path: str,
):
    """
    After annotation, compile the final benchmark with verified groups.
    Keeps only sub-questions where:
      - answer_correct == "yes"
      - logically_necessary == "yes"  
      - clear_and_unambiguous == "yes"
    """
    with open(annotation_path) as f:
        annotations = json.load(f)

    # Filter accepted sub-questions
    accepted = [
        a for a in annotations
        if a.get("answer_correct") == "yes"
        and a.get("logically_necessary") == "yes"
        and a.get("clear_and_unambiguous") == "yes"
    ]

    # Group by target question
    target_to_subs = {}
    for a in accepted:
        tid = a["target_question_id"]
        if tid not in target_to_subs:
            target_to_subs[tid] = []
        target_to_subs[tid].append({
            "question": a.get("edited_question") or a["sub_question"],
            "answer": a.get("corrected_answer") or a["proposed_answer"],
            "predicate_type": a["predicate_type"],
        })

    # Build final benchmark
    final = []
    qid_counter = 0
    for orig in original_questions:
        qid_counter += 1
        target_entry = {
            **orig,
            "question_id": qid_counter,
            "target_question_id": None,  # this IS the target
            "sub_question_ids": [],
        }
        final.append(target_entry)

        subs = target_to_subs.get(orig["question_id"], [])
        for sub in subs:
            qid_counter += 1
            final.append({
                "video_path": orig.get("video_path", ""),
                "question_id": qid_counter,
                "question": sub["question"],
                "answer": sub["answer"],
                "group_id": target_entry["question_id"],
                "target_question_id": target_entry["question_id"],
                "predicate_type": sub["predicate_type"],
            })
            target_entry["sub_question_ids"].append(qid_counter)

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    groups_with_subs = sum(1 for q in final
                           if q["target_question_id"] is None
                           and len(q.get("sub_question_ids", [])) > 0)
    total_subs = sum(1 for q in final if q["target_question_id"] is not None)

    print(f"Final benchmark saved: {output_path}")
    print(f"  {len(final)} total questions")
    print(f"  {groups_with_subs} groups with sub-questions")
    print(f"  {total_subs} sub-questions total")
    print(f"  Avg {total_subs/max(groups_with_subs,1):.1f} subs per group")


# ---- Example usage ----
if __name__ == "__main__":
    sample_questions = [
        {
            "question_id": 1,
            "question": "Is the woman who walks toward us at the beginning the one who later stands up and leaves?",
            "answer": "No",
            "video_path": "videos/0045.mp4",
            "video_title": "Crazy Rich Asians - Mahjong Scene",
        },
        {
            "question_id": 3,
            "question": "Did the man on the left open three boxes?",
            "answer": "No",
            "video_path": "videos/0046.mp4",
            "video_title": "Box of Lies with Chris Pratt",
        },
        {
            "question_id": 12,
            "question": "Did he go to the laundry room before visiting his dorm room?",
            "answer": "Yes",
            "video_path": "videos/0048.mp4",
            "video_title": "College Dorm Tour 2025 | Stanford University",
        },
    ]

    # Step 1: Auto-generate candidates
    generated = generate_sub_questions(sample_questions)
    
    # Step 2: Create annotation file
    build_annotation_file(generated, "annotation_tasks.json")
    
    # Step 3: (Annotators fill in the file)
    
    # Step 4: Compile final benchmark
    # compile_final_benchmark("annotation_tasks_completed.json",
    #                         sample_questions, "benchmark_final.json")