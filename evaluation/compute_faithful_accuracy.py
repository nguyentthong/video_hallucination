#!/usr/bin/env python3
"""
Compute Faithful Accuracy (A_faith) for every evaluated system on STRAND.

A_faith = (1/N) * sum_i  1[ target_i correct AND every sub-question of target_i correct ]

where N is ALL targets (977), not only the ones the model answered correctly.
The denominator is fixed, which is what makes A_faith comparable across models.

Input: a JSON or JSONL file of per-question predictions, one record per target:

    {"system": "Ours (Gemini-3-Flash)",
     "target_id": "v0007_q3",
     "target_correct": true,
     "sub_correct": [true, true, false, true]}

Adapt `load_records` to whatever your evaluation dumps. Everything else stays.

Usage:
    python3 compute_faithful_accuracy.py predictions.jsonl
    python3 compute_faithful_accuracy.py predictions.jsonl --latex
"""

import argparse
import json
import sys
from collections import defaultdict

N_TARGETS_EXPECTED = 977
N_SUBQ_EXPECTED = 2516


def load_records(path):
    """Yield dicts with keys: system, target_id, target_correct, sub_correct."""
    with open(path) as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            yield from json.load(fh)
        else:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def summarise(records):
    by_system = defaultdict(list)
    for r in records:
        by_system[r["system"]].append(r)

    out = {}
    for system, rows in by_system.items():
        n = len(rows)
        n_sub = sum(len(r["sub_correct"]) for r in rows)

        faith = sum(
            1 for r in rows if r["target_correct"] and all(r["sub_correct"])
        )
        target = sum(1 for r in rows if r["target_correct"])
        sub = sum(sum(r["sub_correct"]) for r in rows)

        # A_cons: over correctly answered targets, mean PROPORTION of sub-Qs correct
        correct_rows = [r for r in rows if r["target_correct"]]
        cons = (
            100.0
            * sum(sum(r["sub_correct"]) / len(r["sub_correct"]) for r in correct_rows)
            / len(correct_rows)
            if correct_rows
            else 0.0
        )

        out[system] = {
            "n_targets": n,
            "n_subq": n_sub,
            "A_faith": 100.0 * faith / n if n else 0.0,
            "A_target": 100.0 * target / n if n else 0.0,
            "A_sub": 100.0 * sub / n_sub if n_sub else 0.0,
            "A_cons": cons,
        }
    return out


def check(system, m):
    """Return a list of violated invariants. Empty list means the row is sound."""
    problems = []
    if m["n_targets"] != N_TARGETS_EXPECTED:
        problems.append(
            f"denominator is {m['n_targets']}, expected {N_TARGETS_EXPECTED} "
            "(A_faith must be over ALL targets)"
        )
    if m["n_subq"] != N_SUBQ_EXPECTED:
        problems.append(f"sub-question count is {m['n_subq']}, expected {N_SUBQ_EXPECTED}")
    if m["A_faith"] > m["A_target"] + 1e-9:
        problems.append(
            f"A_faith {m['A_faith']:.1f} > A_target {m['A_target']:.1f} (definitional)"
        )
    ceiling = m["A_target"] * m["A_cons"] / 100.0
    if m["A_faith"] > ceiling + 1e-9:
        problems.append(
            f"A_faith {m['A_faith']:.1f} > A_target*A_cons {ceiling:.1f} (definitional)"
        )
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    args = ap.parse_args()

    results = summarise(load_records(args.predictions))

    failed = False
    print(f"{'System':<26}{'A_faith':>9}{'A_target':>10}{'A_sub':>8}{'A_cons':>8}")
    for system, m in sorted(results.items(), key=lambda kv: -kv[1]["A_faith"]):
        print(
            f"{system:<26}{m['A_faith']:>9.1f}{m['A_target']:>10.1f}"
            f"{m['A_sub']:>8.1f}{m['A_cons']:>8.1f}"
        )
        for p in check(system, m):
            failed = True
            print(f"    !! {p}")

    if args.latex:
        print("\n% paste into tab:strand_results, then redo bold/underline")
        for system, m in results.items():
            print(
                f"{system} & {m['A_faith']:.1f} & {m['A_target']:.1f} & "
                f"{m['A_sub']:.1f} & {m['A_cons']:.1f} \\\\"
            )

    if failed:
        print(
            "\nSome rows violate an invariant. Fix the evaluation before "
            "putting these in the paper.",
            file=sys.stderr,
        )
        return 1
    print("\nAll rows satisfy A_faith <= A_target and A_faith <= A_target * A_cons.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
