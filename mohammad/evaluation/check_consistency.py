import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paths import (
    FINETUNED_OUTPUT,
    RATIONALE_INCONSISTENCIES,
)

INPUT_PATH = str(FINETUNED_OUTPUT)
OUTPUT_PATH = str(RATIONALE_INCONSISTENCIES)


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


patterns = {
    0: [
        "irrelevant",
        "unsupported",
        "unrelated",
        "no usable",
        "no valid",
    ],
    1: [
        "wrong operation",
        "wrong mathematical path",
        "superficially",
        "main operation",
    ],
    2: [
        "major",
        "significant",
        "conceptual",
        "breaks the problem logic",
    ],
    3: [
        "minor",
        "small",
        "mostly correct",
        "close to correct",
        "final calculation",
    ],
    4: [
        "correct reasoning",
        "correct final answer",
        "correct result",
        "matches the reference",
    ],
}


bad = []


for i, x in enumerate(data):
    score = x["pred_score"]
    rationale = x["generated"].lower()

    matched = any(
        p in rationale
        for p in patterns.get(score, [])
    )

    if not matched:
        bad.append(
            {
                "index": i,
                "pred_score": score,
                "generated": x["generated"],
                "true_score": x["true_score"],
                "expected": x["expected"],
            }
        )


print("\n===== RATIONALE-SCORE CONSISTENCY =====")
print("Total samples:", len(data))
print("Potential inconsistencies:", len(bad))
print("Consistency rate:", round((len(data) - len(bad)) / len(data), 4))


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        bad,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("Saved:", OUTPUT_PATH)