import json
import re

with open("/home/mohammad/.ssh/data/evaluation/eval_outputs_base_mistral.json", "r", encoding="utf-8") as f:
    data = json.load(f)

patterns = {
    0: ["irrelevant", "unsupported", "unrelated", "no usable", "no valid"],
    1: ["wrong operation", "wrong mathematical path", "superficially", "main operation"],
    2: ["major", "significant", "conceptual", "breaks the problem logic"],
    3: ["minor", "small", "mostly correct", "close to correct", "final calculation"],
    4: ["correct reasoning", "correct final answer", "correct result", "matches the reference"],
}

bad = []

for i, x in enumerate(data):
    score = x["pred_score"]
    rationale = x["generated"].lower()

    matched = any(p in rationale for p in patterns.get(score, []))

    if not matched:
        bad.append({
            "index": i,
            "pred_score": score,
            "generated": x["generated"],
            "true_score": x["true_score"],
            "expected": x["expected"],
        })

print("\n===== RATIONALE-SCORE CONSISTENCY =====")
print("Total samples:", len(data))
print("Potential inconsistencies:", len(bad))
print("Consistency rate:", round((len(data) - len(bad)) / len(data), 4))

with open("rationale_inconsistencies.json", "w", encoding="utf-8") as f:
    json.dump(bad, f, indent=2, ensure_ascii=False)

print("Saved: rationale_inconsistencies.json")
