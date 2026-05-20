import json

INPUT_PATH = "./data/evaluation/eval_outputs.json"
OUTPUT_PATH = "errors_2_3.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

errors = []

for i, x in enumerate(data):
    true_score = x["true_score"]
    pred_score = x["pred_score"]

    if true_score in [2, 3] and pred_score in [2, 3] and true_score != pred_score:
        errors.append({
            "index": i,
            "true_score": true_score,
            "pred_score": pred_score,
            "expected": x["expected"],
            "generated": x["generated"],
        })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(errors, f, indent=2, ensure_ascii=False)

print("2/3 errors:", len(errors))
print("Saved:", OUTPUT_PATH)
