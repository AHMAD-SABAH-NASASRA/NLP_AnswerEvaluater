import json
import os

INPUT = r"C:\Users\Sanasys\Downloads\dataEX\dataEX\data\gsm8k_test.jsonl"
OUTPUT = r"C:\Users\Sanasys\Downloads\dataEX\dataEX\data\test_instruct.jsonl"

# تأكد أن مجلد الإخراج موجود
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

RUBRIC = {
    "0": "No understanding",
    "1": "Wrong reasoning and wrong answer",
    "2": "Partially correct reasoning",
    "3": "Almost correct with minor mistake",
    "4": "Fully correct solution"
}

def convert(item):
    return {
        "task": item["question"],
        "reference": item["answer"],
        "submission": item["answer"],  # baseline (perfect case)
        "rubric": RUBRIC,
        "score": 4,
        "rationale": "Ground truth solution from dataset."
    }

data = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        data.append(convert(json.loads(line)))

with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Converted test set saved: {len(data)} samples")