import json
import random
from collections import defaultdict

INPUT_PATH = "/home/mohammad/.ssh/data/raw/synthetic_gsm8k_grading.jsonl"
TRAIN_PATH = "/home/mohammad/.ssh/data/processed/train_chat.jsonl"
TEST_PATH = "/home/mohammad/.ssh/data/processed/test_chat.jsonl"

random.seed(42)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

by_score = defaultdict(list)

for x in data:
    by_score[int(x["score"])].append(x)

train_data = []
test_data = []

for score, items in by_score.items():
    random.shuffle(items)
    split = int(len(items) * 0.8)
    train_data.extend(items[:split])
    test_data.extend(items[split:])

random.shuffle(train_data)
random.shuffle(test_data)

def make_item(x):
    rubric_text = "\n".join([f"Score {k}: {v}" for k, v in x["rubric"].items()])

    user_prompt = f"""You are a strict math-answer evaluator.

Evaluate the student's answer using the rubric.

Important distinction:
- Score 2: The answer has some relevant reasoning, but contains a major conceptual error, wrong setup, wrong operation, or broken logic.
- Score 3:
The submission is close to correct overall. This may happen in TWO cases:
1. The method is mostly correct but contains a minor arithmetic or final-step mistake.
2. The final answer is correct or nearly correct, but the reasoning/explanation is incomplete, weak, or insufficiently justified.

Return ONLY this format:
Score: <0-4>
Rationale: <brief explanation>

Task:
{x["task"]}

Reference Solution:
{x["reference"]}

Student Submission:
{x["submission"]}

Rubric:
{rubric_text}
"""

    assistant_answer = f"""Score: {x["score"]}
Rationale: {x["rationale"]}"""

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer}
        ]
    }

with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for x in train_data:
        f.write(json.dumps(make_item(x), ensure_ascii=False) + "\n")

with open(TEST_PATH, "w", encoding="utf-8") as f:
    for x in test_data:
        f.write(json.dumps(make_item(x), ensure_ascii=False) + "\n")

print("Train:", len(train_data))
print("Test:", len(test_data))

print("\nTrain score distribution:")
for s in sorted(by_score):
    print(s, sum(1 for x in train_data if int(x["score"]) == s))

print("\nTest score distribution:")
for s in sorted(by_score):
    print(s, sum(1 for x in test_data if int(x["score"]) == s))

print("\nSaved:", TRAIN_PATH, TEST_PATH)