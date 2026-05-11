import json
import random

INPUT_PATH = "/home/mohammad/.ssh/synthetic_gsm8k_grading.jsonl"

TRAIN_PATH = "/home/mohammad/.ssh/train_chat.jsonl"
TEST_PATH = "/home/mohammad/.ssh/test_chat.jsonl"

random.seed(42)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)

split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

def make_item(x):
    rubric_text = "\n".join([f"{k}: {v}" for k, v in x["rubric"].items()])

    user_prompt = f"""Evaluate the student's math answer.

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
print("Saved:", TRAIN_PATH, TEST_PATH)