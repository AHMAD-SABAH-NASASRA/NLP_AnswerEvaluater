import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paths import (
    RAW_DATA,
    TRAIN_RAW,
    VAL_RAW,
    TEST_RAW,
    TRAIN_CHAT,
    VAL_CHAT,
    TEST_CHAT,
    make_dirs,
)

make_dirs()

INPUT_PATH = RAW_DATA

TRAIN_RAW_PATH = TRAIN_RAW
VAL_RAW_PATH = VAL_RAW
TEST_RAW_PATH = TEST_RAW

TRAIN_CHAT_PATH = TRAIN_CHAT
VAL_CHAT_PATH = VAL_CHAT
TEST_CHAT_PATH = TEST_CHAT

random.seed(42)

REQUIRED_FIELDS = [
    "task",
    "reference",
    "submission",
    "rubric",
    "score",
    "rationale",
]


def validate_item(x, idx):
    for field in REQUIRED_FIELDS:
        if field not in x:
            raise ValueError(f"Missing field '{field}' in item {idx}")

    if int(x["score"]) not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid score in item {idx}: {x['score']}")

    for s in ["0", "1", "2", "3", "4"]:
        if s not in x["rubric"]:
            raise ValueError(f"Missing rubric score {s} in item {idx}")


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

for i, x in enumerate(data):
    validate_item(x, i)

by_score = defaultdict(list)

for x in data:
    by_score[int(x["score"])].append(x)

train_data = []
val_data = []
test_data = []

for score, items in by_score.items():
    random.shuffle(items)

    n = len(items)

    n_train = int(n * 0.70)
    n_val = int(n * 0.10)

    train_data.extend(items[:n_train])
    val_data.extend(items[n_train:n_train + n_val])
    test_data.extend(items[n_train + n_val:])

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)


def make_chat_item(x):
    rubric_text = "\n".join(
        [
            f"Score {k}: {x['rubric'][k]}"
            for k in sorted(x["rubric"], key=lambda z: int(z))
        ]
    )

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
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": assistant_answer,
            },
        ]
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_chat_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(
                json.dumps(
                    make_chat_item(x),
                    ensure_ascii=False,
                ) + "\n"
            )


write_jsonl(TRAIN_RAW_PATH, train_data)
write_jsonl(VAL_RAW_PATH, val_data)
write_jsonl(TEST_RAW_PATH, test_data)

write_chat_jsonl(TRAIN_CHAT_PATH, train_data)
write_chat_jsonl(VAL_CHAT_PATH, val_data)
write_chat_jsonl(TEST_CHAT_PATH, test_data)


def print_dist(name, rows):
    print(f"\n{name}: {len(rows)}")

    for s in range(5):
        count = sum(1 for x in rows if int(x["score"]) == s)
        print(f"Score {s}: {count}")


print_dist("Train", train_data)
print_dist("Validation", val_data)
print_dist("Test", test_data)

print("\nSaved files:")
print(TRAIN_RAW_PATH)
print(VAL_RAW_PATH)
print(TEST_RAW_PATH)
print(TRAIN_CHAT_PATH)
print(VAL_CHAT_PATH)
print(TEST_CHAT_PATH)