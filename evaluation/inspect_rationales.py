import json
import random

INPUT_PATH = "/home/mohammad/.ssh/data/evaluation/eval_outputs.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)
samples = random.sample(data, min(20, len(data)))

for i, item in enumerate(samples, start=1):
    print("\n" + "=" * 80)
    print(f"Example {i}")
    print("=" * 80)

    print("\nTRUE:")
    print(item["expected"])

    print("\nPREDICTED:")
    print(item["generated"])
