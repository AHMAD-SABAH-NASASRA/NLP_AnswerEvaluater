import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from paths import MISTRAL_OUTPUT

INPUT_PATH = MISTRAL_OUTPUT

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
