import re
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------
# إعدادات
# ---------------------------
OUTPUT_PATH = "data/train.jsonl"
MAX_SAMPLES = 300  # عدّل حسب الحاجة

RUBRIC = {
    "0": "No understanding",
    "1": "Wrong reasoning and wrong answer",
    "2": "Partially correct reasoning but wrong answer",
    "3": "Correct reasoning with minor mistake OR missing step",
    "4": "Fully correct reasoning and answer"
}

# ---------------------------
# استخراج الجواب النهائي
# ---------------------------
def extract_final_answer(solution):
    match = re.findall(r"[-+]?\d*\.?\d+", solution)
    if match:
        return match[-1]
    return None

# ---------------------------
# توليد correct (paraphrase بسيط)
# ---------------------------
def generate_correct(reference):
    return reference.replace("=", " = ")

# ---------------------------
# توليد partial (خطأ حسابي بسيط)
# ---------------------------
def generate_partial(reference):
    nums = re.findall(r"\d+", reference)
    if not nums:
        return reference

    wrong_num = str(int(nums[-1]) + random.choice([1, 2, 3]))
    return reference.replace(nums[-1], wrong_num, 1)

# ---------------------------
# توليد wrong (تغيير العملية)
# ---------------------------
def generate_wrong(reference):
    if "/" in reference:
        return reference.replace("/", "*")
    elif "*" in reference:
        return reference.replace("*", "+")
    else:
        return reference + " + 10"

# ---------------------------
# توليد rationale
# ---------------------------
def generate_rationale(score):
    if score == 4:
        return "The solution is fully correct with proper reasoning."
    elif score == 3:
        return "The reasoning is correct but there is a small mistake in calculation."
    elif score == 2:
        return "The solution shows partial understanding but contains mistakes."
    elif score == 1:
        return "The reasoning is incorrect and leads to a wrong answer."
    else:
        return "The answer shows no understanding of the problem."

# ---------------------------
# بناء dataset
# ---------------------------
def build_dataset():
    dataset = load_dataset("gsm8k", "main")["train"]

    samples = []
    count = 0

    for item in tqdm(dataset):
        if count >= MAX_SAMPLES:
            break

        task = item["question"]
        reference = item["answer"]

        # استخراج answer
        final_answer = extract_final_answer(reference)

        # توليد submissions
        correct = generate_correct(reference)
        partial = generate_partial(reference)
        wrong = generate_wrong(reference)

        entries = [
            (correct, 4),
            (partial, 3),
            (wrong, 1),
        ]

        for submission, score in entries:
            sample = {
                "task": task,
                "reference": reference,
                "submission": submission,
                "rubric": RUBRIC,
                "score": score,
                "rationale": generate_rationale(score)
            }
            samples.append(sample)

        count += 1

    # حفظ JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()