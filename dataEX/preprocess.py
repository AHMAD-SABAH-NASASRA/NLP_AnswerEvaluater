import json

INPUT_PATH = "data/train.jsonl"
OUTPUT_PATH = "data/train_instruct.jsonl"


def format_sample(sample):
    instruction = "Evaluate the student answer."

    rubric_text = "\n".join([f"{k}: {v}" for k, v in sample["rubric"].items()])

    input_text = f"""Task:
{sample['task']}

Reference:
{sample['reference']}

Submission:
{sample['submission']}

Rubric:
{rubric_text}
"""

    output_text = f"""Score: {sample['score']}
Rationale: {sample['rationale']}"""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def preprocess():
    data = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            formatted = format_sample(sample)
            data.append(formatted)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()