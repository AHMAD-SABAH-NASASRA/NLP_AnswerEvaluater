import json

def convert(sample):

    rubric_text = "\n".join(
        [f"{k}: {v}" for k, v in sample["rubric"].items()]
    )

    return {
        "instruction": "Evaluate the student's answer.",
        "input": f"""Task:
{sample['task']}

Reference:
{sample['reference']}

Student Submission:
{sample['submission']}

Rubric:
{rubric_text}
""",
        "output": f"""Score: {sample['score']}
Rationale: {sample['rationale']}"""
    }

for split in ["train", "test"]:

    out = []

    with open(f"{split}.jsonl") as f:
        for line in f:
            out.append(convert(json.loads(line)))

    with open(f"{split}_instruct.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")

    print(split, len(out))