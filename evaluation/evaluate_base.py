import json
import re

from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error
from unsloth import FastLanguageModel
from transformers.utils import logging

logging.set_verbosity_error()

INPUT_PATH = "/home/mohammad/.ssh/data/processed/test_instruct.jsonl"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    test = [json.loads(line) for line in f]

print("Test samples:", len(test))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def build_prompt(sample):
    return f"""### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
"""

def extract_score(text):
    patterns = [
        r"Score:\s*([0-4])",
        r"score:\s*([0-4])",
        r"Score\s*=\s*([0-4])",
        r"score\s*=\s*([0-4])",
        r"\b([0-4])\b",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))

    return None

true_scores = []
pred_scores = []
failed = 0

for sample in tqdm(test):
    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = text[len(prompt):]

    pred = extract_score(generated)
    true = extract_score(sample["output"])

    if pred is not None and true is not None:
        true_scores.append(true)
        pred_scores.append(pred)
    else:
        failed += 1

print("Parsed samples:", len(true_scores))
print("Failed samples:", failed)

if len(true_scores) == 0:
    raise ValueError("No valid scores extracted.")

accuracy = accuracy_score(true_scores, pred_scores)
mae = mean_absolute_error(true_scores, pred_scores)

print("\n========== BASE RESULTS ==========")
print("Samples:", len(true_scores))
print("Accuracy:", round(accuracy, 4))
print("MAE:", round(mae, 4))