import json
import re

from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error
from unsloth import FastLanguageModel
from transformers.utils import logging

logging.set_verbosity_error()

INPUT_PATH = "/home/mohammad/.ssh/data/processed/test_chat.jsonl"
MODEL_PATH = "/home/mohammad/.ssh/models/finetuned_chat"
OUTPUT_PATH = "/home/mohammad/.ssh/data/evaluation/eval_outputs.json"

# =========================
# Load dataset
# =========================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    test = [json.loads(line) for line in f]

print("Test samples:", len(test))

# =========================
# Load fine-tuned model
# =========================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# =========================
# Helpers
# =========================

def build_prompt(sample):
    user_message = sample["messages"][0]

    prompt = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def extract_score(text):
    patterns = [
        r"Score:\s*([0-4])",
        r"score:\s*([0-4])",
        r"re:\s*([0-4])",
        r"Score\s*=\s*([0-4])",
        r"score\s*=\s*([0-4])",
        r"score\s+is\s+([0-4])",
        r"The score is\s+([0-4])",
        r"\bscore\b[^0-9]{0,20}([0-4])",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


# =========================
# Evaluation
# =========================

true_scores = []
pred_scores = []
failed_outputs = []
eval_outputs = []

for sample in tqdm(test):

    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    generated = full_text.split("[/INST]")[-1].strip()

    pred = extract_score(generated)
    true = extract_score(sample["messages"][1]["content"])
    eval_outputs.append({
    "expected": sample["messages"][1]["content"],
    "generated": generated,
    "true_score": true,
    "pred_score": pred,
    })
    if pred is not None and true is not None:
        true_scores.append(true)
        pred_scores.append(pred)
    else:
        failed_outputs.append({
            "expected": sample["messages"][1]["content"],
            "generated": generated,
        })


# =========================
# Debug
# =========================

print("\nParsed samples:", len(true_scores))
print("Failed samples:", len(failed_outputs))

if len(failed_outputs) > 0:
    print("\n========== FIRST 10 FAILED OUTPUTS ==========")

    for i, item in enumerate(failed_outputs[:10], start=1):
        print(f"\n--- FAILED #{i} ---")
        print("EXPECTED:")
        print(item["expected"])
        print("\nGENERATED:")
        print(item["generated"][:1000])
        print("=" * 60)


# =========================
# Metrics
# =========================

if len(true_scores) == 0:
    raise ValueError("No valid scores extracted.")

accuracy = accuracy_score(true_scores, pred_scores)
mae = mean_absolute_error(true_scores, pred_scores)

print("\n========== FINETUNED CHAT RESULTS ==========")
print("Samples:", len(true_scores))
print("Accuracy:", round(accuracy, 4))
print("MAE:", round(mae, 4))


# =========================
# Score distribution
# =========================

print("\n========== SCORE DISTRIBUTION ==========")

for s in range(5):
    true_count = true_scores.count(s)
    pred_count = pred_scores.count(s)
    print(f"Score {s}: true={true_count}, pred={pred_count}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_outputs, f, ensure_ascii=False, indent=2)

print("\nSaved outputs to eval_outputs.json")