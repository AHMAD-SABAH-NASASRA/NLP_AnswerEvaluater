import json
import re
import sys
from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from unsloth import FastLanguageModel
from transformers.utils import logging

logging.set_verbosity_error()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paths import (
    TEST_CHAT,
    BASE_QWEN_OUTPUT,
    make_dirs,
)

make_dirs()

INPUT_PATH = str(TEST_CHAT)
OUTPUT_PATH = str(BASE_QWEN_OUTPUT)

MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    test = [json.loads(line) for line in f if line.strip()]

print("Test samples:", len(test))


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


def build_prompt(sample):
    user_message = sample["messages"][0]

    return tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_score(text):
    if text is None:
        return None

    text = text.strip()

    patterns = [
        r"\*{0,2}\s*Score\s*\*{0,2}\s*[:=\-]\s*\*{0,2}\s*([0-4])",
        r"\bScore\b\s+([0-4])\b",
        r"\bscore\b\s+([0-4])\b",
        r"\bscore\b\s*(?:is|should be|would be|of)?\s*[:=\-]?\s*([0-4])\b",
        r"\bgrade\b\s*(?:is|should be|would be)?\s*[:=\-]?\s*([0-4])\b",
        r"\brating\b\s*(?:is|should be|would be)?\s*[:=\-]?\s*([0-4])\b",
        r"\b([0-4])\s*/\s*4\b",
        r"I would give.*?\b([0-4])\b",
        r"I would assign.*?\b([0-4])\b",
        r"This should receive.*?\b([0-4])\b",
        r"The answer deserves.*?\b([0-4])\b",
        r"The appropriate score.*?\b([0-4])\b",
        r"The correct score.*?\b([0-4])\b",
        r"Final score.*?\b([0-4])\b",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)

        if m:
            return int(m.group(1))

    first_line = text.splitlines()[0].strip()
    first_line = first_line.replace("*", "").strip()

    if re.fullmatch(r"[0-4]", first_line):
        return int(first_line)

    return None


true_scores_parsed = []
pred_scores_parsed = []

eval_outputs = []
failed_outputs = []


for sample in tqdm(test):
    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    expected = sample["messages"][1]["content"]

    pred = extract_score(generated)
    true = extract_score(expected)

    item = {
        "expected": expected,
        "generated": generated,
        "true_score": true,
        "pred_score": pred,
    }

    eval_outputs.append(item)

    if pred is not None and true is not None:
        true_scores_parsed.append(true)
        pred_scores_parsed.append(pred)

    else:
        failed_outputs.append(item)


total_samples = len(eval_outputs)
parsed_samples = len(true_scores_parsed)
failed_samples = len(failed_outputs)

correct_all = sum(
    1
    for x in eval_outputs
    if x["pred_score"] is not None
    and x["pred_score"] == x["true_score"]
)

accuracy_all = correct_all / total_samples


print("\n========== BASE QWEN RESULTS ==========")

print("Total samples:", total_samples)
print("Parsed samples:", parsed_samples)
print("Failed samples:", failed_samples)
print("Accuracy all samples:", round(accuracy_all, 4))


if parsed_samples > 0:
    acc_parsed = accuracy_score(
        true_scores_parsed,
        pred_scores_parsed,
    )

    mae_parsed = mean_absolute_error(
        true_scores_parsed,
        pred_scores_parsed,
    )

    qwk_parsed = cohen_kappa_score(
        true_scores_parsed,
        pred_scores_parsed,
        weights="quadratic",
    )

    print("Accuracy parsed only:", round(acc_parsed, 4))
    print("MAE parsed only:", round(mae_parsed, 4))
    print("QWK parsed only:", round(qwk_parsed, 4))

    print("\n========== CONFUSION MATRIX parsed only ==========")

    print(
        confusion_matrix(
            true_scores_parsed,
            pred_scores_parsed,
            labels=[0, 1, 2, 3, 4],
        )
    )

    print("\n========== CLASSIFICATION REPORT parsed only ==========")

    print(
        classification_report(
            true_scores_parsed,
            pred_scores_parsed,
            labels=[0, 1, 2, 3, 4],
            zero_division=0,
        )
    )


print("\n========== SCORE DISTRIBUTION ==========")

for s in range(5):
    true_count = sum(
        1
        for x in eval_outputs
        if x["true_score"] == s
    )

    pred_count = sum(
        1
        for x in eval_outputs
        if x["pred_score"] == s
    )

    print(f"Score {s}: true={true_count}, pred={pred_count}")

print("Failed predictions:", failed_samples)


if failed_outputs:
    print("\n========== FIRST 20 FAILED OUTPUTS ==========")

    for i, item in enumerate(failed_outputs[:20], start=1):
        print(f"\n--- FAILED #{i} ---")

        print("EXPECTED:")
        print(item["expected"][:1000])

        print("\nGENERATED:")
        print(item["generated"][:1500])

        print("=" * 80)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        eval_outputs,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:", OUTPUT_PATH)