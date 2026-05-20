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
    FINETUNED_MODEL,
    FINETUNED_OUTPUT,
    make_dirs,
)

make_dirs()

INPUT_PATH = str(TEST_CHAT)
MODEL_PATH = str(FINETUNED_MODEL)
OUTPUT_PATH = str(FINETUNED_OUTPUT)


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    test = [json.loads(line) for line in f if line.strip()]

print("Test samples:", len(test))


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
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


print("\n========== FINETUNED QWEN RESULTS ==========")

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


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        eval_outputs,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved outputs to:", OUTPUT_PATH)