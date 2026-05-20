import json
import re
import sys
from pathlib import Path

import evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paths import FINETUNED_OUTPUT

INPUT_PATH = str(FINETUNED_OUTPUT)


def extract_rationale(text):
    if text is None:
        return ""

    match = re.search(
        r"Rationale:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text.strip()


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


references = [
    extract_rationale(x["expected"])
    for x in data
]

predictions = [
    extract_rationale(x["generated"])
    for x in data
]


rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


rouge_result = rouge.compute(
    predictions=predictions,
    references=references,
)

bert_result = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
)


bert_precision = sum(bert_result["precision"]) / len(bert_result["precision"])
bert_recall = sum(bert_result["recall"]) / len(bert_result["recall"])
bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])


print("\n===== RATIONALE-ONLY METRICS =====")
print("ROUGE-L:", round(rouge_result["rougeL"], 4))
print("BERTScore Precision:", round(bert_precision, 4))
print("BERTScore Recall:", round(bert_recall, 4))
print("BERTScore F1:", round(bert_f1, 4))