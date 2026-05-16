import json
import evaluate

with open("/home/mohammad/.ssh/data/evaluation/eval_outputs_base_mistral.json", "r", encoding="utf-8") as f:
    data = json.load(f)

references = [x["expected"] for x in data]
predictions = [x["generated"] for x in data]

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

print("\n===== RATIONALE METRICS =====")
print("ROUGE-L:", round(rouge_result["rougeL"], 4))
print("BERTScore Precision:", round(sum(bert_result["precision"]) / len(bert_result["precision"]), 4))
print("BERTScore Recall:", round(sum(bert_result["recall"]) / len(bert_result["recall"]), 4))
print("BERTScore F1:", round(sum(bert_result["f1"]) / len(bert_result["f1"]), 4))
