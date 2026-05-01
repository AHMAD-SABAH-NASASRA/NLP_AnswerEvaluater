from unsloth import FastLanguageModel
import torch
import os
import json
from datetime import datetime

path = r"C:\Users\Sanasys\Downloads\dataEX\dataEX\models\finetuned"
results_file = "evaluation_results.jsonl"

print("Exists:", os.path.exists(path))
print("Content:", os.listdir(path))

model, tokenizer = FastLanguageModel.from_pretrained(
    path,
    load_in_4bit=True
)

FastLanguageModel.for_inference(model)

def evaluate(task, submission):
    prompt = f"""
### Instruction:
You are a strict math teacher. Grade the student's answer using rubric 0-4.

### Task:
{task}

### Student Answer:
{submission}

### Output format:
Score: (0-4)
Reasoning: explain briefly
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    record = {
        "time": datetime.now().isoformat(),
        "task": task,
        "student_answer": submission,
        "model_result": result
    }

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return result


task = "Janet has 16 eggs. She eats 3 and uses 4. How many are left?"
student_answer = "She has 16 - 3 = 13 then 13 - 4 = 10 so answer is 10"

print(evaluate(task, student_answer))
print("Saved to:", results_file)