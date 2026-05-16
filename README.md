# NLP Answer Evaluator

An automatic evaluator for mathematical student answers using instruction-tuned Large Language Models (LLMs).

## Overview

This project focuses on evaluating mathematical answers rather than solving math problems. The system receives:

* A math problem
* A reference solution
* A student submission
* A grading rubric

The model then predicts:

* A score from 0 to 4
* A textual rationale explaining the evaluation

The project is designed for rubric-based assessment of reasoning quality and mathematical correctness.

---

# Problem Statement

Traditional automatic grading systems usually focus only on the final answer. However, mathematical reasoning quality is also important.

This project attempts to evaluate:

* Correctness of reasoning
* Logical consistency
* Partial understanding
* Arithmetic mistakes
* Alignment with grading rubrics

The system is trained to behave like a grading assistant instead of a mathematical solver.

---

# Rubric

The grading rubric used in the project:

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 0     | Irrelevant answer or no useful mathematical reasoning                    |
| 1     | Uses numbers or operations but reasoning is fundamentally incorrect      |
| 2     | Partially correct reasoning with a major conceptual or calculation error |
| 3     | Mostly correct method with a minor arithmetic or final-answer mistake    |
| 4     | Fully correct reasoning and final answer                                 |

---

# Project Structure

```text
.
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── evaluation/
├── training/
├── models/
└── outputs/
```

---

# Dataset

The project uses synthetic rubric-based grading data.

The dataset contains samples structured as conversational chat data:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "question + reference solution + student answer + rubric"
    },
    {
      "role": "assistant",
      "content": "Score: X\nRationale: ..."
    }
  ]
}
```

Processed datasets include:

* `train_chat.jsonl`
* `test_chat.jsonl`
* `train_instruct.jsonl`
* `test_instruct.jsonl`

---

# Models

The project experiments with instruction-tuned LLMs including:

* Qwen2.5-Math-7B-Instruct
* Mistral-based models

Training uses:

* LoRA
* QLoRA
* 4-bit quantization
* Unsloth optimization

---

# Training Approach

The project uses:

* Supervised Fine-Tuning (SFT)
* Chat-template based training
* PEFT / LoRA adapters

Main training framework:

* Transformers
* TRL
* Unsloth
* Hugging Face Datasets

---

# Example Training Configuration

```python
MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"
```

Key settings:

* 4-bit quantization
* LoRA rank = 16
* Learning rate = 1e-4
* Sequence length = 2048
* bf16 training

---

# Evaluation

Evaluation scripts are located in:

```text
evaluation/
```

Implemented evaluation utilities include:

* Base model evaluation
* Fine-tuned model evaluation
* Rationale inspection
* Consistency checking
* Error extraction
* Advanced metrics analysis

Metrics used include:

* Accuracy
* Mean Absolute Error (MAE)

---

# Example Output

```text
Score: 3
Rationale: The student followed the correct solution strategy but made a minor arithmetic mistake in the final computation.
```

---

# Repository Contents

Important directories:

| Directory          | Purpose                                 |
| ------------------ | --------------------------------------- |
| `training/`        | Fine-tuning scripts                     |
| `evaluation/`      | Evaluation and analysis scripts         |
| `data/processed/`  | Processed training and testing datasets |
| `data/evaluation/` | Evaluation outputs and annotations      |
| `models/`          | Fine-tuned LoRA adapter models          |

---

# Notes

* The project evaluates answers instead of generating complete math solutions.
* LoRA adapters are used instead of full model fine-tuning.
* Outputs are designed to follow a fixed rubric structure.
* Checkpoints and temporary training outputs are excluded from the repository.

---

# Requirements

Main dependencies:

```text
transformers
trl
datasets
unsloth
torch
accelerate
bitsandbytes
scikit-learn
```

---

# Running Training

Example:

```bash
python training/train_chat.py
```

---

# Running Evaluation

Example:

```bash
python evaluation/evaluate_finetuned.py
```

---

# Purpose of the Project

The project is intended as a research-oriented experiment in:

* Automatic grading
* Mathematical reasoning evaluation
* Rubric-conditioned LLM behavior
* Educational NLP systems
* Instruction-tuned evaluation models
