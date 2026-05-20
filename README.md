# Automatic Math Answer Evaluator Using Fine-Tuned LLMs

## Overview

This project investigates the use of instruction-tuned Large Language Models (LLMs) for automatic evaluation of student mathematical answers.

Instead of solving mathematical problems directly, the system evaluates:
- student reasoning,
- partial correctness,
- mathematical logic,
- and final answers

using a rubric-based grading system from **0 to 4**.

The project focuses on:
- rubric-based grading,
- rationale generation,
- LoRA fine-tuning,
- and evaluation consistency.

---

# Main Idea

The model receives:
- a math problem,
- a reference solution,
- a student submission,
- and a grading rubric.

The model then predicts:
- a score from 0–4,
- and a short rationale explaining the grade.

---

# Pipeline

```text
GSM8K Problems
        ↓
Synthetic Student Answers
        ↓
Preprocessing & Validation
        ↓
Train / Validation / Test Split
        ↓
Chat Formatting
        ↓
Base Model Evaluation
        ↓
LoRA Fine-Tuning
        ↓
Evaluation Metrics
        ↓
Error Analysis
        ↓
Human Evaluation
        ↓
Streamlit Application
```

---

# Models Used

## Base Models
- Mistral-7B-Instruct
- Qwen2.5-Math
- Gemma-3-27B-IT

## Fine-Tuning
- LoRA
- Unsloth
- 4-bit Quantization
- SFTTrainer

---

# Dataset

The dataset was built synthetically using GSM8K-style arithmetic problems.

Each sample contains:
- task
- reference solution
- student submission
- rubric
- score
- rationale

---

# Rubric

| Score | Meaning |
|---|---|
| 0 | Irrelevant or completely incorrect |
| 1 | Wrong mathematical method |
| 2 | Partially correct with major conceptual error |
| 3 | Mostly correct with minor mistake |
| 4 | Fully correct reasoning and final answer |

---

# Project Structure

```text
mohammad/
│
├── app.py
├── paths.py
│
├── data/
│   ├── processed/
│   └── evaluation/
│
├── evaluation/
│   ├── evaluate_base_qwen.py
│   ├── evaluate_finetuned.py
│   ├── compare_base_finetuned.py
│   ├── rationale_metrics.py
│   └── check_consistency.py
│
├── stage_2/
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── demo.mp4
└── final_report.pdf
```

---

# Evaluation Metrics

The project uses:
- Accuracy
- MAE
- QWK
- ROUGE-L
- BERTScore
- Consistency Rate

---

# Main Results

| Model | Accuracy | QWK |
|---|---:|---:|
| Base Mistral | 58.57% | 0.8507 |
| Fine-Tuned Mistral | 86.33% | 0.9551 |
| Fine-Tuned Qwen | 89.18% | 0.9721 |
| Fine-Tuned Gemma-27B | 90.02% | 0.9744 |

---

# Key Findings

- Fine-tuning dramatically improved grading quality.
- Most remaining errors occurred between scores 2 and 3.
- Human evaluators also disagreed on borderline cases.
- Strong score prediction does not always imply strong rationale quality.

---

# Streamlit Application

The project includes an interactive Streamlit demo that allows users to:
- choose questions,
- submit student answers,
- select models,
- and receive automatic grading + rationales.

Run:

```bash
streamlit run app.py
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Technologies

- Python
- PyTorch
- Transformers
- Unsloth
- Streamlit
- Scikit-learn

---

# Future Work

- Larger datasets
- Human-annotated grading data
- Better rationale evaluation
- Multi-subject grading
- Classroom deployment

---

# Authors

Mohammad Ismael
