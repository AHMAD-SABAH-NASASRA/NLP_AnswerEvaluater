# Automatic Math Answer Evaluator Using Fine-Tuned LLMs

## Overview

This project investigates the use of instruction-tuned Large Language Models (LLMs) for automatic evaluation of student mathematical answers.

Instead of solving mathematical problems directly, the system evaluates:

* student reasoning,
* partial correctness,
* mathematical logic,
* and final answers

using a rubric-based grading system ranging from **0 to 4**.

The project focuses on:

* rubric-based grading,
* rationale generation,
* LoRA fine-tuning,
* evaluation consistency,
* and explanation quality.

---

# Main Idea

The model receives:

* a math problem,
* a reference solution,
* a student submission,
* and a grading rubric.

The model predicts:

* a score from 0–4,
* and a short rationale explaining the assigned score.

Unlike traditional math benchmarks that focus on solving problems, this project focuses on evaluating another person's solution.

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
Consistency Analysis
        ↓
Human Evaluation
        ↓
Streamlit Application
```

---

# Models Used

## Base Models

* Mistral-7B-Instruct-v0.2
* Qwen2.5-Math-7B-Instruct
* Gemma-3-27B-IT

## Fine-Tuning Framework

* LoRA (Low-Rank Adaptation)
* Unsloth
* 4-bit Quantization
* SFTTrainer
* Hugging Face Transformers
* PyTorch

---

# Dataset

The dataset was generated synthetically using GSM8K-style arithmetic reasoning problems.

Each sample contains:

* task
* reference solution
* student submission
* grading rubric
* score
* rationale

The dataset was designed specifically for rubric-based answer evaluation rather than mathematical problem solving.

---

# Grading Rubric

| Score | Meaning                                                               |
| ----- | --------------------------------------------------------------------- |
| 0     | Irrelevant or completely incorrect                                    |
| 1     | Uses numbers but applies incorrect logic or method                    |
| 2     | Partially correct reasoning with a major conceptual error             |
| 3     | Mostly correct method with a minor arithmetic or final-answer mistake |
| 4     | Fully correct reasoning and final answer                              |

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

The project evaluates both grading performance and explanation quality.

## Grading Metrics

* Accuracy
* Mean Absolute Error (MAE)
* Quadratic Weighted Kappa (QWK)

## Rationale Metrics

* ROUGE-L
* BERTScore

## Consistency Metric

* Score-Rationale Consistency Rate

A rationale is considered consistent when its explanation logically supports the assigned score according to the grading rubric.

---

# Final Benchmark Results

| Model                | Accuracy |    MAE |    QWK | BERTScore F1 | Consistency |
| -------------------- | -------: | -----: | -----: | -----------: | ----------: |
| Base Mistral         |   0.5857 | 0.4534 | 0.8507 |       0.8666 |      0.4762 |
| Fine-tuned Mistral   |   0.8633 | 0.1453 | 0.9551 |       0.9632 |      0.8872 |
| Base Qwen            |   0.0065 | 1.3333 | 0.3891 |       0.8184 |      0.0173 |
| Fine-tuned Qwen      |   0.8918 | 0.1082 | 0.9721 |       0.9494 |      0.8182 |
| Base Gemma-27B       |   0.7587 | 0.2565 | 0.9278 |       0.7602 |      0.7483 |
| Fine-tuned Gemma-27B |   0.9002 | 0.0998 | 0.9744 |       0.9070 |      0.9065 |

---

# Key Findings

* Fine-tuning dramatically improved grading quality across all models.
* Fine-tuned models consistently outperformed their base counterparts.
* Most remaining classification errors occurred between Scores 2 and 3.
* Human evaluators also showed disagreement on borderline cases.
* Strong score prediction does not always imply strong rationale quality.
* Fine-tuning significantly improved alignment between scores and explanations.

The largest improvement was observed in score-rationale consistency. While some base models generated plausible explanations, they frequently assigned grades that did not match those explanations. Fine-tuning greatly improved both grading accuracy and explanation alignment.

---

# Rationale Evaluation

The quality of generated rationales was evaluated separately from score prediction.

ROUGE-L was used to measure lexical overlap between generated and reference rationales.

BERTScore was used to measure semantic similarity between generated and reference rationales.

Since multiple valid explanations may exist for the same grading decision, BERTScore is considered a more reliable measure of rationale quality.

---

# Human Evaluation

A manual evaluation was conducted to inspect rationale quality and score-rationale alignment.

Human review showed that:

* Fine-tuned models produced clearer grading justifications.
* Rationales were generally more rubric-aware after fine-tuning.
* Borderline cases remained challenging even for human evaluators.

---

# Streamlit Application

The project includes an interactive Streamlit application that allows users to:

* select questions,
* enter student submissions,
* choose grading models,
* compare base and fine-tuned models,
* receive automatic scores,
* and view generated rationales.

Run the application:

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

* Python
* PyTorch
* Hugging Face Transformers
* Unsloth
* PEFT / LoRA
* Streamlit
* Scikit-learn
* ROUGE
* BERTScore

---

# Future Work

* Larger and more diverse datasets
* Human-annotated grading data
* More advanced rationale evaluation metrics
* Multi-subject grading systems
* Multi-turn feedback generation
* Classroom deployment and real-world testing

---

# Authors

Mohammad Ismael

An-Najah National University
