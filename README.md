# Automatic Math Answer Evaluator

This research prototype assigns an ordinal score from 0 to 4 to a student's mathematical rationale and generates an explanation. The repository contains a 2,300-example synthetic GSM8K-derived dataset, preprocessing and evaluation scripts, LoRA adapter configuration files, recorded evaluation outputs, a report, and a Streamlit demo.

## Workflow

`JSONL data → deterministic train/validation/test split → chat formatting → 4-bit LoRA fine-tuning → generated score/rationale → ordinal and rationale evaluation`

The rubric ranges from 0 (irrelevant or fully incorrect) to 4 (correct reasoning and answer). The processed split contains 1,609 training, 229 validation and 462 test examples.

## Fine-tuning configuration

The committed Qwen adapter configuration identifies `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit`, 4-bit loading, LoRA rank/alpha 16/16, and adapters on attention and MLP projection layers. `train_chat.py` uses TRL `SFTTrainer`, sequence length 2048, effective batch size 16, learning rate `1e-4`, 800 steps, bf16 and 8-bit AdamW. The `stage_2` adapter metadata targets Gemma-3-27B-IT with rank 32, alpha 64 and attention projections.

## Verified recorded results

Metrics below were recomputed from the committed JSON evaluation outputs. They verify the files, not a fresh model run.

| Model/output | n | Accuracy | MAE | Quadratic weighted kappa |
|---|---:|---:|---:|---:|
| Base Mistral | 461 | 0.5857 | 0.4534 | 0.8507 |
| Fine-tuned Mistral | 461 | 0.8633 | 0.1453 | 0.9551 |
| Fine-tuned Qwen2.5-Math-7B | 462 | **0.8918** | **0.1082** | **0.9721** |

The base-Qwen output has MAE 1.3333 and QWK 0.3891, but its stored accuracy field is inconsistent with the predictions and should not be quoted. Gemma results appear only in the PDF report and cannot be independently recomputed from committed prediction files.

## Run the demo

Install a CUDA-compatible PyTorch build first; the 7B/27B 4-bit workflow has no supported CPU path.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NLP_FINETUNED_MODEL=/absolute/path/to/checkpoint-800
streamlit run mohammad/app.py
```

## Structure

- `mohammad/training/` — preprocessing and Qwen training
- `mohammad/evaluation/` — grading, comparison and rationale scripts
- `mohammad/data/` — raw/processed data and recorded outputs
- `mohammad/finetuned_qwen/`, `mohammad/stage_2/` — adapter metadata/tokenizers
- `mohammad/app.py` — Streamlit demo
- `mohammad/final_report.pdf` — project report

## Known limitations

- Adapter weight files are not committed; a fresh clone cannot load the fine-tuned model.
- No Mistral training script is included even though Mistral outputs are reported.
- Gemma metrics are report-only and lack prediction artifacts.
- Evaluation is performed on synthetic data; generalization to real student answers is untested.
- “Consistency” in the evaluation code is a keyword-based heuristic, not formal mathematical verification.
- There is no automated test suite or CI, and GPU/version sensitivity limits reproducibility.
