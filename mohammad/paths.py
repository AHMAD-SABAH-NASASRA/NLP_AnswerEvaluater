from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs_qwen"

RAW_DATA = RAW_DIR / "synthetic_gsm8k_grading.jsonl"

TRAIN_CHAT = PROCESSED_DIR / "train_chat.jsonl"
VAL_CHAT = PROCESSED_DIR / "val_chat.jsonl"
TEST_CHAT = PROCESSED_DIR / "test_chat.jsonl"

TRAIN_RAW = PROCESSED_DIR / "train_raw.jsonl"
VAL_RAW = PROCESSED_DIR / "val_raw.jsonl"
TEST_RAW = PROCESSED_DIR / "test_raw.jsonl"

FINETUNED_MODEL = ROOT.parent / "outputs_qwen/checkpoint-800"
MODELS = {
    "qwen": ROOT.parent / "outputs_qwen/checkpoint-800",
    "stage2": ROOT.parent / "/home/mohammad/.ssh/mohammad/stage_2",
    "json": ROOT.parent / "models/finetuned_qwen_json",
}
BASE_QWEN_OUTPUT = EVALUATION_DIR / "eval_outputs_base_qwen.json"
FINETUNED_OUTPUT = EVALUATION_DIR / "eval_outputs_qwen_math.json"
RATIONALE_INCONSISTENCIES = EVALUATION_DIR / "rationale_inconsistencies.json"


def make_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)