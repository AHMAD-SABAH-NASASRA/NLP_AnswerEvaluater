from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"

TRAIN_PATH = "/home/mohammad/.ssh/mohammad/data/processed/train_chat.jsonl"
VAL_PATH = "/home/mohammad/.ssh/mohammad/data/processed/val_chat.jsonl"

OUTPUT_DIR = "/home/mohammad/.ssh/mohammad/outputs_qwen"
SAVE_DIR = "/home/mohammad/.ssh/mohammad/models/finetuned_qwen"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_PATH,
    split="train",
)

val_dataset = load_dataset(
    "json",
    data_files=VAL_PATH,
    split="train",
)


def formatting_func(examples):
    messages_data = examples["messages"]

    if isinstance(messages_data[0], dict):
        text = tokenizer.apply_chat_template(
            messages_data,
            tokenize=False,
            add_generation_prompt=False,
        )
        return [text]

    texts = []
    for messages in messages_data:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return texts


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=30,
        max_steps=800,
        logging_steps=10,
        save_steps=200,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
)

trainer.train()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Fine-tuning completed!")
print("Saved model to:", SAVE_DIR)