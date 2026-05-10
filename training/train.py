from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)


dataset = load_dataset(
    "json",
    data_files="train_instruct.jsonl",
    split="train",
)


def formatting_func(example):
    text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""

    return [text]


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,

    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=20,
        max_steps=300,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
    ),
)


trainer.train()

model.save_pretrained("models/finetuned")
tokenizer.save_pretrained("models/finetuned")

print("Fine-tuning completed!")