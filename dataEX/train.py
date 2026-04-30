from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ---------------------------
# إعدادات مناسبة لـ 4GB GPU
# ---------------------------
max_seq_length = 768

# ---------------------------
# تحميل مودل أصغر من Mistral 7B
# ---------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# ---------------------------
# LoRA setup
# ---------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=4,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# ---------------------------
# تحميل البيانات
# ---------------------------
dataset = load_dataset(
    "json",
    data_files="data/train_instruct.jsonl",
    split="train"
)

# ---------------------------
# دمج instruction + input + output
# ---------------------------
def format_prompt(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
{example['output']}"""

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    return {
        "text": tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    }

dataset = dataset.map(format_prompt)

# ---------------------------
# التدريب
# ---------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=5,
        output_dir="outputs",
        save_steps=15,
        save_total_limit=1,
        optim="adamw_8bit",
    ),
)

trainer.train()

# ---------------------------
# حفظ النموذج
# ---------------------------
model.save_pretrained("models/finetuned")
tokenizer.save_pretrained("models/finetuned")

print("Training finished and model saved.")