from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"

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

dataset = load_dataset(
    "json",
    data_files="/home/mohammad/.ssh/data/processed/train_chat.jsonl",
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
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="outputs_qwen",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=30,
        max_steps=800,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
    ),
)

trainer.train()

model.save_pretrained("models/finetuned_qwen")
tokenizer.save_pretrained("models/finetuned_qwen")

print("Fine-tuning completed!")
