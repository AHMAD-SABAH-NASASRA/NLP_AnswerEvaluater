import json
import re
import random
import torch
import streamlit as st

from unsloth import FastLanguageModel

DATA_PATH = "/home/mohammad/.ssh/data/raw/synthetic_gsm8k_grading.jsonl"
MODEL_PATH = "/home/mohammad/.ssh/models/finetuned_chat"

st.set_page_config(
    page_title="Math Answer Evaluator",
    page_icon="🧠",
    layout="wide",
)

RUBRIC = {
    "0": "Irrelevant or completely incorrect answer.",
    "1": "Minimal understanding; uses some problem information incorrectly.",
    "2": "Partially correct reasoning with a major conceptual error.",
    "3": "Mostly correct reasoning with a minor arithmetic mistake.",
    "4": "Fully correct reasoning and final answer.",
}


@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@st.cache_data
def load_questions():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_score(text):
    patterns = [
        r"Score:\s*([0-4])",
        r"score:\s*([0-4])",
        r"re:\s*([0-4])",
        r"Score\s*=\s*([0-4])",
        r"score\s*=\s*([0-4])",
        r"score\s+is\s+([0-4])",
        r"The score is\s+([0-4])",
        r"\bscore\b[^0-9]{0,20}([0-4])",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def build_prompt(tokenizer, task, reference, student_answer):
    rubric_text = "\n".join([f"{k}: {v}" for k, v in RUBRIC.items()])

    user_prompt = f"""Evaluate the student's math answer.

Return ONLY this format:
Score: <0-4>
Rationale: <brief explanation>

Task:
{task}

Reference Solution:
{reference}

Student Submission:
{student_answer}

Rubric:
{rubric_text}
"""

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate_answer(model, tokenizer, task, reference, student_answer):
    prompt = build_prompt(tokenizer, task, reference, student_answer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated = full_text.split("[/INST]")[-1].strip()
    score = extract_score(generated)

    return score, generated


st.title("🧠 Automatic Math Answer Evaluator")
st.write("Fine-tuned LLM evaluator using LoRA + Unsloth.")

questions = load_questions()
model, tokenizer = load_model()

if "question" not in st.session_state:
    st.session_state.question = random.choice(questions)

if st.button("Load Random Question"):
    st.session_state.question = random.choice(questions)
    st.session_state.student_answer = ""
    st.session_state.result = None

sample = st.session_state.question

st.subheader("Question")
st.info(sample["task"])

student_answer = st.text_area(
    "Write your answer:",
    key="student_answer",
    height=160,
)

col1, col2 = st.columns([1, 1])

with col1:
    evaluate_btn = st.button("Evaluate Answer", type="primary")

with col2:
    show_reference = st.checkbox("Show reference solution")

if evaluate_btn:
    if not student_answer.strip():
        st.warning("Please write an answer first.")
    else:
        with st.spinner("Evaluating..."):
            score, generated = evaluate_answer(
                model=model,
                tokenizer=tokenizer,
                task=sample["task"],
                reference=sample["reference"],
                student_answer=student_answer,
            )

        st.session_state.result = {
            "score": score,
            "generated": generated,
        }

if st.session_state.get("result"):
    result = st.session_state.result

    st.subheader("Evaluation Result")

    score = result["score"]

    if score is not None:
        st.metric("Predicted Score", f"{score} / 4")
    else:
        st.error("Could not extract score.")

    st.text_area(
        "Model Output",
        value=result["generated"],
        height=160,
    )

if show_reference:
    st.subheader("Reference Solution")
    st.code(sample["reference"])
