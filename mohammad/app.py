import json
import re
import random
import sys
from pathlib import Path

import torch
import streamlit as st
from unsloth import FastLanguageModel
ROOT = Path(__file__).resolve().parent
sys.path.append(str(Path(__file__).resolve().parent))

from paths import RAW_DATA, FINETUNED_MODEL


DATA_PATH = str(RAW_DATA)

MODEL_OPTIONS = {
    "Fine-tuned Qwen2.5-Math-7B": str(FINETUNED_MODEL),
    "Base Qwen2.5-Math-7B": "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
    "Gemma_3_27B":"/home/mohammad/.ssh/mohammad/stage_2",
}


st.set_page_config(
    page_title="Math Answer Evaluator",
    page_icon="",
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
def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


@st.cache_data
def load_questions():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_score(text):
    if text is None:
        return None

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
    rubric_text = "\n".join(
        [f"Score {k}: {v}" for k, v in RUBRIC.items()]
    )

    user_prompt = f"""You are a strict math-answer evaluator.

Evaluate the student's answer using the rubric.

Important distinction:
- Score 2: The answer has some relevant reasoning, but contains a major conceptual error, wrong setup, wrong operation, or broken logic.
- Score 3:
The submission is close to correct overall. This may happen in TWO cases:
1. The method is mostly correct but contains a minor arithmetic or final-step mistake.
2. The final answer is correct or nearly correct, but the reasoning/explanation is incomplete, weak, or insufficiently justified.

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
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate_answer(model, tokenizer, task, reference, student_answer):
    prompt = build_prompt(
        tokenizer=tokenizer,
        task=task,
        reference=reference,
        student_answer=student_answer,
    )

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

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    score = extract_score(generated)

    return score, generated


st.title(" Automatic Math Answer Evaluator")
st.write("Math-answer grading using Qwen2.5-Math with optional LoRA fine-tuning.")

selected_model_name = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_OPTIONS.keys()),
)

selected_model_path = MODEL_OPTIONS[selected_model_name]

st.sidebar.write("Selected model:")
st.sidebar.code(selected_model_name)

questions = load_questions()

model, tokenizer = load_model(selected_model_path)


if "question" not in st.session_state:
    st.session_state.question = random.choice(questions)

if "result" not in st.session_state:
    st.session_state.result = None

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
        with st.spinner(f"Evaluating with {selected_model_name}..."):
            score, generated = evaluate_answer(
                model=model,
                tokenizer=tokenizer,
                task=sample["task"],
                reference=sample["reference"],
                student_answer=student_answer,
            )

        st.session_state.result = {
            "model": selected_model_name,
            "score": score,
            "generated": generated,
        }


if st.session_state.result:
    result = st.session_state.result

    st.subheader("Evaluation Result")

    st.write("Model used:")
    st.code(result["model"])

    score = result["score"]

    if score is not None:
        st.metric("Predicted Score", f"{score} / 4")
    else:
        st.error("Could not extract score.")

    st.text_area(
        "Model Output",
        value=result["generated"],
        height=180,
    )


if show_reference:
    st.subheader("Reference Solution")
    st.code(sample["reference"])