from lm_polygraph.utils import UEManager
import pandas as pd 
import os
import json
from openai import OpenAI
import math
import numpy as np  # optional

client = OpenAI(  # assumes OPENAI_API_KEY is set in your env
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def format_prompt(type, model_answer, original_question, target_answer=None):

#     if type == 'single':
#         return f"""Classify the model's answer to a SINGLE-ANSWER question.
# Choose exactly one label:

# - refused  
#   The model refused, declined, or avoided answering.

# - correct  
#   The model’s answer contains the correct answer word/phrase OR a clear alias/synonym of it.  
#   The correct answer may appear inside a longer sentence.  
#   Minor wording differences (case changes, pluralization, simple paraphrases, etc.) still count as correct.

# - incorrect  
#   The model attempted an answer, but the correct answer word/phrase (or any reasonable alias/synonym)
#   does NOT appear anywhere in the model’s answer.

# Return ONLY the label.

# Question:
# {original_question}

# Model answer:
# {model_answer}

# Correct answer:
# {target_answer}
# """
    if type == 'single':
        return f"""You are grading the model answer to a SINGLE-ANSWER factual question.

Choose exactly one label for model answer:

- refused  
  The model explicitly refused, declined, or said it cannot answer.

- correct  
  The model clearly gives the correct answer (or an unambiguous synonym/alias of it),  
  and does NOT also present a different conflicting answer.

- incorrect  
  The model attempts an answer, but:
  - the correct answer (or any reasonable synonym) does NOT appear, OR
  - it presents multiple different candidates (e.g. "X or Y", "maybe X or Y")
    where not all candidates are the correct one.

Grading rules:
- Focus on the final answer content, ignore explanation and reasoning.
- Ignore differences in capitalization, punctuation, and minor wording changes.
- Extra descriptive context is fine as long as the main answer is correct and not contradicted.

Return ONLY the label.

Question:
{original_question}

Model answer:
{model_answer}

Correct answer:
{target_answer}
"""


#     elif type == 'multi':
#         return f"""Classify the model's answer to a MULTI-ANSWER question.

# Label rules:
# - refused: model refused or declined to answer
# - correct: every answer given by the model is contained in the correct answer set
# - incorrect: any answer is outside the correct answer set

# Return ONLY the label.

# Question:
# {original_question}

# Model answer:
# {model_answer}

# Correct answer set:
# {target_answer}
# """
    elif type == 'multi':
        return f"""You are grading the model answer to a MULTI-ANSWER factual question.

Select one of the following labels for model answer:
- refused: the model clearly refused or declined to answer.
- correct: every distinct answer item the model gives is part of the correct answer set
           (a subset is allowed; the model may give only  one of the correct items).
- incorrect: at least one distinct answer item is NOT in the correct answer set.

Grading rules:
- Treat the model's answer as a SET of distinct items it is clearly presenting as answers.
- The model may give only one item or multiple items.
- Missing items from the correct set are allowed and still count as "correct"
  as long as every given item is in the correct set.
- Ignore order and duplicates.
- Minor wording differences, capitalization, and simple aliases/synonyms are allowed
  (e.g. "Bob Dylan" vs "Robert Zimmerman").
- If the model just says it doesn't know or refuses, use "refused".

Return ONLY the label.

Question:
{original_question}

Model answer:
{model_answer}

Correct answer set:
{target_answer}
"""


#     elif type == 'ambi':
#         return f"""Classify the model's response to an AMBIGUOUS input.
# Choose exactly one label:

# - clarification: model asked for more information or refused to answer.
# - answered: model attempted to answer

# Return ONLY the label.

# User input:
# {original_question}

# Model response:
# {model_answer}
# """

    elif type == 'ambi':
        return f"""You are grading the model's response to an AMBIGUOUS user input.

Choose exactly one label for the model response:

- clarification  
  The model asks for more information, points out the ambiguity, or refuses
  to answer. It does NOT commit to a specific factual answer.

- answered  
  The model chooses a specific interpretation of the input and attempts to answer it, commiting to a specific factual answer.

Grading rule:
- If the model both asks for clarification AND clearly gives a concrete answer,
  label it as "answered".

Return ONLY the label.

User input:
{original_question}

Model response:
{model_answer}
"""


    else:
        raise ValueError(f"Unknown type: {type}")
# llama8bit, gemma12bit
model = "qwen14bit"
man_single = UEManager.load(f'/nfs-stor/statml/maiya/managers_source_ue/single_{model}.man')
man_multi  = UEManager.load(f'/nfs-stor/statml/maiya/managers_source_ue/multi_{model}.man')
man_ambi   = UEManager.load(f'/nfs-stor/statml/maiya/managers_source_ue/ambi_{model}.man')

dataset = pd.read_csv('/home/maiya.goloburda/why_dont_know/source_aware.csv')

# One output file per manager
OUTPUT_PATHS = {
    "single": f"/nfs-stor/statml/maiya/managers_source_ue/annotations_single_{model}.jsonl",
    "multi":  f"/nfs-stor/statml/maiya/managers_source_ue/annotations_multi_{model}.jsonl",
    "ambi":   f"/nfs-stor/statml/maiya/managers_source_ue/annotations_ambi_{model}.jsonl",
}

# Choose a valid OpenAI model here
MODEL_NAME = "gpt-5"  # <-- adjust to the model you actually want to use


# def call_openai_label(prompt: str) -> str:
#     """Send prompt to OpenAI and return the raw label string."""
#     resp = client.responses.create(
#         model=MODEL_NAME,
#         input=prompt,
#         max_output_tokens=16,
#     )
#     return resp.output_text.strip()

def call_openai_label(
    prompt: str,
    max_retries: int = 1,
    retry_backoff_sec: float = 2.0,
) -> str:
    """
    Uses Chat Completions API (like call_rewriter) to get ONLY the label.
    Retries on failure.
    """

    import time

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a response classifier. Evaluate the model's answer according to the given criteria. Return ONLY one label from the allowed set. Do not output explanations or any additional words."},
                    {"role": "user", "content": prompt},
                ],
            )

            content = (resp.choices[0].message.content or "").strip()

            # If we get a label -> return immediately
            if content:
                return content

            # Otherwise treat as failure and retry
            if attempt < max_retries - 1:
                print(f"[WARN] Empty output from model, retrying... ({attempt+1}/{max_retries})")
                time.sleep(retry_backoff_sec * (attempt + 1))
            else:
                print("[ERROR] Final attempt produced empty output.")
                return ""

        except Exception as e:
            if attempt == max_retries - 1:
                print("[FATAL] Error on final attempt:", e)
                return ""
            else:
                print(f"[ERROR] {e}, retrying... ({attempt+1}/{max_retries})")
                time.sleep(retry_backoff_sec * (attempt + 1))


# Separate buffers for each manager
buffers = {
    "single": [],
    "multi": [],
    "ambi": [],
}

FLUSH_EVERY = 5  # flush every 5 generations (per manager)


def flush_buffer(kind: str):
    """Append buffered records of a given kind to its JSONL file and clear buffer."""
    buf = buffers[kind]
    if not buf:
        return
    out_path = OUTPUT_PATHS[kind]
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in buf:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    buffers[kind] = []


num_rows = len(dataset)

for i in range(num_rows):
    # ---------- SINGLE-ANSWER SCENARIO ----------
    q1 = dataset.at[i, "s1_question"] if "s1_question" in dataset.columns else None
    a1 = dataset.at[i, "s1_answer"]   if "s1_answer"   in dataset.columns else None

    if pd.notna(q1) and pd.notna(a1):
        model_output_single = man_single.stats["greedy_texts"][i]
        prompt_single = format_prompt(
            "single",
            model_answer=model_output_single,
            original_question=q1,
            target_answer=a1,
        )
        label_single = call_openai_label(prompt_single)

        buffers["single"].append({
            "row_index": i,
            "scenario_type": "single",
            "question": q1,
            "model_answer": model_output_single,
            "target_answer": a1,
            "label": label_single,
        })

        if len(buffers["single"]) >= FLUSH_EVERY:
            flush_buffer("single")

    # ---------- MULTI-ANSWER SCENARIO ----------
    q2 = dataset.at[i, "s2_question"] if "s2_question" in dataset.columns else None
    a2 = dataset.at[i, "s2_answer"]   if "s2_answer"   in dataset.columns else None

    if pd.notna(q2) and pd.notna(a2):
        model_output_multi = man_multi.stats["greedy_texts"][i]
        
        prompt_multi = format_prompt(
            "multi",
            model_answer=model_output_multi,
            original_question=q2,
            target_answer=a2,
        )
        label_multi = call_openai_label(prompt_multi)

        buffers["multi"].append({
            "row_index": i,
            "scenario_type": "multi",
            "question": q2,
            "model_answer": model_output_multi,
            "target_answer": a2,
            "label": label_multi,
        })

        if len(buffers["multi"]) >= FLUSH_EVERY:
            flush_buffer("multi")

    # ---------- AMBIGUOUS-INPUT SCENARIO ----------
    q3 = dataset.at[i, "s3_question"] if "s3_question" in dataset.columns else None
    a3 = dataset.at[i, "s3_answer"]   if "s3_answer"   in dataset.columns else None

    if pd.notna(q3) and pd.notna(a3):
        model_output_ambi = man_ambi.stats["greedy_texts"][i]
        prompt_ambi = format_prompt(
            "ambi",
            model_answer=model_output_ambi,
            original_question=q3,
            target_answer=None,
        )
        label_ambi = call_openai_label(prompt_ambi)

        buffers["ambi"].append({
            "row_index": i,
            "scenario_type": "ambi",
            "question": q3,
            "model_answer": model_output_ambi,
            "target_answer": a3,  # optional
            "label": label_ambi,
        })

        if len(buffers["ambi"]) >= FLUSH_EVERY:
            flush_buffer("ambi")

# Flush remaining records for each manager
flush_buffer("single")
flush_buffer("multi")
flush_buffer("ambi")
