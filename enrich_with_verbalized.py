from pathlib import Path
import json, re, time, math
import pandas as pd
from openai import OpenAI
from lm_polygraph.utils import UEManager

# ---------------- CONFIG ----------------
MODEL = "gpt-4o-mini"
RATE_DELAY = 0.25         # time.sleep between calls
MAX_RETRIES = 5           # network/backoff retries
CHECKPOINT_DIR = Path("prompt_runs_ckpt")  # per-stage checkpoints
FINAL_DIR = Path("prompt_runs")            # final outputs
FINAL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT: Set environment variable OPENAI_API_KEY
client = OpenAI(api_key='')  # reads from env

# --------------- LOAD DATA ---------------
man = UEManager.load('coqa_gpt40_mini.man')
inputs = list(man.stats['input_texts'])
outputs = list(man.stats['greedy_texts'])
N = len(inputs)

# ------------- PROMPTS & TAXONOMY ----------
CONFIDENCE_SYSTEM = "You are a calibrated assessor. Return ONLY a probability between 0.0 and 1.0."
CONFIDENCE_USER_TMPL = """Question:
{question}

Your guess:
{guess}

Provide the probability that your guess is correct.
Give ONLY the probability, no other words or explanation.
For example:
Probability: <0.0..1.0>"""

TAXONOMY = [
    {
        "name": "Prompt Ambiguity",
        "desc": "Uncertainty from how the question is asked (lexical, referential, underspecified phrasing).",
        "children": [
            {"name": "Linguistic ambiguity", "desc": "The wording itself has multiple interpretations."},
            {"name": "Referential ambiguity", "desc": "References in the input are unclear or could refer to multiple entities."},
            {"name": "Underspecification", "desc": "The input does not provide enough detail to determine a unique answer."}
        ]
    },
    {
        "name": "Model / Knowledge Uncertainty",
        "desc": "Uncertainty from what the model knows (knowledge gaps, outdated information, or domain mismatch).",
        "children": [
            {"name": "Information missing", "desc": "The required knowledge is absent from the model."},
            {"name": "Information outdated", "desc": "The model relies on stale knowledge or facts that have changed."},
            {"name": "Domain shift", "desc": "The input is outside the model’s training distribution or knowledge scope."}
        ]
    },
    {
        "name": "Outcome Variability",
        "desc": "Uncertainty because multiple answers may be acceptable or definitions are ill-posed.",
        "children": [
            {"name": "Multiple correct answers", "desc": "More than one valid answer exists for the same input."},
            {"name": "Subjectivity", "desc": "The answer depends on opinion, preference, or perspective."},
            {"name": "Conflicting answers", "desc": "Different valid sources or viewpoints contradict one another."}
        ]
    }
]

def _taxonomy_block():
    # Pretty-prints the taxonomy with subcategories for the prompt
    lines = []
    for idx, t in enumerate(TAXONOMY, start=1):
        lines.append(f"{idx}. {t['name']}\n   - {t['desc']}")
        children = t.get("children") or []
        for c in children:
            lines.append(f"   * {c['name']}")
    return "\n".join(lines)

SRC_TAX_SYSTEM = (
 "You are a calibrated assessor. Return ONLY a probability between 0.0 and 1.0."
)

# NOTE: Double braces {{...}} inside .format() template to avoid KeyError
SRC_TAX_USER_TMPL = """Question:
{question}
Answer:
{answer}

Taxonomy of uncertainty sources (for your internal reflection):
{taxonomy}

Instructions:
1) Privately consider which uncertainty sources apply.
2) Based on this reflection, estimate the probability the answer is correct.
3) Do NOT reveal your analysis. Output ONLY the probability.

Format:
Probability: <0.0..1.0>"""

# -------------- UTILITIES -----------------
def retrying_call(fn, *args, **kwargs):
    """Run a function with exponential backoff retries."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep_s = (2 ** attempt) * 0.5
            time.sleep(sleep_s)

def parse_probability(text: str) -> float:
    # Extracts 0, 1, 0.x, .x, x.y
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text.strip())
    return float(m.group(1)) if m else float("nan")

def ensure_indexed(results_dict):
    """Make sure each partial outputs dict has 'index' key."""
    for r in results_dict:
        assert 'index' in r, "Missing 'index' in result item."
    return results_dict

def load_checkpoint(path: Path) -> dict:
    """Return {index:int -> payload:dict} for quick resume."""
    if not path.exists():
        return {}
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                data[int(rec["index"])] = rec
    return data

def append_checkpoint(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def remaining_indices(done_map: dict, total: int) -> list:
    return [i for i in range(total) if i not in done_map]

def save_aggregate_csv(stage_payloads: dict, out_csv: Path):
    """Merge all stage payloads (by index) into a single CSV."""
    df = pd.DataFrame({
        "index": range(N),
        "question": inputs,
        "greedy_guess": outputs,
    })
    df.set_index("index", inplace=True)

    for _, stage_map in stage_payloads.items():
        if not stage_map:
            continue
        stage_df = pd.DataFrame.from_dict(stage_map, orient="index")
        df = df.join(stage_df, how="left")

    df.reset_index(inplace=True)
    df.to_csv(out_csv, index=False)
    return df

# --------- JSON PARSE HELPERS (robust) ---------
def _extract_first_json_obj(text: str):
    """Return the first valid JSON object from possibly-fenced / chatty text."""
    if not text:
        return {}
    s = text.strip()
    # Strip ```json fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: grab the first {...} span
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return {}
    return {}

def _clean_keys(obj):
    """Strip whitespace/quotes from dict keys recursively."""
    if isinstance(obj, dict):
        return {str(k).strip().strip('"'): _clean_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_keys(x) for x in obj]
    return obj

def _norm_str(x):
    return str(x).strip().strip('"').strip()

# -------------- STAGE RUNNERS --------------

def run_stage_confidence():
    """
    Stage 1: confidence_only for greedy guess.
    Writes checkpoint lines to ckpt_confidence.jsonl
    """
    ckpt_path = CHECKPOINT_DIR / "ckpt_confidence.jsonl"
    done = load_checkpoint(ckpt_path)
    todo = remaining_indices(done, N)
    print(f"[Stage: confidence_only] {len(done)}/{N} done; {len(todo)} remaining.")

    for i in todo:
        q = inputs[i]
        g = outputs[i]

        def call():
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": CONFIDENCE_SYSTEM},
                    {"role": "user", "content": CONFIDENCE_USER_TMPL.format(question=q, guess=g)},
                ],
                temperature=0
            )

        resp = retrying_call(call)
        text = resp.choices[0].message.content.strip()
        conf = parse_probability(text)

        rec = {"index": i, "confidence_only": conf}
        append_checkpoint(ckpt_path, rec)
        time.sleep(RATE_DELAY)

    return load_checkpoint(ckpt_path)


def run_stage_src_tax(start_idx=0):
    """
    Stage 4: source_reflection_taxonomy (strict JSON)
    Writes checkpoint lines to ckpt_src_tax.jsonl
    """
    ckpt_path = CHECKPOINT_DIR / "ckpt_src_tax.jsonl"
    done = load_checkpoint(ckpt_path)
    todo_all = remaining_indices(done, N)
    todo = [i for i in todo_all if i >= start_idx]
    print(f"[Stage: source_reflection_taxonomy] {len(done)}/{N} done; {len(todo)} remaining (starting at {start_idx}).")

    # Build validation sets from TAXONOMY:
    valid_cat_names = {t["name"] for t in TAXONOMY}
    valid_subcat_names = set()
    for t in TAXONOMY:
        for c in (t.get("children") or []):
            nm = c.get("name")
            if nm:
                valid_subcat_names.add(nm)

    def _norm_list_field(val):
        if isinstance(val, dict):
            val = list(val.values())
        elif isinstance(val, str):
            val = [p.strip() for p in val.split(",") if p.strip()]
        elif not isinstance(val, list):
            val = [val]
        return [_norm_str(x) for x in val if str(x).strip()]

    for i in todo:
        q = inputs[i]
        a = outputs[i]

        def call():
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SRC_TAX_SYSTEM},
                    {"role": "user", "content": SRC_TAX_USER_TMPL.format(
                        question=q,
                        answer=a,
                        taxonomy=_taxonomy_block()
                    )},
                ],
                temperature=0.0
            )

        resp = retrying_call(call)
        text = resp.choices[0].message.content

        # Strict parse with robust fallback & key cleaning
        try:
            data = json.loads(text)
        except Exception:
            data = _extract_first_json_obj(text)

        data = _clean_keys(data) if isinstance(data, (dict, list)) else {}

        # Categories
        raw_cats = data.get("uncertainty_categories") or data.get("categories") or []
        cats = [c for c in _norm_list_field(raw_cats) if c in valid_cat_names]
        cats = list(dict.fromkeys(cats))  # de-dup, preserve order

        # Subcategories
        raw_subcats = data.get("uncertainty_subcategories") or data.get("subcategories") or []
        subcats = [s for s in _norm_list_field(raw_subcats) if s in valid_subcat_names]
        subcats = list(dict.fromkeys(subcats))

        # Notes
        notes = data.get("uncertainty_notes", [])
        if isinstance(notes, str):
            notes = [notes]
        elif not isinstance(notes, list):
            notes = [str(notes)]

        # Confidence — clamp to [0,1] if numeric, else NaN
        conf = data.get("confidence")
        try:
            conf = float(conf)
            if math.isnan(conf):
                raise ValueError
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = float("nan")

        guess = data.get("guess", "")

        rec = {
            "index": i,
            "src_tax_categories": " | ".join(cats),
            "src_tax_subcategories": " | ".join(subcats),
            "src_tax_notes": notes,
            "src_tax_confidence": conf,
            "src_tax_guess": guess,
        }
        append_checkpoint(ckpt_path, rec)
        time.sleep(RATE_DELAY)

    return load_checkpoint(ckpt_path)


REFLECT_SYSTEM = (
    "You are a calibrated assessor. Think through the uncertainty sources privately "
    "and return ONLY a probability between 0.0 and 1.0."
)

REFLECT_USER_TMPL = """Question:
{question}

Answer:
{answer}

Taxonomy of uncertainty sources (for your internal reflection):
{taxonomy}

Instructions:
1) Privately assess which uncertainty sources (categories/subcategories) apply and how they affect correctness.
2) Based on this reflection, estimate the probability that the answer is correct.
3) Do NOT reveal your analysis or categories. Output ONLY the final probability.

Format:
Probability: <0.0..1.0>"""

def run_stage_reflect_prob():
    """
    Treatment: reflection using taxonomy, output ONLY probability.
    Writes checkpoint lines to ckpt_reflect_prob.jsonl
    """
    ckpt_path = CHECKPOINT_DIR / "ckpt_reflect_prob.jsonl"
    done = load_checkpoint(ckpt_path)
    todo = remaining_indices(done, N)
    print(f"[Stage: reflect_prob] {len(done)}/{N} done; {len(todo)} remaining.")

    for i in todo:
        q = inputs[i]
        a = outputs[i]

        def call():
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": REFLECT_SYSTEM},
                    {"role": "user", "content": REFLECT_USER_TMPL.format(
                        question=q, answer=a, taxonomy=_taxonomy_block()
                    )},
                ],
                temperature=0
            )

        resp = retrying_call(call)
        text = resp.choices[0].message.content.strip()
        conf = parse_probability(text)

        rec = {"index": i, "reflect_prob": conf, "reflect_prob_raw": text}
        append_checkpoint(ckpt_path, rec)
        time.sleep(RATE_DELAY)

    return load_checkpoint(ckpt_path)

# -------------- PIPELINE DRIVER --------------
def main():
    # Run both stages, aggregate
    conf_map = {k: v for k, v in run_stage_confidence().items()}
    refl_map = {k: v for k, v in run_stage_reflect_prob().items()}     

    final_df = save_aggregate_csv(
        {"confidence": conf_map, "tax": refl_map},
        FINAL_DIR / "aggregate_final.csv"
    )
    print(f"Saved: {FINAL_DIR / 'aggregate_final.csv'}")

if __name__ == "__main__":
    main()
