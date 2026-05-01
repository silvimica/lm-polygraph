import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from typing import Tuple
from pathlib import Path
from tqdm import tqdm

from lm_polygraph.utils import UEManager

# ---------------------------
# Model loader functions (your versions)
# ---------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple

# --- Generic loader (for Llama, Qwen, etc.) ---
def load_model_generic(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model

def load_tokenizer_generic(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# --- Gemma-aware loader (handles text_config fixes + special EOS) ---
def load_model_gemma(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="eager",
    )
    # Gemma's HF wrapper exposes text_config; ensure compatibility fields exist
    if hasattr(model.config, "text_config"):
        if not hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = model.config.text_config.num_hidden_layers
        if not hasattr(model.config, "num_attention_heads"):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
    model.eval()
    return model

def load_tokenizer_gemma(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma italian variants sometimes use a special token as eos
    # Heuristic: if path ends with "-it" set eos_token_id to <end_of_turn> if present
    try:
        if model_path.split("-")[-1] == "it":
            end_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_token_id is not None and end_token_id >= 0:
                tokenizer.eos_token_id = end_token_id
    except Exception:
        pass

    return tokenizer

# --- Dispatcher: choose correct loader based on model_path string ---
def load_model_and_tokenizer(model_path: str, device_map: str = "auto") -> Tuple[object, object]:
    """
    Choose gemma loader if model_path contains 'gemma' or ends with '-it',
    otherwise use the generic loader.
    """
    key = model_path.lower()
    is_gemma = ("gemma" in key) or key.endswith("-it")
    if is_gemma:
        model = load_model_gemma(model_path, device_map)
        tokenizer = load_tokenizer_gemma(model_path)
    else:
        model = load_model_generic(model_path, device_map)
        tokenizer = load_tokenizer_generic(model_path)
    return model, tokenizer

# Prompt + parsing helpers
# ---------------------------
def build_confidence_prompt(question: str, candidate_answer: str) -> str:
    return f"""
The question is:
{question}

YOUR GUESS:
{candidate_answer}

Now:
Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.

For example:
Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
""".strip()

def parse_confidence(response_text: str) -> Tuple[float, str]:
    raw = (response_text or "").strip()

    # 1) explicit "Probability:" line (most strict/likely)
    m = re.search(r"(?mi)^\s*Probability\s*:\s*([0-9]*\.?[0-9]+)\s*(%|percent)?\s*$", raw)
    if m:
        val = float(m.group(1))
        if m.group(2):  # percent sign or the word 'percent'
            val = val / 100.0
        val = max(0.0, min(1.0, val))
        return val, raw

    # 2) inline like "Probability: 0.75" but not on own line
    m2 = re.search(r"(?mi)Probability\s*:\s*([0-9]*\.?[0-9]+)\s*(%|percent)?", raw)
    if m2:
        val = float(m2.group(1))
        if m2.group(2):
            val = val / 100.0
        val = max(0.0, min(1.0, val))
        return val, raw

    # 3) embedded JSON {"probability": ..., "score": ...}
    jmatch = re.search(r"\{.*?\}", raw, flags=re.S)
    if jmatch:
        try:
            j = json.loads(jmatch.group(0))
            if "probability" in j:
                val = float(j["probability"])
                if val > 1.0:  # maybe 0-100
                    val = val / 100.0
                val = max(0.0, min(1.0, val))
                return val, raw
            if "score" in j:
                val = float(j["score"])
                if val > 1.0:
                    val = val / 100.0
                val = max(0.0, min(1.0, val))
                return val, raw
        except Exception:
            pass

    # 4) fallback: first standalone number (could be 0-1 or 0-100)
    m3 = re.search(r"([0-9]*\.?[0-9]+)", raw)
    if m3:
        val = float(m3.group(1))
        # interpret >1 as percent (0..100)
        if val > 1.0:
            val = val / 100.0
        val = max(0.0, min(1.0, val))
        return val, raw

    # 5) nothing usable found
    return 0.0, raw

# ---------------------------
# Generation wrapper (single sample)
# ---------------------------
def ask_model_confidence(model, tokenizer, input_text: str, model_output: str, max_new_tokens: int = 128):
    """
    Generate evaluator reply from the model and parse confidence.
    Returns (conf_0_1, explanation, raw_reply)
    """
    prompt = build_confidence_prompt(input_text, model_output)

    # Determine device for tensors (don't forcibly move model if device_map used)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        # fallback CPU
        model_device = torch.device("cpu")

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        # keep deterministic
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )

    # decode and remove prompt if echoed
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    reply = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded

    conf, expl = parse_confidence(reply)
    return conf, expl, reply

import gc
import time

def unload_model(model, tokenizer):
    """Delete model and tokenizer objects and free GPU memory."""
    try:
        del tokenizer
    except Exception:
        pass
    try:
        # move model to cpu then delete, sometimes helpful for pinned memory
        model.to("cpu")
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # small pause can help on some clusters
    time.sleep(1.0)

def process_one_model(model_name: str, model, tokenizer, man_root: Path, save_path: str, types: list):
    """Process managers for a single loaded model/tokenizer pair."""
    for t in types:
        man_path = man_root / f"{t}_{model_name}.man"
        if not man_path.exists():
            print(f"Manager file not found: {man_path} — skipping")
            continue

        print(f"\nLoading manager: {man_path}")
        try:
            man = UEManager.load(str(man_path))
        except Exception as e:
            print(f"Failed to load manager {man_path}: {e}")
            continue

        # Try to obtain arrays from manager; fallbacks are defensive
        inputs = None
        outputs = None
        try:
            stats = man.stats
            inputs = stats["input_texts"]
            outputs = stats["greedy_texts"]
        except Exception:
            inputs = None
            outputs = None

        if not inputs or not outputs:
            print(f"Manager {man_path} missing expected stats fields (input_texts / greedy_texts). Skipping.")
            continue

        # prepare estimations container
        if not hasattr(man, "estimations") or man.estimations is None:
            man.estimations = {}

        key = ('sequence', 'Verbalized2S')
        if key not in man.estimations or man.estimations.get(key) is None:
            man.estimations[key] = []

        # per-example verbose list
        if not hasattr(man, "estimations_verbose") or man.estimations_verbose is None:
            man.estimations_verbose = []

        n_examples = min(len(inputs), len(outputs))
        print(f"Processing {n_examples} examples for {t}_{model_name} ...")
        for i in tqdm(range(n_examples), desc=f"{t}_{model_name}", unit="ex"):
            inp = inputs[i]
            outp = outputs[i]
            try:
                conf, explanation, raw = ask_model_confidence(model, tokenizer, inp, outp, max_new_tokens=36)
            except Exception as e:
                print(f"Generation failed for example {i}: {e}")
                conf, explanation, raw = 0.0, f"generation_error: {e}", ""

            man.estimations[key].append(conf)
            man.estimations_verbose.append({
                "idx": i,
                "input": inp,
                "output": outp,
                "confidence": conf,
                "explanation": explanation,
                "raw_reply": raw,
            })

        # Save manager back to disk
        out_save_path = f"{save_path}/{t}_{model_name}.man"
        try:
            # try signature with save_path kwarg
            man.save(save_path=out_save_path)
        except TypeError:
            try:
                man.save(out_save_path)
            except Exception as e:
                print(f"Warning: could not save manager using known signatures: {e}")

        print(f"Saved manager with {len(man.estimations[key])} confidences at {out_save_path}")

def main():
    # 'llama8bit' , 'gemma12bit',    
    models = ['qwen14bit']
    # 'single', 
    types = ['single']
    model_paths = {
        'llama8bit': 'meta-llama/Llama-3.1-8B-Instruct',
        'gemma12bit': 'google/gemma-3-12b-it',
         'qwen14bit': 'Qwen/Qwen2.5-14B-Instruct',
    }

    man_root = Path('/nfs-stor/statml/maiya/managers_source_ue')
    save_path = '/nfs-stor/statml/maiya/managers_source_ue/final'

    # Device map behavior - keep "auto" unless you need a specific device.
    device_map = "auto"

    for model_name in models:
        path = model_paths.get(model_name)
        if path is None:
            print(f"Skipping {model_name}: no path found.")
            continue

        print(f"\n=== Processing model: {model_name} (path={path}) ===")

        # Try loading with memory-saving kwargs
        try:
            # you can add load_in_8bit=True if using bitsandbytes and have it installed
            model, tokenizer = load_model_and_tokenizer(path, device_map=device_map)
            print(f"Loaded {model_name}. Device: {next(model.parameters()).device}")
        except Exception as e:
            print(f"Failed to load {model_name} ({path}): {e}")
            print("Trying with low_cpu_mem_usage=True fallback...")
            try:
                # fallback: direct HF load with low_cpu_mem_usage to reduce peak CPU RAM
                model = AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, device_map=device_map, low_cpu_mem_usage=True
                )
                tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", add_bos_token=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"Loaded {model_name} (fallback).")
            except Exception as e2:
                print(f"Second load attempt failed: {e2}. Skipping {model_name}.")
                # ensure memory cleared before continuing
                try:
                    del model, tokenizer
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # Run the per-model processing (this will iterate the manager files for that model)
        try:
            process_one_model(model_name, model, tokenizer, man_root, save_path, types)
        except Exception as e:
            print(f"Error while processing {model_name}: {e}")
        finally:
            # unload model before next iteration
            print(f"Unloading {model_name} and freeing memory...")
            unload_model(model, tokenizer)

    print("\nAll models processed. Done.")

if __name__ == "__main__":
    main()

