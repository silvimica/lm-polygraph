from lm_polygraph.utils import UEManager

import json
import copy
from collections import defaultdict
from types import SimpleNamespace
from lm_polygraph.ue_metrics import PredictionRejectionArea
import pandas as pd

base_path = '/nfs-stor/statml/maiya/managers_source_ue'

models = ['llama8bit', 'gemma12bit', 'qwen14bit']

types = ['ambi', 'single', 'multi']


def load_labels(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            labels.append(row["label"])
    return labels


for model in models: 
    man_single = UEManager.load(f'{base_path}/final/single_{model}_no_embeddings.man')
    man_multi = UEManager.load(f'{base_path}/final/multi_{model}_no_embeddings.man')
    man_ambi = UEManager.load(f'{base_path}/final/ambi_{model}_no_embeddings.man')
    managers = {
    'Ambiguous input': man_ambi,
    'Multi-answer':    man_multi,
    'Single answer':   man_single,
}

    label_files = {
        'Ambiguous input': f"{base_path}/annotations_ambi_{model}.jsonl",
        'Multi-answer':    f"{base_path}/annotations_multi_{model}.jsonl",
        'Single answer':   f"{base_path}/annotations_single_{model}.jsonl",
    }
    for scenario, man in managers.items():
        labels = load_labels(label_files[scenario])
        man.stats["labels"] = labels

    def compute_decision_metric(manager, type):
        labels = manager.stats["labels"]       # list of labels
        n = len(labels)

        decisions = []
        for label in labels:
            if label == "correct" or (type=='Ambiguous input' and label == 'clarification'):
                decisions.append(1)
            else:  # "incorrect", "answered", etc.
                decisions.append(0)

        # Store inside manager.gen_metrics
        manager.gen_metrics[('sequence', 'decision')] = decisions

    for scenario, man in managers.items():
        compute_decision_metric(man, scenario)
    


    combined = SimpleNamespace(
        stats={},
        estimations=defaultdict(list),
        metrics=defaultdict(list),
        gen_metrics=defaultdict(list),
    )

    for scenario_name, man in managers.items():
        # ---- stats ----
        for key, value in man.stats.items():
            if isinstance(value, list):
                
                # per-generation list -> concatenate
                combined.stats.setdefault(key, []).extend(value)
            else:
                # scalar/statistic -> store per scenario
                combined.stats.setdefault(key, {})[scenario_name] = value

        # add scenario label per generation (based on labels length)
        n = len(man.stats["labels"])
        combined.stats.setdefault("scenario", []).extend([scenario_name] * n)

        # ---- estimations ----
        for key, values in man.estimations.items():         # key is ('sequence', method)
            combined.estimations[key].extend(values)

        # ---- metrics ----
        for key, value in man.metrics.items():
            if isinstance(value, list):
                combined.metrics[key].extend(value)
            else:
                combined.metrics.setdefault(key, {})[scenario_name] = value

        # ---- gen_metrics ----
        for key, value in man.gen_metrics.items():
            if isinstance(value, list):
                combined.gen_metrics[key].extend(value)
            else:
                combined.gen_metrics.setdefault(key, {})[scenario_name] = value

    managers_single_ambi = { 'Ambiguous input': man_ambi, 'Single answer': man_single }

    # create an empty combined manager-like object
    single_ambi = SimpleNamespace(
        stats={},
        estimations=defaultdict(list),
        metrics=defaultdict(list),
        gen_metrics=defaultdict(list),
    )

    for scenario_name, man in managers_single_ambi.items():
        # ---- stats ----
        for key, value in man.stats.items():
            if isinstance(value, list):
                
                # per-generation list -> concatenate
                single_ambi.stats.setdefault(key, []).extend(value)
            else:
                # scalar/statistic -> store per scenario
                single_ambi.stats.setdefault(key, {})[scenario_name] = value

        # add scenario label per generation (based on labels length)
        n = len(man.stats["labels"])
        single_ambi.stats.setdefault("scenario", []).extend([scenario_name] * n)

        # ---- estimations ----
        for key, values in man.estimations.items():         # key is ('sequence', method)
            single_ambi.estimations[key].extend(values)

        # ---- metrics ----
        for key, value in man.metrics.items():
            if isinstance(value, list):
                single_ambi.metrics[key].extend(value)
            else:
                single_ambi.metrics.setdefault(key, {})[scenario_name] = value

        # ---- gen_metrics ----
        for key, value in man.gen_metrics.items():
            if isinstance(value, list):
                single_ambi.gen_metrics[key].extend(value)
            else:
                single_ambi.gen_metrics.setdefault(key, {})[scenario_name] = value


    order = ['Ambiguous input', 'Multi-answer', 'Single answer']

    base_name = order[0]
    combined = copy.deepcopy(managers[base_name])  # start from first manager

    n_base = len(combined.stats["labels"])
    combined.stats["scenario"] = [base_name] * n_base  # per-generation scenario label
    def ensure_scenario_dict(stats_or_metrics, key, scenario_name):
        """Turn a scalar value into {scenario_name: value} if needed."""
        val = stats_or_metrics.get(key)
        if isinstance(val, dict):
            return
        stats_or_metrics[key] = {scenario_name: val}

    for k, v in list(combined.stats.items()):
        if not isinstance(v, list) and k != "scenario":
            ensure_scenario_dict(combined.stats, k, base_name)

    for k, v in list(combined.metrics.items()):
        if not isinstance(v, list):
            ensure_scenario_dict(combined.metrics, k, base_name)

    for k, v in list(combined.gen_metrics.items()):
        if not isinstance(v, list):
            ensure_scenario_dict(combined.gen_metrics, k, base_name)

    def append_manager(combined, other, other_name):
        # --- stats ---
        n_other = len(other.stats["labels"])
        combined.stats["scenario"].extend([other_name] * n_other)

        for key, val in other.stats.items():
            if key == "scenario":
                continue
            if isinstance(val, list):
                combined.stats.setdefault(key, [])
                combined.stats[key].extend(val)
            else:
                # per-scenario scalar
                ensure_scenario_dict(combined.stats, key, other_name)
                combined.stats[key][other_name] = val

        # --- estimations ---
        for key, vals in other.estimations.items():  # key is ('sequence', method)
            combined.estimations.setdefault(key, [])
            combined.estimations[key].extend(vals)

        # --- metrics ---
        for key, val in other.metrics.items():
            if isinstance(val, list):
                combined.metrics.setdefault(key, [])
                combined.metrics[key].extend(val)
            else:
                ensure_scenario_dict(combined.metrics, key, other_name)
                combined.metrics[key][other_name] = val

        # --- gen_metrics ---
        for key, val in other.gen_metrics.items():
            if isinstance(val, list):
                combined.gen_metrics.setdefault(key, [])
                combined.gen_metrics[key].extend(val)
            else:
                ensure_scenario_dict(combined.gen_metrics, key, other_name)
                combined.gen_metrics[key][other_name] = val
    for name in order[1:]:
        append_manager(combined, managers[name], name)


    order = ['Ambiguous input','Single answer']

    base_name = order[0]
    single_ambi = copy.deepcopy(managers_single_ambi[base_name])  # start from first manager

    n_base = len(single_ambi.stats["labels"])
    single_ambi.stats["scenario"] = [base_name] * n_base  # per-generation scenario label
    def ensure_scenario_dict(stats_or_metrics, key, scenario_name):
        """Turn a scalar value into {scenario_name: value} if needed."""
        val = stats_or_metrics.get(key)
        if isinstance(val, dict):
            return
        stats_or_metrics[key] = {scenario_name: val}

    for k, v in list(single_ambi.stats.items()):
        if not isinstance(v, list) and k != "scenario":
            ensure_scenario_dict(single_ambi.stats, k, base_name)

    for k, v in list(single_ambi.metrics.items()):
        if not isinstance(v, list):
            ensure_scenario_dict(single_ambi.metrics, k, base_name)

    for k, v in list(single_ambi.gen_metrics.items()):
        if not isinstance(v, list):
            ensure_scenario_dict(single_ambi.gen_metrics, k, base_name)

    def append_manager(single_ambi, other, other_name):
        # --- stats ---
        n_other = len(other.stats["labels"])
        single_ambi.stats["scenario"].extend([other_name] * n_other)

        for key, val in other.stats.items():
            if key == "scenario":
                continue
            if isinstance(val, list):
                single_ambi.stats.setdefault(key, [])
                single_ambi.stats[key].extend(val)
            else:
                # per-scenario scalar
                ensure_scenario_dict(combined.stats, key, other_name)
                single_ambi.stats[key][other_name] = val

        # --- estimations ---
        for key, vals in other.estimations.items():  # key is ('sequence', method)
            single_ambi.estimations.setdefault(key, [])
            single_ambi.estimations[key].extend(vals)

        for key, val in other.gen_metrics.items():
            if isinstance(val, list):
                single_ambi.gen_metrics.setdefault(key, [])
                single_ambi.gen_metrics[key].extend(val)
            else:
                ensure_scenario_dict(combined.gen_metrics, key, other_name)
                single_ambi.gen_metrics[key][other_name] = val
    for name in order[1:]:
        append_manager(single_ambi, managers_single_ambi[name], name)


    ue_metrics = [PredictionRejectionArea(max_rejection=0.5), PredictionRejectionArea(), PredictionRejectionArea(max_rejection=0.75)]
    combined.ue_metrics =ue_metrics
    combined.eval_ue()

    single_ambi.ue_metrics = ue_metrics
    single_ambi.eval_ue()

    man_single.ue_metrics = ue_metrics
    man_single.eval_ue()

    man_multi.ue_metrics = ue_metrics
    man_multi.eval_ue()


    def extract_prr(metrics, max_rejection = None):
        rows = []
        for key, value in metrics.items():
            if max_rejection:
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and key[0] == "sequence"
                    and key[2] == "decision"
                    and key[3] == f"prr_{max_rejection}_normalized"
                ):
                    method = key[1]
                    rows.append({"method": method, "prr": float(value)})
            else:
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and key[0] == "sequence"
                    and key[2] == "decision"
                    and key[3] == "prr_normalized"
                ):
                    method = key[1]
                    rows.append({"method": method, "prr": float(value)})

        df = pd.DataFrame(rows).set_index("method")
        return df["prr"]

    mrs = [0.5 , 0.75, None]

    for mr in mrs:
        s_combined = extract_prr(combined.metrics, max_rejection=mr).rename("combined")
        s_multi    = extract_prr(man_multi.metrics, max_rejection=mr).rename("multi")
        s_single   = extract_prr(man_single.metrics, max_rejection=mr).rename("single")
        s_single_ambi   = extract_prr(single_ambi.metrics, max_rejection=mr).rename("single_ambi")

        df = pd.concat([
        s_single.rename("prr").to_frame().assign(type="single"),
        s_multi.rename("prr").to_frame().assign(type="multi"),
        s_combined.rename("prr").to_frame().assign(type="combined"),
        s_single_ambi.rename("prr").to_frame().assign(type="single_ambi")
    ], axis=0)

        # df = df.reset_index().rename(columns={"index": "method"})

        filename = f"results/{model}_prr_results_{mr}_reject_with_NLI_and_verb.csv"
        df["method"] = df.index
        df.to_csv(filename, index=False)
        print("Saved to:", filename)
