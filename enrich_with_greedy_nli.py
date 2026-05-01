
# save as scripts/enrich_managers_with_nli.py
import itertools
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from typing import Dict, List


from lm_polygraph.utils import UEManager
# import your DeBERTa wrapper / model object type here
# from my_nli import DebertaWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import the base calculator class you already have
from lm_polygraph.stat_calculators import BaseGreedySemanticMatrixCalculator
# <- replace the import path above with real import in your project
class DebertaWrapper:
    def __init__(self, model_name: str, batch_size: int = 10, device: str = None):
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer + model
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.deberta = AutoModelForSequenceClassification.from_pretrained(model_name)
        # ensure the model is on device
        self.deberta.to(self.device)
        self.deberta.eval()

        # Normalize label2id keys to uppercase strings for safe lookup
        # Many models use 'contradiction','entailment','neutral' lowercased; base code expects uppercase keys
        cfg = self.deberta.config
        if getattr(cfg, "label2id", None) is None:
            # Build a label2id if possible from id2label
            if getattr(cfg, "id2label", None) is not None:
                cfg.label2id = {v.upper(): int(k) for k, v in cfg.id2label.items()}
            else:
                raise RuntimeError("Model config lacks label2id and id2label; cannot determine label IDs.")

        # create a safe mapping accessible as config.label2id with uppercase keys
        cfg.label2id = {k.upper(): v for k, v in cfg.label2id.items()}



class SimpleGreedyCalculator(BaseGreedySemanticMatrixCalculator):
    """
    Concrete, minimal implementation so we can instantiate the base and call calculate_semantic_matrix.
    """

    @staticmethod
    def meta_info():
        stats = [
            "greedy_semantic_matrix_entail_forward",
            "greedy_semantic_matrix_entail_backward",
            "greedy_semantic_matrix_entail",
            "greedy_semantic_matrix_neutral_forward",
            "greedy_semantic_matrix_neutral_backward",
            "greedy_semantic_matrix_neutral",
            "greedy_semantic_matrix_contra_forward",
            "greedy_semantic_matrix_contra_backward",
            "greedy_semantic_matrix_contra",
        ]
        dependencies = ["greedy_texts", "sample_texts"]
        return stats, dependencies

    def __init__(self, nli_model):
        super().__init__(nli_model)

    def __call__(
        self,
        dependencies: Dict[str, List[str]],
        texts: List[str] = None,
        model = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        # prepare batch_pairs and batch_invs exactly like GreedySemanticMatrixCalculator
        batch_texts = dependencies["sample_texts"]
        batch_greedy_texts = dependencies["greedy_texts"]

        batch_pairs = []
        batch_invs = []
        for texts_i, greedy_text in zip(batch_texts, batch_greedy_texts):
            unique_texts, inv = np.unique(texts_i, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_invs.append(inv)

        # call the heavy lifting implemented in the base class
        E_f, E_b, E, N_f, N_b, N, C_f, C_b, C = self.calculate_semantic_matrix(
            batch_pairs, batch_invs
        )

        return {
            "greedy_semantic_matrix_entail_forward": E_f,
            "greedy_semantic_matrix_entail_backward": E_b,
            "greedy_semantic_matrix_entail": E,
            "greedy_semantic_matrix_neutral_forward": N_f,
            "greedy_semantic_matrix_neutral_backward": N_b,
            "greedy_semantic_matrix_neutral": N,
            "greedy_semantic_matrix_contra_forward": C_f,
            "greedy_semantic_matrix_contra_backward": C_b,
            "greedy_semantic_matrix_contra": C,
        }

def enrich_managers_batch(
    base_path: str,
    models: list,
    managers: list,
    nli_model,
    save_overwrite: bool = True,
):
    """
    For each manager file at {base_path}/{manager}_{model}.man:
      - load manager
      - build batch_pairs + batch_invs (same logic as GreedySemanticMatrixCalculator)
      - call BaseGreedySemanticMatrixCalculator.calculate_semantic_matrix
      - write outputs into manager.stats under expected keys
      - save manager (attempt man.save() or write to same path)
    """

    calc = SimpleGreedyCalculator(nli_wrapper)


    for model_name in models:
        for manager_name in managers:
            man_path = Path(f"{base_path}/{manager_name}_{model_name}.man")
            print(f"\nProcessing {man_path} ...")
            if not man_path.exists():
                print(f"  Skipping (file not found): {man_path}")
                continue

            # Load manager (you used UEManager.load previously)
            man = UEManager.load(str(man_path))

            # ---- read the two required stats ----
            try:
                greedy_texts = man.stats["greedy_texts"]
                sample_texts = man.stats["sample_texts"]
            except KeyError as e:
                print(f"  Missing required stat key: {e}. Skipping.")
                continue

            # Normalize sample_texts into an iterable of lists (one entry per batch element)
            if isinstance(sample_texts, np.ndarray):
                # assume shape (batch, k)
                sample_texts_list = [list(row) for row in sample_texts]
            else:
                # If it's already a list-of-lists or list-of-arrays
                sample_texts_list = [list(x) for x in sample_texts]

            # Ensure greedy_texts is an iterable aligned with sample_texts_list
            if isinstance(greedy_texts, np.ndarray):
                greedy_texts_list = list(greedy_texts)
            else:
                greedy_texts_list = list(greedy_texts)

            if len(greedy_texts_list) != len(sample_texts_list):
                print(
                    f"  Warning: len(greedy_texts) ({len(greedy_texts_list)}) != "
                    f"len(sample_texts) ({len(sample_texts_list)}). Attempting to continue."
                )

            # Build batch_pairs and batch_invs exactly as in GreedySemanticMatrixCalculator
            batch_pairs = []
            batch_invs = []
            for samples, greedy_text in zip(sample_texts_list, greedy_texts_list):
                unique_texts, inv = np.unique(samples, return_inverse=True)
                # itertools.product([greedy_text], unique_texts) -> list of (greedy, unique)
                pairs = list(itertools.product([greedy_text], unique_texts))
                batch_pairs.append(pairs)
                batch_invs.append(inv)

            # Call the heavy-lifting function from your base calculator
            # it returns: (E_f, E_b, E, N_f, N_b, N, C_f, C_b, C)
            try:
                outputs = calc.calculate_semantic_matrix(batch_pairs, batch_invs)
            except Exception as exc:
                print(f"  Error while calculating semantic matrices: {exc}")
                raise

            (
                E_f,
                E_b,
                E,
                N_f,
                N_b,
                N,
                C_f,
                C_b,
                C,
            ) = outputs

            # Write back into manager.stats using same keys as meta_info()
            man.stats.update(
                {
                    "greedy_semantic_matrix_entail_forward": E_f,
                    "greedy_semantic_matrix_entail_backward": E_b,
                    "greedy_semantic_matrix_entail": E,
                    "greedy_semantic_matrix_neutral_forward": N_f,
                    "greedy_semantic_matrix_neutral_backward": N_b,
                    "greedy_semantic_matrix_neutral": N,
                    "greedy_semantic_matrix_contra_forward": C_f,
                    "greedy_semantic_matrix_contra_backward": C_b,
                    "greedy_semantic_matrix_contra": C,
                }
            )

            # Try to save manager. UEManager may define .save() or a different API.
            man.save(save_path=f'{base_path}/with_semantic/{manager_name}_{model_name}.man')

    print("\nAll done.")

# Example usage:
if __name__ == "__main__":
    models = ["llama8bit", "gemma12bit", "qwen"]
    managers = ["single", "multi", "ambi"]
    base_path = "/l/users/maiya.goloburda/why_dk/with_labels"

    # ---- YOU MUST provide your NLI wrapper instance here ----
    # `nli_model` must match what BaseGreedySemanticMatrixCalculator expects:
    # attributes used:
    #   nli_model.batch_size (int)
    #   nli_model.deberta (HF model with .config.label2id and callable .to(device))
    #   nli_model.deberta_tokenizer (transformers tokenizer)
    #   nli_model.device (torch device string)
    #
    # e.g.
    # nli_model = DebertaWrapper.from_pretrained("microsoft/deberta-large-mnli")
    hf_model_name = "microsoft/deberta-large-mnli"
    deberta_batch_size = 10

    print("Loading DeBERTa model (this may take a minute)...")
    nli_wrapper = DebertaWrapper(hf_model_name, batch_size=deberta_batch_size)

    enrich_managers_batch(base_path, models, managers, nli_wrapper)

