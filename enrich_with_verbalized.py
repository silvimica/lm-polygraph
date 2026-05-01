from lm_polygraph.utils import UEManager
from lm_polygraph.estimators import Consistency

import json

base_path = '/nfs-stor/statml/maiya/managers_source_ue'

models = [ 'qwen14bit']
# 'ambi', 'single',
types = [ 'single']

def load_labels(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            labels.append(row["label"])
    return labels


def compute_decision_metric(manager, type):
    labels = manager.stats["labels"]       # list of labels
    n = len(labels)

    # 1 = correct
    # 0 = incorrect or answered (for Ambiguous)
    decisions = []
    for label in labels:
        if label == "correct" or (type=='ambi' and label == 'clarification'):
            decisions.append(1)
        else:  # "incorrect", "answered", etc.
            decisions.append(0)

    # Store inside manager.gen_metrics
    manager.gen_metrics[('sequence', 'decision')] = decisions



for model in models: 
    for type in types:
        # man = UEManager.load(f'{base_path}/{type}_{model}_no_embeddings.man')
        man = UEManager.load(f'{base_path}/final/{type}_{model}.man')
        print('loaded old')
        man.stats['attention_all'] = []
        man.stats['embeddings_decoder'] = []


        estimator = Consistency()
        man.estimations[('sequence', str(estimator))] = list(estimator(man.stats))
        man.estimations[('sequence', 'UEVerbalized2S')] = [
                -x for x in man.estimations[('sequence', 'Verbalized2S')]
            ]
        labels_path = f"{base_path}/annotations_{type}_{model}.jsonl"
        labels = load_labels(labels_path)
        man.stats["labels"] = labels
        compute_decision_metric(man, type)
        man.save(save_path=f'{base_path}/final/{type}_{model}_no_embeddings.man' )
