from lm_polygraph.utils import UEManager
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.estimators import NLIConsistency, NLICocoaMSP, NLICocoaPPL, NLICocoaMTE
from collections import defaultdict
from types import SimpleNamespace
import copy
import pandas as pd
import numpy as np

base_path = '/l/users/maiya.goloburda/why_dk/with_labels/with_semantic'

models = ['gemma12bit', 'llama8bit', 'qwen14bit']
types = ['single', 'ambi','multi']
max_rejections = [ 0.5, 0.75, 1] 


methods = [NLIConsistency(), NLICocoaMSP(), NLICocoaPPL()]

ue_metrics = [PredictionRejectionArea(max_rejection=mr) for mr in max_rejections]

for type in types:
    for model in models:
        man = UEManager.load(f'{base_path}/{type}_{model}.man')
        man.ue_metrics=ue_metrics
        for method in methods:
            man.estimations[('sequence', str(method))] = list(method(man.stats))
            print(len(man.estimations[('sequence', str(method))]))
        
        cocoa = man.estimations[('sequence', 'CocoaMTE')]
        gss = man.stats['greedy_sentence_similarity']

        # compute avg dissimilarity per example
        avg_dissimilarity = np.array([
            np.mean(1 - sim) for sim in gss
        ])

        # recover mean entropy
        mean_entropy = cocoa / avg_dissimilarity
        man.stats['mean_entropy'] = mean_entropy
        print(len(mean_entropy))
        MTE = NLICocoaMTE()
        man.estimations[('sequence', str(MTE))]  =  list(MTE(man.stats))
        print(len(man.estimations[('sequence', str(MTE))]))
        man.eval_ue()
        man.save(save_path=f'{base_path}/{type}_{model}.man')
        print(f'All done for {type} and {model}')