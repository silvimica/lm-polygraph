import numpy as np

from collections import defaultdict
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class SemanticClassesCalculator(StatCalculator):
    """
    Paritions samples into semantic classes based on semantic matrix.
    """

    def __init__(self):
        super().__init__(
            [
                "semantic_classes_entail",
            ],
            [
                "sample_texts",
                "semantic_matrix_entail",
                "semantic_matrix_classes",
                "entailment_id",
            ],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch_entailment = (
            dependencies["semantic_matrix_classes"] == dependencies["entailment_id"][0]
        )

        result = []
        for hyp, entailment in zip(dependecies["sample_texts"], batch_entailment):
            sample_to_class, class_to_sample = self.get_classes(hyp, entailment)
            result.append(
                {
                    "sample_to_class": sample_to_class,
                    "class_to_sample": class_to_sample,
                }
            )

        return {
            "semantic_classes_entail": result
        }

    def get_classes(self, hyp: List[str], entailment: np.ndarray):
        class_to_sample = []
        sample_to_class = {}

        [
            self.determine_class(i, class_to_sample, sample_to_class, entailment)
            for i in range(len(hyp))
        ]

        return sample_to_class, class_to_sample

    def determine_class(self, idx: int, i: int, class_to_sample: List, sample_to_class: Dict, entailment: np.ndarray):
        # For first hypo just create a zeroth class
        if i == 0:
            class_to_sample.append([0])
            sample_to_class[0] = 0

            return 0

        # Iterate over existing classes and return if hypo belongs to one of them
        for class_id in range(len(class_to_sample)):
            class_text_id = class_to_sample[class_id][0]
            forward_entailment = entailment[class_text_id, i]
            backward_entailment = entailment[i, class_text_id]
            if forward_entailment and backward_entailment:
                class_to_sample[class_id].append(i)
                sample_to_class[i] = class_id

                return class_id

        # If none of the existing classes satisfy - create new one
        new_class_id = len(class_to_sample)
        sample_to_class[i] = new_class_id
        class_to_sample.append([i])

        return new_class_id
