import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator


class SemanticEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(
        self, verbose: bool = False, class_probability_estimation: str = "sum"
    ):
        self.class_probability_estimation = class_probability_estimation
        if self.class_probability_estimation == "sum":
            deps = ["sample_log_probs", "sample_texts", "semantic_classes_entail"]
        elif self.class_probability_estimation == "frequency":
            deps = ["sample_texts", "semantic_classes_entail"]
        else:
            raise ValueError(
                f"Unknown class_probability_estimation: {self.class_probability_estimation}. Use 'sum' or 'frequency'."
            )

        super().__init__(deps, "sequence")
        self.verbose = verbose

    def __str__(self):
        if self.class_probability_estimation == "sum":
            return "SemanticEntropy"
        elif self.class_probability_estimation == "frequency":
            return "SemanticEntropyEmpirical"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        if self.class_probability_estimation == "sum":
            loglikelihoods_list = stats["sample_log_probs"]
            hyps_list = stats["sample_texts"]
        elif self.class_probability_estimation == "frequency":
            loglikelihoods_list = None
            hyps_list = stats["sample_texts"]

        return self.batched_call(hyps_list, loglikelihoods_list, stats["semantic_classes_entail"])

    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: Optional[List[List[float]]],
        classes: List[Dict],
        log_weights: Optional[List[List[float]]] = None,
    ) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_to_sample = classes[i]["class_to_sample"]
            sample_to_class = classes[i]["sample_to_class"]

            if self.class_probability_estimation == "sum":
                class_likelihoods = [
                    np.array(loglikelihoods_list[i])[np.array(class_idx)]
                    for class_idx in class_to_sample
                ]
                class_lp = [
                    np.logaddexp.reduce(likelihoods)
                    for likelihoods in class_likelihoods
                ]
            elif self.class_probability_estimation == "frequency":
                num_samples = len(hyps_list[i])
                class_lp = np.log(
                    [
                        len(class_idx) / num_samples
                        for class_idx in class_to_sample
                    ]
                )

            if log_weights[i] is None:
                log_weights[i] = [0 for _ in hyps_list[i]]
            semantic_logits[i] = -np.mean(
                [
                    class_lp[sample_to_class[j]] * np.exp(log_weights[i][j])
                    for j in range(len(hyps_list[i]))
                ]
            )
        return np.array([semantic_logits[i] for i in range(len(hyps_list))])
