import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import uuid

class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __str__(self):
        return "prr"

    def get_ue_rejection(self, estimator, target, num_remaining_points):
        return np.flip(np.cumsum(target[np.argsort(estimator)]) / num_remaining_points)

    def get_oracle_rejection(self, estimator, target, num_remaining_points):
        return np.flip(np.cumsum(np.flip(np.sort(target))) / num_remaining_points)

    def get_random_rejection(self, estimator, target, num_remaining_points, N_EXAMPLES):
        random_rejection_accuracies = []
        for _ in range(1000):
            order = np.arange(0, N_EXAMPLES)
            np.random.shuffle(order)
            random_rejection_accuracies.append(np.flip(np.cumsum(target[order]) / num_remaining_points))

        return np.mean(random_rejection_accuracies, axis=0)

    def __call__(self, estimator: List[float], target: List[float], generate_curve:bool = False, e_level:str = '', e_name:str ='', gen_name:str ='', ue_metric:str ='' ) -> float:
        """
        Measures the area under the Prediction-Rejection curve between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Prediction-Rejection curve.
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)
        num_obs = len(ue)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)
        scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_obs

        if generate_curve:
            plots_dir = './plots'
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            N_EXAMPLES = len(estimator)  # Number of examples
            num_remaining_points = np.arange( 1, N_EXAMPLES + 1)

            # Compute rejection accuracies for UE, Oracle, and Random baselines
            ue_rejected_accuracy = self.get_ue_rejection(estimator, target, num_remaining_points)
            oracle_rejected_accuracy = self.get_oracle_rejection(estimator, target, num_remaining_points)
            random_rejection_accuracy = self.get_random_rejection(estimator, target, num_remaining_points, N_EXAMPLES)

            rejection_rates = np.linspace(0, 1, N_EXAMPLES)

            # Plot using Seaborn for visualization
            sns.lineplot(x=rejection_rates, y=ue_rejected_accuracy, label='UE')
            sns.lineplot(x=rejection_rates, y=oracle_rejected_accuracy, label='Oracle')
            g = sns.lineplot(x=rejection_rates, y=random_rejection_accuracy, label='Random')
            g.set_xlabel('Rejection Rate')
            g.set_ylabel(f'{gen_name}')
            g.set_title(f'PRR curve: {e_level}, {e_name}')
            g.grid()
            
            # Generate a random UUID for the filename
            base_filename = 'prr_curve'
            extension = 'png'
            unique_id = uuid.uuid4()
            new_filename = f"{base_filename}_{e_name}_{gen_name}_{unique_id}.{extension}"
            save_path = os.path.join(plots_dir, new_filename)

            plt.savefig(save_path)
            plt.close() 

        return prr_score
