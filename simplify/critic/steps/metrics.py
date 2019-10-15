"""
.. module:: metrics
:synopsis: metrics for model performance
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from simplify.critic.review import CriticTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'accuracy': ['sklearn.metrics', 'accuracy_score'],
    'adjusted_mutual_info': ['sklearn.metrics', 'adjusted_mutual_info_score'],
    'adjusted_rand': ['sklearn.metrics', 'adjusted_rand_score'],
    'balanced_accuracy': ['sklearn.metrics', 'balanced_accuracy_score'],
    'brier_score_loss': ['sklearn.metrics', 'brier_score_loss'],
    'calinski': ['sklearn.metrics', 'calinski_harabasz_score'],
    'davies': ['sklearn.metrics', 'davies_bouldin_score'],
    'completeness': ['sklearn.metrics', 'completeness_score'],
    'contingency_matrix': ['sklearn.metrics', 'cluster.contingency_matrix'],
    'explained_variance': ['sklearn.metrics', 'explained_variance_score'],
    'f1': ['sklearn.metrics', 'f1_score'],
    'f1_weighted': ['sklearn.metrics', 'f1_score'],
    'fbeta': ['sklearn.metrics', 'fbeta_score'],
    'fowlkes': ['sklearn.metrics', 'fowlkes_mallows_score'],
    'hamming': ['sklearn.metrics', 'hamming_loss'],
    'h_completness': ['sklearn.metrics', 'homogeneity_completeness_v_measure'],
    'homogeniety': ['sklearn.metrics', 'homogeneity_score'],
    'jaccard': ['sklearn.metrics', 'jaccard_similarity_score'],
    'mae': ['sklearn.metrics', 'median_absolute_error'],
    'matthews_corrcoef': ['sklearn.metrics', 'matthews_corrcoef'],
    'max_error': ['sklearn.metrics', 'max_error'],
    'mean_absolute_error': ['sklearn.metrics', 'mean_absolute_error'],
    'mse': ['sklearn.metrics', 'mean_squared_error'],
    'msle': ['sklearn.metrics', 'mean_squared_log_error'],
    'mutual_info': ['sklearn.metrics', 'mutual_info_score'],
    'neg_log_loss': ['sklearn.metrics', 'log_loss'],
    'norm_mutual_info': ['sklearn.metrics', 'normalized_mutual_info_score'],
    'precision': ['sklearn.metrics', 'precision_score'],
    'precision_weighted': ['sklearn.metrics', 'precision_score'],
    'r2': ['sklearn.metrics', 'r2_score'],
    'recall': ['sklearn.metrics', 'recall_score'],
    'recall_weighted': ['sklearn.metrics', 'recall_score'],
    'roc_auc': ['sklearn.metrics', 'roc_auc_score'],
    'silhouette': ['sklearn.metrics', 'silhouette_score'],
    'v_measure': ['sklearn.metrics', 'v_measure_score'],
    'zero_one': ['sklearn.metrics', 'zero_one_loss']}


@dataclass
class Metrics(CriticTechnique):
    """Measures model performance.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    technique: object = None
    parameters: object = None
    name: str = 'metrics'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        super().__post_init__()
        return self

    def _get_conditional_parameters(self, parameters, recipe = None):
        if self.technique in 'brier_score_loss':
            parameters = {
                'y_true': getattr(recipe.ingredients,
                                  'y_' + self.data_to_review),
                'y_prob': recipe.probabilities[:, 1]}
        elif self.technique in ['roc_auc']:
             parameters = {
                 'y_true': getattr(recipe.ingredients,
                                   'y_' + self.data_to_review),
                 'y_score': recipe.probabilities[:, 1]}
        return parameters

    def draft(self):
        super().draft()
        self.negative_options = [
            'brier_loss_score',
            'neg_log_loss',
            'zero_one']
        self.extra_parameters = {
            'fbeta': {'beta': 1},
            'f1_weighted': {'average': 'weighted'},
            'precision_weighted': {'average': 'weighted'},
            'recall_weighted': {'average': 'weighted'}}
        return self

    # def edit(self, name, metric, special_type = None,
    #          special_parameters = None, negative_metric = False):
    #     """Allows user to manually add a metric to report."""
    #     self.options.update({name: metric})
    #     if special_type in ['probability']:
    #         self.prob_options.update({name: metric})
    #     elif special_type in ['scorer']:
    #         self.score_options.update({name: metric})
    #     if special_parameters:
    #        self.special_options.update({name: special_parameters})
    #     if negative_metric:
    #        self.negative_options.append[name]
    #     return self

    def implement(self, recipe):
        self.runtime_parameters = {
            'y_true': getattr(recipe.ingredients, 'y_' + self.data_to_review),
            'y_pred': recipe.predictions}
        super().implement()
        return self
