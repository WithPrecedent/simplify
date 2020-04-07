"""
.. module:: metrics
:synopsis: measurements for data analysis performance
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from dataclasses.dataclasses import dataclasses.field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.base import SimpleSettings
from simplify.core.library import Technique
from simplify.core.repository import SimpleRepository
from simplify.critic.critic import Evaluator


@dataclasses.dataclass
class Metric(Technique):
    """Base class for model performance evaluation measurements.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        step (Optional[str]): name of step when the class instance is to be
            applied. Defaults to None.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): process object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    To Do:
        Add attributes for cluster metrics.

    """
    name: Optional[str] = None
    step: Optional[str] = dataclasses.field(default_factory = lambda: 'measure')
    module: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    negative: Optional[bool] = False
    probabilities: Optional[bool] = False
    actual: Optional[str] = 'y_true'
    predicted: Optional[str] = 'y_pred'
    conditional: Optional[bool] = False


@dataclasses.dataclass
class SklearnMetrics(SimpleRepository):
    """A dictonary of Evaluator options for the Analyst subpackage.

    Args:
        idea (Optional['Idea']): shared 'Idea' instance with project settings.

    To Do:
        Add attributes for cluster metrics.

    """
    idea: Optional['Idea'] = None

    """ Private Methods """

    def _cluster_metrics(self) -> None:
        self.contents = {
            'adjusted_mutual_info': Metric(
                name = 'adjusted_mutual_info',
                module = 'sklearn.metrics',
                algorithm = 'adjusted_mutual_info_score'),
            'adjusted_rand': Metric(
                name = 'adjusted_rand',
                module = 'sklearn.metrics',
                algorithm = 'adjusted_rand_score'),
            'calinski_harabasz': Metric(
                name = 'calinski_harabasz',
                module = 'sklearn.metrics',
                algorithm = 'calinski_harabasz_score'),
            'davies_bouldin': Metric(
                name = 'davies_bouldin',
                module = 'sklearn.metrics',
                algorithm = 'davies_bouldin_score'),
            'completeness': Metric(
                name = 'completeness',
                module = 'sklearn.metrics',
                algorithm = 'completeness_score'),
            'fowlkes_mallows': Metric(
                name = 'fowlkes_mallows',
                module = 'sklearn.metrics',
                algorithm = 'fowlkes_mallows_score'),
            'homogeneity': Metric(
                name = 'homogeneity',
                module = 'sklearn.metrics',
                algorithm = 'homogeneity'),
            'mutual_info': Metric(
                name = 'mutual_info',
                module = 'sklearn.metrics',
                algorithm = 'mutual_info_score'),
            'normalized_mutual_info': Metric(
                name = 'normalized_mutual_info',
                module = 'sklearn.metrics',
                algorithm = 'normalized_mutual_info_score'),
            'silhouette': Metric(
                name = 'silhouette',
                module = 'sklearn.metrics',
                algorithm = 'silhouette_score'),
            'silhouette_samples': Metric(
                name = 'accuracy',
                module = 'sklearn.metrics',
                algorithm = 'accuracy_score'),
            'v_measure': Metric(
                name = 'v_measure',
                module = 'sklearn.metrics',
                algorithm = 'v_measure_score')}
        return self

    def _classify_metrics(self) -> None:
        self.contents = {
            'accuracy': Metric(
                name = 'accuracy',
                module = 'sklearn.metrics',
                algorithm = 'accuracy_score'),
            'balanced_accuracy': Metric(
                name = 'balanced_accuracy',
                module = 'sklearn.metrics',
                algorithm = 'balanced_accuracy_score'),
            'brier_loss': Metric(
                name = 'brier_loss',
                module = 'sklearn.metrics',
                algorithm = 'brier_score_loss',
                negative = True,
                probabilities = True,
                predicated = 'y_prob'),
            'cohen_kappa': Metric(
                name = 'cohen_kappa',
                module = 'sklearn.metrics',
                algorithm = 'cohen_kappa_score',
                predicted = 'y1',
                actual = 'y2'),
            'dcg': Metric(
                name = 'dcg',
                module = 'sklearn.metrics',
                algorithm = 'dcg_score',
                probabilities = True,
                predicted = 'y_score'),
            'f1': Metric(
                name = 'f1',
                module = 'sklearn.metrics',
                algorithm = 'f1_score'),
            'fbeta': Metric(
                name = 'fbeta',
                module = 'sklearn.metrics',
                algorithm = 'fbeta_score',
                parameters = {'beta': 1}),
            'hamming_loss': Metric(
                name = 'hamming_loss',
                module = 'sklearn.metrics',
                algorithm = 'hamming_loss',
                negative = True),
            'hinge_loss': Metric(
                name = 'hinge_loss',
                module = 'sklearn.metrics',
                algorithm = 'hinge_loss',
                predicted = 'pred_decision',
                condtional = True),
            'jaccard': Metric(
                name = 'jaccard',
                module = 'sklearn.metrics',
                algorithm = 'jaccard_score'),
            'neg_log_loss': Metric(
                name = 'neg_log_loss',
                module = 'sklearn.metrics',
                algorithm = 'log_loss'),
            'matthews': Metric(
                name = 'matthews',
                module = 'sklearn.metrics',
                algorithm = 'matthews_corrcoef'),
            'ndcg': Metric(
                name = 'ndcg',
                module = 'sklearn.metrics',
                algorithm = 'ndcg_score',
                probabilities = True,
                predicted = 'y_score'),
            'precision': Metric(
                name = 'precision_',
                module = 'sklearn.metrics',
                algorithm = 'precision__score'),
            'recall': Metric(
                name = 'recall',
                module = 'sklearn.metrics',
                algorithm = 'recall_score'),
            'roc_auc': Metric(
                name = 'roc_auc',
                module = 'sklearn.metrics',
                algorithm = 'roc_auc_score',
                probabilities = True,
                predicted = 'y_score'),
            'zero_one_loss': Metric(
                name = 'zero_one_loss',
                module = 'sklearn.metrics',
                algorithm = 'zero_one_loss',
                negative = True)}
        return self

    def _regress_metrics(self) -> None:
        self.contents = {
            'adjusted_r2': Metric(
                name = 'adjusted_r2',
                module = 'simplify.critic.metrics',
                algorithm = 'adjusted_r2',
                parameters = {'data': 'data', 'r2': 'r2'}),
            'explained_variance': Metric(
                name = 'explained_variance',
                module = 'sklearn.metrics',
                algorithm = 'explained_variance_score'),
            'max_error': Metric(
                name = 'max_error',
                module = 'sklearn.metrics',
                algorithm = 'max_error'),
            'mean_absolute_error': Metric(
                name = 'mean_absolute_error',
                module = 'sklearn.metrics',
                algorithm = 'mean_absolute_error'),
            'mean_squared_error': Metric(
                name = 'mean_squared_error',
                module = 'sklearn.metrics',
                algorithm = 'mean_squared_error'),
            'mean_squared_log_error': Metric(
                name = 'mean_squared_log_error',
                module = 'sklearn.metrics',
                algorithm = 'mean_squared_log_error'),
            'median_absolute_error': Metric(
                name = 'median_absolute_error',
                module = 'sklearn.metrics',
                algorithm = 'median_absolute_error'),
            'r2': Metric(
                name = 'r2',
                module = 'sklearn.metrics',
                algorithm = 'r2_score'),
            'mean_poisson_deviance': Metric(
                name = 'mean_poisson_deviance',
                module = 'sklearn.metrics',
                algorithm = 'mean_poisson_deviance'),
            'mean_gamma_deviance': Metric(
                name = 'mean_gamma_deviance',
                module = 'sklearn.metrics',
                algorithm = 'mean_gamma_deviance'),
            'mean_tweedie_deviance': Metric(
                name = 'mean_tweedie_deviance',
                module = 'sklearn.metrics',
                algorithm = 'mean_tweedie_deviance')}
        return self

    """ Private Methods """

    def create(self) -> None:
        getattr(self, '_'.join(
            ['_', self.idea['analyst']['model_type'], 'metrics']))()
        return self


def adjusted_r2(data: 'DataBundle', r2: float) -> float:
    return 1 - (1-r2)*(len(data.y)-1)/(len(data.y)-data.x.shape[1]-1)