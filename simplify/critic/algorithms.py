"""
critic.algorithms
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    

    
"""

from __future__ import annotations
import copy
import dataclasses
import functools
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import numpy as np
import pandas as pd
import scipy
import sklearn

import simplify
import sourdough



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
        idea (Optional[Idea]): shared 'Idea' instance with project settings.

    To Do:
        Add attributes for cluster metrics.

    """
    idea: Optional[core.Idea] = None

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


@dataclasses.dataclass
class ConfusionMatrix(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None

    """ Private Methods """

    def _create_report(self,
            actual: Union[np.ndarray, pd.Series],
            prediction: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(
            self.algorithm(
                actual,
                prediction,
                labels = ['yes', 'no']),
                index = ['actual:yes', 'actual:no'],
                columns =['predicted:yes', 'predicted:no'])

    """ Core siMpLify Methods """

    def apply(self, data: 'Chapter') -> 'Chapter':
        self.algorithm = algorithm.load('algorithm')
        actual = getattr(data.data, '_'.join(
            'y', self.idea['critic']['data_to_review']))
        for key, prediction in data.predictions:
            new_key = '_'.join('classification', key)
            data.reports[new_key] = self._create_report(
                prediction = prediction,
                actual = actual)
        return data


@dataclasses.dataclass
class ClassificationReport(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None

    """ Private Methods """

    def _create_report(self,
            actual: Union[np.ndarray, pd.Series],
            prediction: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(
            self.algorithm(actual, prediction, output_dict = True)).transpose()

    """ Core siMpLify Methods """

    def apply(self, data: 'Chapter') -> 'Chapter':
        self.algorithm = algorithm.load('algorithm')
        actual = getattr(data.data, '_'.join(
            'y', self.idea['critic']['data_to_review']))
        for key, prediction in data.predictions:
            new_key = '_'.join('classification', key)
            data.reports[new_key] = self._create_report(
                prediction = prediction,
                actual = actual)
        return data


@dataclasses.dataclass
class SimplifyReporter(Reporter):
    """Summary report for Analyst performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None

# @dataclasses.dataclass
# class Article(object):

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _add_row(self, recipe, report):
#         new_row = pd.Series(index = self.columns)
#         for column, variable in self.required_columns.items():
#             new_row[column] = getattr(recipe, variable)
#         for column in report:
#             new_row[column] = report[column]
#         self.text.loc[len(self.text)] = new_row
#         return self

#     def _check_best(self, recipe):
#         """Checks if the current recipe is better than the current best recipe
#         based upon the primary scoring metric.

#         Args:
#             recipe: an instance of Recipe to be tested versus the current best
#                 recipe stored in the 'best_recipe' attribute.
#         """
#         if not self._exists('best_recipe'):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         elif (self.article.loc[
#                 self.article.index[-1],
#                 utilities.listify(self.metrics)[0]] > self.best_recipe_score):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         return self

#     def _format_step(self, attribute):
#         if getattr(self.recipe, attribute).step in ['none', 'all']:
#             step_column = getattr(self.recipe, attribute).step
#         else:
#             step = getattr(self.recipe, attribute).step
#             parameters = getattr(self.recipe, attribute).parameters
#             step_column = f'{step}, parameters = {parameters}'
#         return step_column

#     def _get_step_name(self, step):
#         """Returns appropriate algorithm to the report attribute."""
#         if step.step in ['none', 'all']:
#             return step.step
#         else:
#             return step.algorithm

#     def print_best(self):
#         """Prints output to the console about the best recipe."""
#         if self.verbose:
#             print('The best test recipe, based upon the',
#                   utilities.listify(self.metrics)[0], 'metric with a score of',
#                   f'{self.best_recipe_score: 4.4f}', 'is:')
#             for step in getattr(self,
#                     self.iterator).best_recipe.steps:
#                 print(step.capitalize(), ':',
#                       getattr(getattr(self, self.iterator).best_recipe,
#                               step).step)
#         return

#     def _set_columns(self, recipe):
#         self.required_columns = {
#             'recipe_number': 'number',
#             'options': 'steps',
#             'seed': 'seed',
#             'validation_set': 'using_val_set'}
#         self.columns = list(self.required_columns.keys())
#         self.columns.extend(recipe.steps)
#         for step in self.steps:
#             if (hasattr(getattr(self, step), 'columns')
#                     and getattr(self, step).name != 'summarize'):
#                 self.columns.extend(getattr(self, step).columns)
#         return self

#     def _start_report(self, recipe):
#         self._set_columns(recipe = recipe)
#         self.text = pd.DataFrame(columns = self.columns)
#         return self

#     """ Public Import/Export Methods """

#     def save(self, report = None):
#         """Exports the review report to disk.

#         Args:
#             review(Review.report): 'report' from an instance of review
#         """
#         self.clerk.save(
#             variable = report,
#             folder = self.clerk.experiment,
#             file_name = self.model_type + '_review',
#             file_format = 'csv',
#             header = True)
#         return


# eli5_explanation = Technique(
#     name = 'eli5_explanation',
#     module = 'eli5',
#     algorithm = 'explain_prediction_df')


# """ Prediction SimpleRepository """

# prediction_gini = Technique(
#     name = 'gini_predictions',
#     module = 'self',
#     algorithm = '_get_gini_predictions')
# prediction_shap = Technique(
#     name = 'shap_predictions',
#     module = 'self',
#     algorithm = '_get_shap_predictions')

# """ Probability Teachniques """

# probability_gini = Technique(
#     name = 'gini_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_gini_probabilities')
# probability_log = Technique(
#     name = 'log_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_log_probabilities')
# probability_shap = Technique(
#     name = 'shap_predicted_probabilities',
#     module = 'self',
#     algorithm = '_get_shap_probabilities')

# """ Ranking SimpleRepository """

# rank_gini = Technique(
#     name = 'gini_importances',
#     module = 'self',
#     algorithm = '_get_gini_importances')
# rank_eli5 = Technique(
#     name = 'eli5_importances',
#     module = 'self',
#     algorithm = '_get_permutation_importances')
# rank_permutation = Technique(
#     name = 'permutation_importances',
#     module = 'self',
#     algorithm = '_get_eli5_importances')
# rank_shap = Technique(
#     name = 'shap_importances',
#     module = 'self',
#     algorithm = '_get_shap_importances')

""" Metrics SimpleRepository """

# @dataclasses.dataclass
# class Article(object):

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _add_row(self, recipe, report):
#         new_row = pd.Series(index = self.columns)
#         for column, variable in self.required_columns.items():
#             new_row[column] = getattr(recipe, variable)
#         for column in report:
#             new_row[column] = report[column]
#         self.text.loc[len(self.text)] = new_row
#         return self

#     def _check_best(self, recipe):
#         """Checks if the current recipe is better than the current best recipe
#         based upon the primary scoring metric.

#         Args:
#             recipe: an instance of Recipe to be tested versus the current best
#                 recipe stored in the 'best_recipe' attribute.
#         """
#         if not self._exists('best_recipe'):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         elif (self.article.loc[
#                 self.article.index[-1],
#                 utilities.listify(self.metrics)[0]] > self.best_recipe_score):
#             self.best_recipe = recipe
#             self.best_recipe_score = self.article.loc[
#                     self.article.index[-1],
#                     utilities.listify(self.metrics)[0]]
#         return self

#     def _format_step(self, attribute):
#         if getattr(self.recipe, attribute).step in ['none', 'all']:
#             step_column = getattr(self.recipe, attribute).step
#         else:
#             step = getattr(self.recipe, attribute).step
#             parameters = getattr(self.recipe, attribute).parameters
#             step_column = f'{step}, parameters = {parameters}'
#         return step_column

#     def _get_step_name(self, step):
#         """Returns appropriate algorithm to the report attribute."""
#         if step.step in ['none', 'all']:
#             return step.step
#         else:
#             return step.algorithm

#     def print_best(self):
#         """Prints output to the console about the best recipe."""
#         if self.verbose:
#             print('The best test recipe, based upon the',
#                   utilities.listify(self.metrics)[0], 'metric with a score of',
#                   f'{self.best_recipe_score: 4.4f}', 'is:')
#             for step in getattr(self,
#                     self.iterator).best_recipe.steps:
#                 print(step.capitalize(), ':',
#                       getattr(getattr(self, self.iterator).best_recipe,
#                               step).step)
#         return

#     def _set_columns(self, recipe):
#         self.required_columns = {
#             'recipe_number': 'number',
#             'options': 'steps',
#             'seed': 'seed',
#             'validation_set': 'using_val_set'}
#         self.columns = list(self.required_columns.keys())
#         self.columns.extend(recipe.steps)
#         for step in self.steps:
#             if (hasattr(getattr(self, step), 'columns')
#                     and getattr(self, step).name != 'summarize'):
#                 self.columns.extend(getattr(self, step).columns)
#         return self

#     def _start_report(self, recipe):
#         self._set_columns(recipe = recipe)
#         self.text = pd.DataFrame(columns = self.columns)
#         return self

#     """ Public Import/Export Methods """

#     def save(self, report = None):
#         """Exports the review report to disk.

#         Args:
#             review(Review.report): 'report' from an instance of review
#         """
#         self.clerk.save(
#             variable = report,
#             folder = self.clerk.experiment,
#             file_name = self.model_type + '_review',
#             file_format = 'csv',
#             header = True)
#         return

#     """ Core siMpLify Methods """

#     def draft(self) -> None:
#         super().draft()
#         return self

#     def publish(self):
#         super().publish()
#         return self




# DEFAULT_OPTIONS = {
#     'accuracy': ['sklearn.metrics', 'accuracy_score'],
#     'adjusted_mutual_info': ['sklearn.metrics', 'adjusted_mutual_info_score'],
#     'adjusted_rand': ['sklearn.metrics', 'adjusted_rand_score'],
#     'balanced_accuracy': ['sklearn.metrics', 'balanced_accuracy_score'],
#     'brier_score_loss': ['sklearn.metrics', 'brier_score_loss'],
#     'calinski': ['sklearn.metrics', 'calinski_harabasz_score'],
#     'davies': ['sklearn.metrics', 'davies_bouldin_score'],
#     'completeness': ['sklearn.metrics', 'completeness_score'],
#     'contingency_matrix': ['sklearn.metrics', 'cluster.contingency_matrix'],
#     'explained_variance': ['sklearn.metrics', 'explained_variance_score'],
#     'f1': ['sklearn.metrics', 'f1_score'],
#     'f1_weighted': ['sklearn.metrics', 'f1_score'],
#     'fbeta': ['sklearn.metrics', 'fbeta_score'],
#     'fowlkes': ['sklearn.metrics', 'fowlkes_mallows_score'],
#     'hamming': ['sklearn.metrics', 'hamming_loss'],
#     'h_completness': ['sklearn.metrics', 'homogeneity_completeness_v_measure'],
#     'homogeniety': ['sklearn.metrics', 'homogeneity_score'],
#     'jaccard': ['sklearn.metrics', 'jaccard_similarity_score'],
#     'mae': ['sklearn.metrics', 'median_absolute_error'],
#     'matthews_corrcoef': ['sklearn.metrics', 'matthews_corrcoef'],
#     'max_error': ['sklearn.metrics', 'max_error'],
#     'mean_absolute_error': ['sklearn.metrics', 'mean_absolute_error'],
#     'mse': ['sklearn.metrics', 'mean_squared_error'],
#     'msle': ['sklearn.metrics', 'mean_squared_log_error'],
#     'mutual_info': ['sklearn.metrics', 'mutual_info_score'],
#     'neg_log_loss': ['sklearn.metrics', 'log_loss'],
#     'norm_mutual_info': ['sklearn.metrics', 'normalized_mutual_info_score'],
#     'precision': ['sklearn.metrics', 'precision_score'],
#     'precision_weighted': ['sklearn.metrics', 'precision_score'],
#     'r2': ['sklearn.metrics', 'r2_score'],
#     'recall': ['sklearn.metrics', 'recall_score'],
#     'recall_weighted': ['sklearn.metrics', 'recall_score'],
#     'roc_auc': ['sklearn.metrics', 'roc_auc_score'],
#     'silhouette': ['sklearn.metrics', 'silhouette_score'],
#     'v_measure': ['sklearn.metrics', 'v_measure_score'],
#     'zero_one': ['sklearn.metrics', 'zero_one_loss']}


# @dataclasses.dataclass
# class Metrics(Evaluator):
#     """Measures model performance.

#     Args:
#         step(str): name of step.
#         parameters(dict): dictionary of parameters to pass to selected
#             algorithm.
#         name(str): designates the name of the class which is used throughout
#             siMpLify to match methods and settings with this class and
#             identically named subclasses.
#         auto_draft(bool): whether 'publish' method should be called when
#             the class is instanced. This should generally be set to True.

#     """
#     step: object = None
#     parameters: object = None
#     name: str = 'metrics'
#     auto_draft: bool = True
#     options: Dict = dataclasses.field(default_factory = lambda: DEFAULT_OPTIONS)

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     def _build_conditional_parameters(self, parameters, recipe = None):
#         if self.step in 'brier_score_loss':
#             parameters = {
#                 'y_true': getattr(recipe.dataset,
#                                   'y_' + self.data_to_review),
#                 'y_prob': recipe.probabilities[:, 1]}
#         elif self.step in ['roc_auc']:
#              parameters = {
#                  'y_true': getattr(recipe.dataset,
#                                    'y_' + self.data_to_review),
#                  'y_score': recipe.probabilities[:, 1]}
#         return parameters

#     def draft(self) -> None:
#         super().draft()
#         self.negative_options = [
#             'brier_loss_score',
#             'neg_log_loss',
#             'zero_one']
#         self.required = {
#             'fbeta': {'beta': 1},
#             'f1_weighted': {'average': 'weighted'},
#             'precision_weighted': {'average': 'weighted'},
#             'recall_weighted': {'average': 'weighted'}}
#         return self

#     # def edit(self, name, metric, special_type = None,
#     #          special_parameters = None, negative_metric = False):
#     #     """Allows user to manually add a metric to report."""
#     #     self.workers.update({name: metric})
#     #     if special_type in ['probability']:
#     #         self.prob_options.update({name: metric})
#     #     elif special_type in ['scorer']:
#     #         self.score_options.update({name: metric})
#     #     if special_parameters:
#     #        self.special_options.update({name: special_parameters})
#     #     if negative_metric:
#     #        self.negative_options.append[name]
#     #     return self

#     def publish(self, recipe):
#         self.runtime_parameters = {
#             'y_true': getattr(recipe.dataset, 'y_' + self.data_to_review),
#             'y_pred': recipe.predictions}
#         super().implement()
#         return self


    # def _get_gini_probabilities(self, recipe):
    #     """Estimates probabilities of outcomes from fitted model with gini
    #     method.

    #     Args:
    #         recipe(Recipe): instance of Recipe with a fitted model.

    #     Returns:
    #         Series with predictions from fitted model on test data.

    #     """
    #     if hasattr(recipe.model.algorithm, 'predict_proba'):
    #         return recipe.model.algorithm.predict_proba(
    #             getattr(recipe.dataset, 'x_' + self.data_to_review))[1]
    #     else:
    #         if self.verbose:
    #             print('predict_proba method does not exist for',
    #                   recipe.model.step.name)
    #         return None

    # def _get_log_probabilities(self, recipe):
    #     """Estimates log probabilities of outcomes from fitted model.

    #     Args:
    #         recipe(Recipe): instance of Recipe with a fitted model.

    #     Returns:
    #         Series with predictions from fitted model on test data.

    #     """
    #     if hasattr(recipe.model.algorithm, 'predict_log_proba'):
    #         return recipe.model.algorithm.predict_log_proba(
    #             getattr(recipe.dataset, 'x_' + self.data_to_review))[1]
    #     else:
    #         if self.verbose:
    #             print('predict_log_proba method does not exist for',
    #                   recipe.model.step.name)
    #         return None

    # def _get_permutation_importances(self, recipe):
    #     scorer = utilities.listify(self.metrics_steps)[0]
    #     base_score, score_decreases = self.workers[self.step](
    #             score_func = scorer,
    #             x = getattr(recipe.dataset, 'x_' + self.data_to_review),
    #             y = getattr(recipe.dataset, 'y_' + self.data_to_review))
    #     return np.mean(score_decreases, axis = 'columns')

    # def _get_gini_importances(self, recipe):
    #     features = list(getattr(
    #             recipe.dataset, 'x_' + self.data_to_review).columns)
    #     if hasattr(recipe.model.algorithm, 'feature_importances_'):
    #         importances = pd.Series(
    #             data = recipe.model.algorithm.feature_importances_,
    #             index = features)
    #         return importances.sort_values(ascending = False, inplace = True)
    #     else:
    #         return None

    # def _get_shap_importances(self, recipe):
    #     if hasattr(recipe, 'shap_explain.values'):
    #         return np.abs(recipe.shap_explain.values).mean(0)
    #     else:
    #         return None

    # def _get_eli5_importances(self, recipe):
    #     base_score, score_decreases = get_score_importances(score_func, X, y)
    #     feature_importances = np.mean(score_decreases, axis=0)
    #     from eli5 import show_weights
    #     self.permutation_weights = show_weights(
    #             self.permutation_importances,
    #             feature_names = recipe.dataset.columns.keys())
    #     return self