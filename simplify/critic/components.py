"""
critic.components:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from types import ModuleType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify
from . import base



@dataclasses.dataclass
class Explainer(Evaluator):
    """Base class for explaining model performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None

    def __post_init__(self) -> None:
        """Creates initial attributes."""
        self.draft()
        return self

    """ Private Methods """



    """ Core siMpLify Methods """

    def draft(self, recipe: 'Recipe') -> None:
        """Subclasses can provide their own algorithms for setup."""
        self.model_type = self.idea['analyst']['model_type']
        self.estimator = self._get_estimator(recipe = recipe)
        self.algorithm = self._get_algorithm(estimator = self.estimator)
        self.data_attribute = self._get_data(recipe = recipe)
        return self

    def apply(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        """Completes 'review' from data in 'chapter'.

        Args:
            recipe ('Recipe'): a completed 'Recipe' from a 'Cookbook' instance.
            review ('Review'): an instance to complete based upon the
                performance of 'recipe'.

        Returns:
            'Review': with assessment of 'recipe' performance.

        """
        for step in review.steps:
            try:
                review = getattr(self, '_'.join(['_apply', step]))(
                    recipe = recipe,
                    review = review)
            except AttributeError:
                pass
        return review


@dataclasses.dataclass
class Eli5Explain(object):
    """Explains fit model with eli5 package.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    step: object = None
    parameters: object = None
    name: str = 'eli5'
    auto_draft: bool = True

    """ Core siMpLify Methods """

    def draft(self) -> None:
        super().draft()
        self.models = {
            'baseline': 'none',
            'catboost': 'specific',
            'decision_tree': 'specific',
            'lasso': 'specific',
            'lasso_lars': 'specific',
            'light_gbm': 'specific',
            'logit': 'specific',
            'ols': 'specific',
            'random_forest': 'specific',
            'ridge': 'specific',
            'svm_linear': 'specific',
            'tensor_flow': 'permutation',
            'torch': 'permutation',
            'xgboost': 'specific'}
        return self

    def publish(self, recipe):
        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)
        from eli5 import show_weights
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.dataset.columns.keys())
        return self


@dataclasses.dataclass
class ShapExplain(object):
    """Explains fit model with shap package.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """
    step: object = None
    parameters: object = None
    name: str = 'shap_explanation'
    auto_draft: bool = True

    """ Private Methods """

    def _set_method(self, recipe):
        if self.step in self.models:
            self.method = self.workers[self.models[self.step]]
        else:
            self.method = self.workers['kernel']
        self.evaluator = self.method(
            model = recipe.model.algorithm,
            data = getattr(recipe.dataset, 'x_' + self.data_to_review))
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self.models = {
            'baseline': 'none',
            'catboost': 'tree',
            'decision_tree': 'tree',
            'lasso': 'linear',
            'lasso_lars': 'linear',
            'light_gbm': 'tree',
            'logit': 'linear',
            'ols': 'linear',
            'random_forest': 'tree',
            'ridge': 'linear',
            'svm_linear': 'linear',
            'tensor_flow': 'deep',
            'torch': 'deep',
            'xgboost': 'tree'}
        return self

    def publish(self, recipe):
        """Applies shap evaluator to data based upon type of model used."""
        if self.step != 'none':
            self._set_method()
            setattr(recipe, self.name + '_values', self.evaluator.shap_values(
                    getattr(recipe.dataset, 'x_' + self.data_to_review)))
            if not hasattr(recipe, self.name):
                setattr(recipe, self.name, [])
            getattr(recipe, self.name).append(self.name)
            if self.method == 'tree':
                setattr(recipe, self.name + '_interactions',
                        self.evaluator.shap_interaction_values(
                                getattr(recipe.dataset,
                                        'x_' + self.data_to_review)))
            getattr(recipe, self.name).append(
                    getattr(self, self.name + '_interactions'))
        return getattr(recipe, self.name)


@dataclasses.dataclass
class SkaterExplain(object):
    """Explains fit model with skater package.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): designates the name of the class which is used throughout
            siMpLify to match methods and settings with this class and
            identically named subclasses.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.

    """

    step: object = None
    parameters: object = None
    name: str = 'skater'
    auto_draft: bool = True

@dataclasses.dataclass
class SklearnExplain(Explainer):
    """Explains model performance with the sklearn package.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'sklearn')

    """ Private Methods """

    def _apply_explain(self, recipe: 'Recipe', review: 'Review') -> 'Review':

        return review

    def _apply_predict(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        """Makes predictions based upon sklearn package.

        Args:
            recipe ('Recipe'): a completed 'Recipe' from a 'Cookbook' instance.
            review ('Review'): an instance to complete based upon the
                performance of 'recipe'.

        Returns:
            'Review': with assessment of 'recipe' performance.

        """
        try:
            review.predictions[self.name] = self.estimator.predict(
                recipe.data.x_test)
        except AttributeError:
            pass
        try:
            review.predictions['_'.join([self.name, 'probabilities'])] = (
                self.estimator.predict_proba(recipe.data.x_test))
        except AttributeError:
            pass
        try:
            review.predictions['_'.join([self.name, 'log_probabilities'])] = (
                self.estimator.predict_log_proba(recipe.data.x_test))
        except AttributeError:
            pass
        return review

    def _apply_rank(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_measure(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_report(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self


@dataclasses.dataclass
class Eli5Explain(Explainer):
    """Explains model performance with the ELI5 package.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'eli5')


    """ Private Methods """

    def _apply_explain(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_predict(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_rank(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_measure(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_report(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self.options = {
            'permutation' : Evaluator(
                name = 'permutation_importance',
                module = 'eli5.sklearn',
                algorithm = 'PermutationImportance',
                runtime = {'random_state': 'seed', 'estimator': 'estimator'}),
            'kernel' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'KernelExplainer'),
            'linear' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'LinearExplainer'),
            'tree' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'TreeExplainer')}
        self.algorithm_types = {
            'baseline': 'none',
            'catboost': 'specific',
            'decision_tree': 'specific',
            'lasso': 'specific',
            'lasso_lars': 'specific',
            'light_gbm': 'specific',
            'logit': 'specific',
            'ols': 'specific',
            'random_forest': 'specific',
            'ridge': 'specific',
            'svm_linear': 'specific',
            'tensor_flow': 'permutation',
            'torch': 'permutation',
            'xgboost': 'specific'}
        return self

    def apply(self, data: 'Chapter') -> 'Chapter':
        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis = 0)
        self.permutation_weights = show_weights(
                self.permutation_importances,
                feature_names = recipe.dataset.columns.keys())
        return data


@dataclasses.dataclass
class ShapExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    idea: Optional[core.Idea] = None
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'shap')

    """ Private Methods """

    def _apply_explain(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_predict(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_rank(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_measure(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_report(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _set_algorithm(self, data: 'Chapter') -> object:
        try:
            algorithm = self.options[self.algorithm_types[model.name]]
        except KeyError:
            algorithm = self.options['kernel']
        return algorithm.load('algorithm')

    def _apply_to_chapter(self, chapter: 'Chapter') -> 'Chapter':
        print('test algo', self.algorithm)
        print('test model', self.model.algorithm)
        self.algorithm = self.algorithm(
            model = self.model.algorithm,
            data = getattr(chapter.data, '_'.join(
                ['x', self.idea['critic']['data_to_review']])))
        chapter.explanations['shap_values'] = self.algorithm.shap_values(
            getattr(chapter.data, '_'.join(
                ['x', self.idea['critic']['data_to_review']])))
        if self.algorithm_types[self.model] in ['tree']:
            chapter.explanations['shap_interactions'] = (
                self.algorithm.shap_interaction_values(
                    getattr(chapter.data, '_'.join(
                        ['x', self.idea['critic']['data_to_review']]))))
        import shap
        shap.initjs()
        shap.force_plot(self.algorithm.expected_value, shap_values[0,:], X.iloc[0,:])
        return chapter

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self.options = {
            'deep' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'DeepExplainer'),
            'kernel' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'KernelExplainer'),
            'linear' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'LinearExplainer'),
            'tree' : Evaluator(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'TreeExplainer')}
        self.algorithm_types = {
            'baseline': 'none',
            'catboost': 'tree',
            'decision_tree': 'tree',
            'lasso': 'linear',
            'lasso_lars': 'linear',
            'light_gbm': 'tree',
            'logit': 'linear',
            'ols': 'linear',
            'random_forest': 'tree',
            'ridge': 'linear',
            'svm_linear': 'linear',
            'tensor_flow': 'deep',
            'torch': 'deep',
            'xgboost': 'tree'}
        return self

    def apply(self, data: 'Chapter') -> 'Chapter':
        try:
            self.model = self._get_estimator(chapter = data)
            self.algorithm = self.options[self.algorithm_types[self.model.name]]
        except KeyError:
            self.algorithm = options['kernel']
        self.algorithm = self.algorithm.load('algorithm')
        self._apply_to_chapter(chapter = data)
        return data

@dataclasses.dataclass
class SkaterExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (Optional[Idea]): an instance with project settings.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'skater')
    idea: Optional[core.Idea] = None


    """ Private Methods """

    def _apply_explain(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_predict(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_rank(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_measure(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review

    def _apply_report(self, recipe: 'Recipe', review: 'Review') -> 'Review':
        return review
    
    
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

# def _get_brier_score_loss_parameters(self, parameters, recipe = None):
#     if self.step in 'brier_score_loss':
#         parameters = {
#             'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#             'y_prob': recipe.probabilities[:, 1]}
#     elif self.step in ['roc_auc']:
#             parameters = {
#                 'y_true': getattr(recipe.dataset,
#                                 'y_' + self.data_to_review),
#                 'y_score': recipe.probabilities[:, 1]}
#     return parameters

# metrics_accuracy = Technique(
#     name = 'accuracy',
#     module = 'sklearn.metrics',
#     algorithm = 'accuracy_score')
# metrics_adjusted_mutual_info = Technique(
#     name = 'adjusted_mutual_info_score',
#     module = 'sklearn.metrics',
#     algorithm = 'adjusted_mutual_info')
# metrics_adjusted_rand = Technique(
#     name = 'adjusted_rand',
#     module = 'sklearn.metrics',
#     algorithm = 'adjusted_rand_score')
# metrics_balanced_accuracy = Technique(
#     name = 'balanced_accuracy',
#     module = 'sklearn.metrics',
#     algorithm = 'balanced_accuracy_score')
# metrics_brier_score_loss = Technique(
#     name = 'brier_score_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'brier_score_loss')
# metrics_calinski = Technique(
#     name = 'calinski_harabasz',
#     module = 'sklearn.metrics',
#     algorithm = 'calinski_harabasz_score')
# metrics_davies = Technique(
#     name = 'davies_bouldin',
#     module = 'sklearn.metrics',
#     algorithm = 'davies_bouldin_score')
# metrics_completeness = Technique(
#     name = 'completeness',
#     module = 'sklearn.metrics',
#     algorithm = 'completeness_score')
# metrics_contingency_matrix = Technique(
#     name = 'contingency_matrix',
#     module = 'sklearn.metrics',
#     algorithm = 'cluster.contingency_matrix')
# metrics_explained_variance = Technique(
#     name = 'explained_variance',
#     module = 'sklearn.metrics',
#     algorithm = 'explained_variance_score')
# metrics_f1 = Technique(
#     name = 'f1',
#     module = 'sklearn.metrics',
#     algorithm = 'f1_score')
# metrics_f1_weighted = Technique(
#     name = 'f1_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'f1_score',
#     required = {'average': 'weighted'})
# metrics_fbeta = Technique(
#     name = 'fbeta',
#     module = 'sklearn.metrics',
#     algorithm = 'fbeta_score',
#     required = {'beta': 1})
# metrics_fowlkes = Technique(
#     name = 'fowlkes_mallows',
#     module = 'sklearn.metrics',
#     algorithm = 'fowlkes_mallows_score')
# metrics_hamming = Technique(
#     name = 'hamming_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'hamming_loss')
# metrics_h_completness = Technique(
#     name = 'homogeneity_completeness',
#     module = 'sklearn.metrics',
#     algorithm = 'homogeneity_completeness_v_measure')
# metrics_homogeniety = Technique(
#     name = 'homogeneity',
#     module = 'sklearn.metrics',
#     algorithm = 'homogeneity_score')
# metrics_jaccard = Technique(
#     name = 'jaccard_similarity',
#     module = 'sklearn.metrics',
#     algorithm = 'jaccard_similarity_score')
# metrics_mae = Technique(
#     name = 'median_absolute_error',
#     module = 'sklearn.metrics',
#     algorithm = 'median_absolute_error')
# metrics_matthews_corrcoef = Technique(
#     name = 'matthews_correlation_coefficient',
#     module = 'sklearn.metrics',
#     algorithm = 'matthews_corrcoef')
# metrics_max_error = Technique(
#     name = 'max_error',
#     module = 'sklearn.metrics',
#     algorithm = 'max_error')
# metrics_mean_absolute_error = Technique(
#     name = 'mean_absolute_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_absolute_error')
# metrics_mse = Technique(
#     name = 'mean_squared_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_squared_error')
# metrics_msle = Technique(
#     name = 'mean_squared_log_error',
#     module = 'sklearn.metrics',
#     algorithm = 'mean_squared_log_error')
# metrics_mutual_info = Technique(
#     name = 'mutual_info_score',
#     module = 'sklearn.metrics',
#     algorithm = 'mutual_info_score')
# metrics_log_loss = Technique(
#     name = 'log_loss',
#     module = 'sklearn.metrics',
#     algorithm = 'log_loss')
# metrics_norm_mutual_info = Technique(
#     name = 'normalized_mutual_info',
#     module = 'sklearn.metrics',
#     algorithm = 'normalized_mutual_info_score')
# metrics_precision = Technique(
#     name = 'precision',
#     module = 'sklearn.metrics',
#     algorithm = 'precision_score')
# metrics_precision_weighted = Technique(
#     name = 'precision_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'precision_score',
#     required = {'average': 'weighted'})
# metrics_r2 = Technique(
#     name = 'r2',
#     module = 'sklearn.metrics',
#     algorithm = 'r2_score')
# metrics_recall = Technique(
#     name = 'recall',
#     module = 'sklearn.metrics',
#     algorithm = 'recall_score')
# metrics_recall_weighted = Technique(
#     name = 'recall_weighted',
#     module = 'sklearn.metrics',
#     algorithm = 'recall_score',
#     required = {'average': 'weighted'})
# metrics_roc_auc = Technique(
#     name = 'roc_auc',
#     module = 'sklearn.metrics',
#     algorithm = 'roc_auc_score')
# metrics_silhouette = Technique(
#     name = 'silhouette',
#     module = 'sklearn.metrics',
#     algorithm = 'silhouette_score')
# metrics_v_measure = Technique(
#     name = 'v_measure',
#     module = 'sklearn.metrics',
#     algorithm = 'v_measure_score')
# metrics_zero_one = Technique(
#     name = 'zero_one',
#     module = 'sklearn.metrics',
#     algorithm = 'zero_one_loss')


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