"""
.. module:: explainers
:synopsis: algorithms for explaining "black-box" models
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from dataclasses.dataclasses import dataclasses.field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.base import SimpleSettings
from simplify.critic.critic import Evaluator
from simplify.critic.critic import Review


@dataclasses.dataclass
class Explainer(Evaluator):
    """Base class for explaining model performance.

    Args:
        idea (Optional['Idea']): an instance with project settings.

    """
    idea: Optional['Idea'] = None

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
class SklearnExplain(Explainer):
    """Explains model performance with the sklearn package.

    Args:
        idea (Optional['Idea']): an instance with project settings.

    """
    idea: Optional['Idea'] = None
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
        idea (Optional['Idea']): an instance with project settings.

    """
    idea: Optional['Idea'] = None
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
        idea (Optional['Idea']): an instance with project settings.

    """
    idea: Optional['Idea'] = None
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
        idea (Optional['Idea']): an instance with project settings.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'skater')
    idea: Optional['Idea'] = None


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