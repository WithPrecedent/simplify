"""
.. module:: explainers
:synopsis: algorithms for explaining "black-box" models
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.base import SimpleSettings
from simplify.critic.critic import CriticTechnique


@dataclass
class Explainer(SimpleSettings, CriticTechnique):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    def __post_init__(self) -> None:
        """Creates initial attributes."""
        self.draft()
        return self

    """ Private Methods """

    def _get_estimator(self, chapter: 'Chapter') -> 'Technique':
        """Gets 'model' 'Technique' from a list of 'steps' in 'chapter'.

        Args:
            chapter ('Chapter'): instance with 'model' step.

        Returns:
            'Technique': with a 'step' of 'model'.

        """
        for technique in chapter.techniques:
            if technique.step in ['model']:
                return technique
                break
            else:
                pass

    def _get_algorithm(self, estimator: object) -> object:
        algorithm = self.options[self.algorithm_types[estimator.name]]
        return algorithm.load('algorithm')

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Subclasses can provide their own algorithms for 1-time setup."""
        return self

    def apply(self, data: 'Chapter') -> 'Chapter':
        estimator = self._get_estimator(chapter = data)
        algorithm = self._get_algorithm(estimator = estimator)
        self._apply_to_chapter(
            chapter = data,
            estimator = estimator,
            algorithm = algorithm)
        return data


@dataclass
class Eli5Explain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self.options = {
            'permutation' : CriticTechnique(
                name = 'permutation_importance',
                module = 'eli5.sklearn',
                algorithm = 'PermutationImportance',
                runtime = {'random_state': 'seed', 'estimator': 'estimator'}),
            'kernel' : CriticTechnique(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'KernelExplainer'),
            'linear' : CriticTechnique(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'LinearExplainer'),
            'tree' : CriticTechnique(
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


@dataclass
class ShapExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

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
            'deep' : CriticTechnique(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'DeepExplainer'),
            'kernel' : CriticTechnique(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'KernelExplainer'),
            'linear' : CriticTechnique(
                name = 'shap_explanation',
                module = 'shap',
                algorithm = 'LinearExplainer'),
            'tree' : CriticTechnique(
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

@dataclass
class SkaterExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']