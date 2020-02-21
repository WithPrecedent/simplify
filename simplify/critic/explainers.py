"""
.. module:: explainers
:synopsis: algorithms for explaining "black-box" models
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.critic.critic import CriticTechnique


@dataclass
class Explainer(object):
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


    def _get_model(self, chapter: 'Chapter') -> 'Technique':
        """Gets 'model' 'Technique' from a list of 'steps' in 'chapter'.

        Args:
            chapter ('Chapter'): instance with 'model' step.

        Returns:
            'Technique': with a 'step' of 'model'.

        """
        for step in chapter.steps:
            if step.step in ['model']:
                return step
                break
            else:
                pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Subclasses can provide their own methods for 1-time setup."""
        return self

    def apply(self, cookbook: 'Cookbook') -> 'Cookbook':
        """Applies shap evaluator to data based upon type of model used.

        Args:
            cookbook ('Cookbook'): an instance that has been fitted and applied
                to data.

        Returns:
            'Cookbook': with shap explanations added.

        """
        self._set_options()
        new_chapters = []
        for chapter in cookbook.chapters:
            new_chapters.append(self._apply_chapter(chapter = chapter))
        cookbook.chpaters = new_chapters
        return cookbook


@dataclass
class Eli5Explain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

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


@dataclass
class ShapExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def _apply_chapter(self, chapter: 'Chapter') -> 'Chapter':
        self.method = self.method(model = self.model, data = chapter.data)
        chapter.explanations['shap_values'] = self.method.shap_values(
            getattr(chapter.data, '_'.join(
                ['x', self.idea['critic']['data_to_review']])))
        if self.method_types[self.model] in ['tree']:
            chapter.explanations['shap_interactions'] = (
                self.method.shap_interaction_values(
                    getattr(chapter.data, '_'.join(
                        ['x', self.idea['critic']['data_to_review']]))))
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
        self.method_types = {
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
        try:
            self.model = self._get_model(chapter = chapter)
            self.method =  self.options[self.method_types[self.model.name]]
        except KeyError:
            self.method = options['kernel']
        return self

@dataclass
class SkaterExplain(Explainer):
    """Base class for explaining model performance.

    Args:
        idea (ClassVar['Idea']): an instance with project settings.

    """
    idea: ClassVar['Idea']