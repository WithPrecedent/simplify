"""
.. module:: critic
:synopsis: model evaluation made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import pandas as pd

from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Technique
from simplify.core.repository import Repository
from simplify.core.scholar import Scholar


@dataclass
class Anthology(Book):
    """Applies techniques to 'Cookbook' instances to assess performance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'anthology'
        chapters (Optional[List['Chapter']]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty list.
        _iterable(Optional[str]): name of property to store alternative proxy
            to 'reviews'.

    """
    name: Optional[str] = field(default_factory = lambda: 'anthology')
    chapters: Optional[List['Chapter']] = field(default_factory = list)
    _iterable: Optional[str] = field(default_factory = lambda: 'reviews')


@dataclass
class CriticTechnique(Technique):
    """

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str]
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """ Core siMpLify Methods """

    def apply(self, data: 'Cookbook') -> 'Cookbook':

        return data


@dataclass
class Critic(Scholar):
    """Applies an 'Anthology' instance to an applied 'Cookbook'.

    Args:
        idea (ClassVar['Idea']): an 'Idea' instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def _apply_chapter(self,
            chapter: 'Chapter',
            data: 'Cookbook') -> 'Chapter':
        """Iterates a single chapter and applies 'steps' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'steps' to apply to 'data'.
            data ('Cookbook'): object for 'chapter' 'steps' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        for step in chapter.steps:
            data = step.apply(data = data)
        setattr(chapter, 'data', data)
        return chapter


@dataclass
class Evaluators(Repository):
    """A dictonary of CriticTechnique options for the Analyst subpackage.

    Args:
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def create(self) -> None:
        self.contents = {
            'explain': {
                'eli5': CriticTechnique(
                    name = 'eli5_explain',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'Eli5Explain'),
                'shap': CriticTechnique(
                    name = 'shap_explain',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'ShapExplain'),
                'skater': CriticTechnique(
                    name = 'skater_explain',
                    module = 'skater',
                    algorithm = '')},
            'predict': {
                'gini': CriticTechnique(
                    name = 'gini_predict',
                    module = None,
                    algorithm = 'predict'),
                'shap': CriticTechnique(
                    name = 'shap_predict',
                    module = 'shap',
                    algorithm = '')},
            'estimate': {
                'gini': CriticTechnique(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_proba'),
                'log': CriticTechnique(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_log_proba'),
                'shap': CriticTechnique(
                    name = 'shap_probabilities',
                    module = 'shap',
                    algorithm = '')},
            'rank': {
                'permutation': CriticTechnique(
                    name = 'permutation_importances',
                    module = None,
                    algorithm = ''),
                'gini': CriticTechnique(
                    name = 'gini_importances',
                    module = None,
                    algorithm = 'feature_importances_'),
                'eli5': CriticTechnique(
                    name = 'eli5_importances',
                    module = 'eli5',
                    algorithm = ''),
                'shap': CriticTechnique(
                    name = 'shap_importances',
                    module = 'shap',
                    algorithm = '')},
            'measure': {
                'simplify': CriticTechnique(
                    name = 'simplify_metrics',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'compute_metrics'),
                'pandas': CriticTechnique(
                    name = 'pandas_describe',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'pandas_describe')},
            'report': {
                'simplify': CriticTechnique(
                    name = 'simplify_report',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'simplify_report'),
                'confusion': CriticTechnique(
                    name = 'confusion_matrix',
                    module = 'sklearn.metrics',
                    algorithm = 'confusion_matrix'),
                'classification': CriticTechnique(
                    name = 'classification_report',
                    module = 'sklearn.metrics',
                    algorithm = 'classification_report')}}
        return self
