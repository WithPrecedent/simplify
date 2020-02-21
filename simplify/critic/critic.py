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
        steps (Optional[List[Tuple[str, str]]]): tuples of steps and
            techniques.
        techniques (Optional[List['Technique']]): 'Technique' instances to
            apply. In an ordinary project, 'techniques' are not passed to an
            Anthology instance, but are instead created from 'steps' when the
            'publish' method of a 'Project' instance is called. Defaults to
            an empty list.

    """
    name: Optional[str] = field(default_factory = lambda: 'anthology')
    chapters: Optional[List['Review']] = field(default_factory = list)
    _iterable: Optional[str] = field(default_factory = lambda: 'reviews')
    steps: Optional[List[Tuple[str, str]]] = field(default_factory = list)
    techniques: Optional[List['Technique']] = field(default_factory = list)


@dataclass
class Review(object):
    """Evaluations for a 'Cookbook' recipe.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        explanations (Dict[str, pd.DataFrame]): Defaults to None.
        predictions (Dict[str, pd.Series]): Defaults to None.
        estimations (Dict[str, pd.Series]): Defaults to None.
        importances (Dict[str, pd.DataFrame]): Defaults to None.
        reports (Dict[str, pd.DataFrame]): Defaults to None.

    """
    name: Optional[str] = None
    explanations: Dict[str, pd.DataFrame] = None
    predictions: Dict[str, pd.Series] = None
    estimations: Dict[str, pd.Series] = None
    importances: Dict[str, pd.DataFrame] = None
    reports: Dict[str, pd.DataFrame] = None


@dataclass
class CriticTechnique(Technique):
    """

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None
    storage: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """ Core siMpLify Methods """

    def apply(self, recipe: 'Chapter') -> 'Chapterk':
        return self.algorithm.apply(recipe = recipe)


@dataclass
class Critic(Scholar):
    """Applies an 'Anthology' instance to an applied 'Cookbook'.

    Args:
        idea (ClassVar['Idea']): an 'Idea' instance with project settings.

    """
    idea: ClassVar['Idea']

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        super().__post_init__()

        return self

    """ Private Methods """

    def _finalize_techniques(self,
            book: 'Anthology',
            cookbook: 'Cookbook') -> 'Anthology':
        """Finalizes 'Chapter' instances in 'Book'.

        Args:
            book ('Anthology'): instance containing 'chapters' with
                'techniques'.
            data ('Dataset): instance with potential information to use to
                finalize 'parameters' for 'book'.

        Returns:
            'Anthology': with any necessary modofications made.

        """
        for chapter in book.chapters:
            for technique in chapter.techniques:
                # Creates empty dictionaries in 'book' to store 'Critic'
                # evaluations.
                if not hasattr(book, step.storage):
                    setattr(book, step, {})
        return book

    def _apply_technique(self,
            chapter: 'Chapter',
            recipe: 'Chapter') -> 'Chapter':
        """Iterates a single chapter and applies 'techniques' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'techniques' to apply to 'data'.
            recipe ('Chapter'): object for 'chapter' 'techniques' to be applied.

        Return:
            'Chapter': 'Anthology' 'Chapter' instance with changes made.

        """
        for technique in chapter.techniques:
            chapter = technique.apply(recipe = recipe)
        return chapter

    """ Core siMpLify Methods """

    def apply(self,
            worker: str,
            project: 'Project',
            data: 'Dataset',
            **kwargs) -> ('Project', 'Dataset'):
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            worker (str): key to 'Book' instance to apply in 'project'.
            project ('Project): instance with stored 'Book' instances to apply
                or to have other 'Book' instances applied to.
            data (Optional[Union['Dataset', 'Book']]): a data source 'Book'
                instances in 'project' to potentially be applied.
            kwargs: any additional parameters to pass.

        Returns:
            Tuple('Project', 'Data'): instances with any necessary modifications
                made.

        """
        project[worker] = self._finalize_chapters(
            book = project[worker],
            data = data)
        if self.parallelize:
            self.parallelizer.apply_chapters(
                data = project['analyst'],
                method = self._apply_technique)
        else:
            new_chapters = []
            for i, recipe in enumerate(project['analyst'].chapters):
                for chapter in project[worker].chapters:
                    if self.verbose:
                        print('Evaluating recipe', str(i + 1))
                    new_chapters.append(self._apply_technique(
                        chapter = chapter,
                        recipe = recipe))
                project[worker].chapters = new_recipes
        return project, data


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
                    module = 'simplify.critic.explainers',
                    algorithm = 'Eli5Explain',
                    storage = 'explanations'),
                'shap': CriticTechnique(
                    name = 'shap_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'ShapExplain',
                    storage = 'explanations'),
                'skater': CriticTechnique(
                    name = 'skater_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'SkaterExplain',
                    storage = 'explanations')},
            'predict': {
                'gini': CriticTechnique(
                    name = 'gini_predict',
                    module = None,
                    algorithm = 'predict',
                    storage = 'predictions'),
                'shap': CriticTechnique(
                    name = 'shap_predict',
                    module = 'shap',
                    algorithm = '',
                    storage = 'predictions')},
            'estimate': {
                'gini': CriticTechnique(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_proba',
                    storage = 'estimations'),
                'log': CriticTechnique(
                    name = 'gini_probabilities',
                    module = None,
                    algorithm = 'predict_log_proba',
                    storage = 'estimations'),
                'shap': CriticTechnique(
                    name = 'shap_probabilities',
                    module = 'shap',
                    algorithm = '',
                    storage = 'estimations')},
            'rank': {
                'permutation': CriticTechnique(
                    name = 'permutation_importances',
                    module = None,
                    algorithm = '',
                    storage = 'importances'),
                'gini': CriticTechnique(
                    name = 'gini_importances',
                    module = None,
                    algorithm = 'feature_importances_',
                    storage = 'importances'),
                'eli5': CriticTechnique(
                    name = 'eli5_importances',
                    module = 'eli5',
                    algorithm = '',
                    storage = 'importances'),
                'shap': CriticTechnique(
                    name = 'shap_importances',
                    module = 'shap',
                    algorithm = '',
                    storage = 'importances')},
            'measure': {
                'simplify': CriticTechnique(
                    name = 'simplify_metrics',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'compute_metrics',
                    storage = 'measurements'),
                'pandas': CriticTechnique(
                    name = 'pandas_describe',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'pandas_describe',
                    storage = 'measurements')},
            'report': {
                'simplify': CriticTechnique(
                    name = 'simplify_report',
                    module = 'simplify.critic.algorithms',
                    algorithm = 'simplify_report',
                    storage = 'reports'),
                'confusion': CriticTechnique(
                    name = 'confusion_matrix',
                    module = 'sklearn.metrics',
                    algorithm = 'confusion_matrix',
                    storage = 'reports'),
                'classification': CriticTechnique(
                    name = 'classification_report',
                    module = 'sklearn.metrics',
                    algorithm = 'classification_report',
                    storage = 'reports')}}
        return self
