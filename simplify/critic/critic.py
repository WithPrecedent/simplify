"""
.. module:: critic
:synopsis: model evaluation made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
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
from simplify.core.scholar import Finisher
from simplify.core.scholar import Parallelizer
from simplify.core.scholar import Scholar
from simplify.core.scholar import Specialist


@dataclass
class Anthology(Book):
    """Applies techniques to 'Cookbook' instances to assess performance.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
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
class Review(Chapter):
    """Evaluations for a 'Cookbook' recipe.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        steps (Optional[List[str]]): 
        explanations (Dict[str, pd.DataFrame]): results from any 'Explainer'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        predictions (Dict[str, pd.Series]): results from any 'Predictor'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        estimations (Dict[str, pd.Series]): results from any 'Estimator'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        importances (Dict[str, pd.DataFrame]): results from any 'Ranker'
            methods applied to the data analysis. Defaults to an empty
            dictionary.
        reports (Dict[str, pd.DataFrame]): results from any 'Reporter'
            methods applied to the data analysis. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    steps: Optional[List[str]] = field(default_factory = list)
    explanations: ptional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)
    predictions: ptional[Dict[str, pd.Series]] = field(
        default_factory = dict)
    estimations: ptional[Dict[str, pd.Series]] = field(
        default_factory = dict)
    importances: ptional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)
    reports: ptional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)


@dataclass
class CriticTechnique(Technique):
    """Base method wrapper for applying algorithms to data.

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
        algorithm (Optional[object]): callable object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

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

    def apply(self, data: 'Chapter') -> 'Chapter':
        return self.algorithm.apply(chapter = data)


@dataclass
class Critic(Scholar):
    """Applies an 'Anthology' instance to an applied 'Cookbook'.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (ClassVar['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: ClassVar['Idea']

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        # Creates 'Finisher' instance to finalize 'Technique' instances.
        self.finisher = CriticFinisher(worker = self.worker)
        # Creates 'Specialist' instance to apply 'Technique' instances.
        self.specialist = CriticSpecialist(worker = self.worker)
        # Creates 'Parallelizer' instance to apply 'Chapter' instances, if the
        # option to parallelize has been selected.
        if self.parallelize:
            self.parallelizer = Parallelizer(idea = self.idea)
        return self


@dataclass
class CriticFinisher(Finisher):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (ClassVar['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: ClassVar['Idea']

    """ Private Methods """

    def _add_explain_conditionals(self,
            technique: 'Technique',
            data: 'Dataset') -> 'Technique':
        """Adds any conditional parameters to 'technique'

        Args:
            technique ('Technique'): an instance with 'algorithm' and
                'parameters' not yet combined.
            data ('Dataset'): data object used to derive hyperparameters.

        Returns:
            'Technique': with any applicable parameters added.

        """

        return technique


@dataclass
class CriticSpecialist(Specialist):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (ClassVar['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: ClassVar['Idea']


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
                    algorithm = 'SimplifyReport',
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
        
        self.contents['model'] = model_options[
            self.idea['analyst']['model_type']]
        if self.idea['general']['gpu']:
            self.contents['model'].update(
                gpu_options[idea['analyst']['model_type']])
        return self
