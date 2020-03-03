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

from simplify.core.library import Book
from simplify.core.library import Chapter
from simplify.core.library import Technique
from simplify.core.manager import Worker
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
        iterable(Optional[str]): name of property to store alternative proxy
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
    iterable: Optional[str] = field(default_factory = lambda: 'reviews')
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
    explanations: Optional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)
    predictions: Optional[Dict[str, pd.Series]] = field(
        default_factory = dict)
    importances: Optional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)
    metrics: Optional[Dict[str, pd.Series]] = field(
        default_factory = dict)
    reports: Optional[Dict[str, pd.DataFrame]] = field(
        default_factory = dict)


@dataclass
class Evaluator(Technique):
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
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): callable object which executes the primary
            method of a class instance. Defaults to None.

    """
    name: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None

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

    def apply(self, recipe: 'Recipe') -> 'Review':
        return self.algorithm.apply(recipe = recipe)


@dataclass
class CriticScholar(Scholar):
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

    """ Private Methods """

    def _get_data(self, data: 'Dataset') -> 'DataBunch':
        """Returns 'data' for model evaluation.

        Args:
            data ('Dataset'): primary instance used by 'project'.

        Returns:
            'DataBunch': with data to use in model evaluation.

        """
        return getattr(data, self.idea['critic']['data_to_review'])

    """ Core siMpLify Methods """

    def apply(self, book: 'Book', data: Union['Dataset', 'Book']) -> 'Book':
        """Applies 'Book' instance in 'project' to 'data' or other stored books.

        Args:
            book ('Book'): instance with stored 'Technique' instances (either
                stored in the 'techniques' or 'chapters' attributes).
            data ([Union['Dataset', 'Book']): a data source with information to
                finalize 'parameters' for each 'Technique' instance in 'book'

        Returns:
            'Book': with 'parameters' for each 'Technique' instance finalized
                and connected to 'algorithm'.

        """
        return super().apply(book = book, data = self._get_data(data = data))


@dataclass
class Evaluators(Repository):
    """A dictonary of Evaluator options for the Analyst subpackage.

    Args:
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    idea: ClassVar['Idea']

    """ Private Methods """

    def create(self) -> None:
        self.contents = {
            'explain': {
                'eli5': Evaluator(
                    name = 'eli5_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'Eli5Explain'),
                'shap': Evaluator(
                    name = 'shap_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'ShapExplain'),
                'skater': Evaluator(
                    name = 'skater_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'SkaterExplain'),
                'sklearn': Evaluator(
                    name = 'sklearn_explain',
                    module = 'simplify.critic.explainers',
                    algorithm = 'SklearnExplain')},
            'predict': {
                'eli5': Evaluator(
                    name = 'eli5_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'Eli5Predict'),
                'shap': Evaluator(
                    name = 'shap_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'ShapPredict'),
                'skater': Evaluator(
                    name = 'skater_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'SkaterPredict'),
                'sklearn': Evaluator(
                    name = 'sklearn_predict',
                    module = 'simplify.critic.predictors',
                    algorithm = 'SklearnPredict')},
            'rank': {
                'eli5': Evaluator(
                    name = 'eli5_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'Eli5Rank'),
                'shap': Evaluator(
                    name = 'shap_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'ShapRank'),
                'skater': Evaluator(
                    name = 'skater_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'SkaterRank'),
                'sklearn': Evaluator(
                    name = 'sklearn_rank',
                    module = 'simplify.critic.rankers',
                    algorithm = 'SklearnRank')},
            'measure': {
                'simplify': Evaluator(
                    name = 'simplify_measure',
                    module = 'simplify.critic.metrics',
                    algorithm = 'SimplifyMeasure'),
                'sklearn': Evaluator(
                    name = 'sklearn_measure',
                    module = 'simplify.critic.metrics',
                    algorithm = 'SklearnMeasure')},
            'report': {
                'simplify': Evaluator(
                    name = 'simplify_report',
                    module = 'simplify.critic.reporters',
                    algorithm = 'SimplifyReport'),
                'sklearn': Evaluator(
                    name = 'sklearn_report',
                    module = 'simplify.critic.reporters',
                    algorithm = 'SklearnReport')}}
        return self


@dataclass
class Critic(Worker):
    """Object construction instructions used by a Project instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to 'Publisher'.
        scholar (Optional[str]): name of Scholar class in 'module' to load.
            Defaults to 'Scholar'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union[str, Dict[str, Any]]]): a dictionary containing
            options for the 'Worker' instance to utilize or a string
            corresponding to a dictionary in 'module' to load. Defaults to an
            empty dictionary.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'books' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        idea (ClassVar['Idea']): shared project configuration settings.

    """
    name: Optional[str] = field(default_factory = lambda: 'critic')
    module: Optional[str] = field(
        default_factory = lambda: 'simplify.critic.critic')
    book: Optional[str] = field(default_factory = lambda: 'Anthology')
    chapter: Optional[str] = field(default_factory = lambda: 'Review')
    technique: Optional[str] = field(default_factory = lambda: 'Evaluator')
    scholar: Optional[str] = field(default_factory = lambda: 'CriticScholar')
    options: Optional[str] = field(default_factory = lambda: 'Evaluators')
    data: Optional[str] = field(default_factory = lambda: 'analyst')
    idea: ClassVar['Idea']

    """ Core siMpLify Methods """

    def outline(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        catalog = {}
        steps = self._get_settings(
            section = self.name,
            prefix = self.name,
            suffix = 'steps')
        for step in steps:
            techniques = self._get_settings(
                section = self.name,
                prefix = self.name,
                suffix = 'techniques')
            catalog[step] = []
            for technique in techniques:
                if technique in self.options:
                    catalog[step].append(technique)
        return catalog