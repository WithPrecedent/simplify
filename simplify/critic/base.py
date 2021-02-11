"""
critic.base:
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import simplify


@dataclasses.dataclass
class Critic(base.SimpleManager):
    """Manages a distinct portion of a data science project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout sourdough. For example, if a 
            sourdough instance needs settings from a SimpleSettings
            instance, 'name' should match the appropriate section name in a 
            SimpleSettings instance. Defaults to None. 
        workflow (base.SimpleWorkflow): a workflow of a project subpart derived 
            from 'outline'. Defaults to None.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.
                
    """
    name: str = 'critic'
    workflow: base.SimpleWorkflow = None
    needs: ClassVar[Union[Sequence[str], str]] = ['outline', 'name']


@dataclasses.dataclass
class Anthology(base.SimpleSummary):
    """Collects and stores results of executing a data science project workflow.
    
    Args:
        contents (Mapping[Any, Any]]): stored dictionary. Defaults to an empty 
            dict.
        default (Any): default value to return when the 'get' method is used.
        prefix (str): prefix to use when storing different paths through a 
            workflow. So, for example, a prefix of 'path' will create keys of
            'path_1', 'path_2', etc. Defaults to 'experiment'.
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to 
            a list with 'workflow' and 'data'.          
              
    """
    contents: Mapping[Any, Any] = dataclasses.field(default_factory = dict)
    default: Any = None
    prefix: str = 'experiment'
    needs: ClassVar[Union[Sequence[str], str]] = ['workflow', 'data'] 
    

@dataclasses.dataclass
class SimpleEli5(base.SimpleTechnique):
    
    name: str = 'eli5_explainer'
    module: str = 'simplify.critic'
    algorithm: str = 'Eli5Explain'


@dataclasses.dataclass
class SimpleShap(base.SimpleTechnique):
    
    name: str = 'shap_explainer'
    module: str = 'simplify.critic'
    algorithm: str = 'ShapExplain'



# options = sourdough.types.Catalog(contents = {
#     'explain': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'Eli5Explain'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'ShapExplain'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'SkaterExplain'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_explain',
#             module = 'simplify.critic.explainers',
#             algorithm = 'SklearnExplain')},
#     'predict': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'Eli5Predict'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'ShapPredict'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'SkaterPredict'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_predict',
#             module = 'simplify.critic.predictors',
#             algorithm = 'SklearnPredict')},
#     'rank': {
#         'eli5': base.SimpleTechnique(
#             name = 'eli5_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'Eli5Rank'),
#         'shap': base.SimpleTechnique(
#             name = 'shap_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'ShapRank'),
#         'skater': base.SimpleTechnique(
#             name = 'skater_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'SkaterRank'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_rank',
#             module = 'simplify.critic.rankers',
#             algorithm = 'SklearnRank')},
#     'measure': {
#         'simplify': base.SimpleTechnique(
#             name = 'simplify_measure',
#             module = 'simplify.critic.metrics',
#             algorithm = 'SimplifyMeasure'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_measure',
#             module = 'simplify.critic.metrics',
#             algorithm = 'SklearnMeasure')},
#     'report': {
#         'simplify': base.SimpleTechnique(
#             name = 'simplify_report',
#             module = 'simplify.critic.reporters',
#             algorithm = 'SimplifyReport'),
#         'sklearn': base.SimpleTechnique(
#             name = 'sklearn_report',
#             module = 'simplify.critic.reporters',
#             algorithm = 'SklearnReport')}})


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


@dataclasses.dataclass
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
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'anthology')
    chapters: Optional[List['Review']] = dataclasses.field(default_factory = list)
    iterable: Optional[str] = dataclasses.field(default_factory = lambda: 'reviews')
    steps: Optional[List[Tuple[str, str]]] = dataclasses.field(default_factory = list)
    techniques: Optional[List['Technique']] = dataclasses.field(default_factory = list)


@dataclasses.dataclass
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
    steps: Optional[List[str]] = dataclasses.field(default_factory = list)
    explanations: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)
    predictions: Optional[Dict[str, pd.Series]] = dataclasses.field(
        default_factory = dict)
    importances: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)
    metrics: Optional[Dict[str, pd.Series]] = dataclasses.field(
        default_factory = dict)
    reports: Optional[Dict[str, pd.DataFrame]] = dataclasses.field(
        default_factory = dict)


@dataclasses.dataclass
class Evaluator(Technique):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coord   ination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): process object which executes the primary
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


@dataclasses.dataclass
class CriticScholar(Scholar):
    """Applies an 'Anthology' instance to an applied 'Cookbook'.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

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


@dataclasses.dataclass
class CriticFinisher(Finisher):
    """Finalizes 'Technique' instances with data-dependent parameters.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

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


@dataclasses.dataclass
class CriticSpecialist(Specialist):
    """Base class for applying 'Technique' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (Optional[Idea]): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional[core.Idea] = None

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



@dataclasses.dataclass
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
        import_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'clerk' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        idea (Optional[Idea]): shared project configuration settings.

    """
    name: Optional[str] = dataclasses.field(default_factory = lambda: 'critic')
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.critic.critic')
    book: Optional[str] = dataclasses.field(default_factory = lambda: 'Anthology')
    chapter: Optional[str] = dataclasses.field(default_factory = lambda: 'Review')
    technique: Optional[str] = dataclasses.field(default_factory = lambda: 'Evaluator')
    scholar: Optional[str] = dataclasses.field(default_factory = lambda: 'CriticScholar')
    options: Optional[str] = dataclasses.field(default_factory = lambda: 'Evaluators')
    data: Optional[str] = dataclasses.field(default_factory = lambda: 'analyst')
    idea: Optional[core.Idea] = None

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
    