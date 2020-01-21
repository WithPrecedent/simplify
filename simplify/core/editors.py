"""
.. module:: editors
:synopsis: constructs and applies books, chapters, and techniques
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from inspect import isclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.book import Chapter
from simplify.core.repository import Repository
from simplify.core.repository import Sequence
from simplify.core.technique import Technique
from simplify.core.utilities import listify
from simplify.core.validators import DataValidator


@dataclass
class Editor(ABC):
    """Base class for creating and applying Manuscript subclasses.

    Args:
        project ('Project'): a related Project instance.
        worker ('Worker'): the Worker instance for which a subclass should edit
            or apply a Book instance.

    """
    project: 'Project'
    worker: 'Worker'

    def __post_init__(self) -> None:
        """Adds attributes from an 'idea' in 'project'."""
        try:
            self = self.project.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> NotImplementedError:
        """Creates skeleton of a Book instance."""
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no draft method. Use Author instead.']))

    def publish(self, step: str) -> NotImplementedError:
        """Finalizes a Book instance and its Chapters and Techniques."""
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no publish method. Use Publisher instead.']))

    def apply(self, data: object) -> NotImplementedError:
        """Applies Book instance to 'data'.

        Args:
            data (object): data object for a Book instance methods to be
                applied.

        """
        raise NotImplementedError(' '.join(
            [self.__name__, 'has no apply method. Use Scholar instead.']))


@dataclass
class Author(Editor):
    """ Drafts a basic Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker and Book instances in
            'project'.

    """
    project: 'Project'
    worker: str

    """ Private Methods """

    def _draft_book(self) -> 'Book':
        """Creates initial, empty Book instance.

        Returns:
            Book: instance with only 'name' set.

        """
        # Checks to see if a matching Book instance was already created.
        if self.worker in self.project.library:
            return self.project.library[self.worker]
        else:
            # Loads a Book class based upon 'worker' attributes.
            book = self.project[self.worker].load('book')
            # Creates an empty class instance of a Book.
            return book(
                name = self.worker,
                techniques = self.project[self.worker].techniques)

    def _draft_steps(self) -> List[str]:
        """Creates a list of 'steps' from 'idea'."""
        # Checks existing 'worker' to see if 'steps' exists.
        if self.project[self.worker].steps:
            return self.project[self.worker].steps
        else:
            try:
                # Attempts to get 'steps' from 'idea'.
                return listify(self.project.idea[self.worker]['_'.join(
                    [self.worker, 'steps'])])
            except KeyError:
                return []

    def _draft_options(self) -> 'Repository':
        """Gets options for creating a Book contents.

        Returns:
            'Repository': with possible options included.

        """
        # Checks existing 'worker' to see if 'options' exists.
        if isclass(self.project[self.worker].options):
            return self.project[self.worker].options(
                project = self.project)
        elif isinstance(self.project[self.worker].options, str):
            loaded = self.project[self.worker].load('options')
            return loaded(project = self.project)
        else:
            return self.project[self.worker].options

    def _draft_techniques(self) -> Dict[str, List[str]]:
        """Creates 'techniques' for a Book.

        Returns:
            Dict[str, List[str]]: possible techniques for each step in a Book
                instance.

        """
        # Checks to see if 'techniques' already exists.
        if self.project[self.worker].techniques:
            return self.projects[self.worker].techniques
        else:
            techniques = {}
            for step in self.project[self.worker].steps:
                try:
                    techniques[step] = listify(self.project.idea
                        [self.worker]['_'.join([step, 'techniques'])])
                except KeyError:
                    techniques[step] = ['none']
            return techniques

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Drafts initial attributes and settings of a Book instance. """
        # Adds 'steps' to 'workers'.
        self.project[self.worker].steps = self._draft_steps()
        # Adds 'techniques' to 'workers'.
        self.project[self.worker].techniques = self._draft_techniques()
        # Adds 'steps' to 'workers'.
        self.project[self.worker].options = self._draft_options()
        # Adds 'Book' to 'library'.
        self.project.library[self.worker] = self._draft_book()
        return self


@dataclass
class Publisher(Editor):
    """Finalizes Book instances.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker and Book instances in
            'project'.

    """
    project: 'Project'
    worker: str

    def __post_init__(self) -> None:
        """Creates 'Contributor' and 'Researcher' instances."""
        self.contributor = Contributor(
            project = self.project,
            worker = self.worker)
        self.researcher = Researcher(
            project = self.project,
            worker = self.worker)
        return self

    """ Private Methods """

    def _publish_techniques(self, book: 'Book') -> 'Book':
        """Publishes instanced 'techniques' for a Book instance.

        Args:
            book ('Book'): instance for 'techniques' to be added.

        Returns:
            Book: instance, with 'techniques' added.

        """
        drafted_techniques = self.project[self.worker].techniques
        book.techniques = {}
        for step, techniques in drafted_techniques.items():
            for technique in techniques:
                if step not in book.techniques:
                    book.techniques[step] = {}
                book.techniques[step][technique] = self.researcher.publish(
                    step = step,
                    technique = technique)
        return book

    def _publish_chapters(self, book: 'Book') -> 'Book':
        """Publishes instanced 'chapters' for a Book instance.

        Args:
            book ('Book'): instance for 'chapters' to be added.

        Returns:
            Book: instance, with 'chapters' added.

        """
        # Gets list of steps to pair with techniques.
        drafted_steps = self.project.workers[self.worker].steps
        # Creates a list of lists of possible techniques.
        possible = list(self.project[self.worker].techniques.values())
        # Converts 'possible' to a list of the Cartesian product.
        plans = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of step techniques.
        for i, techniques in enumerate(plans):
            book.add_chapters(
                chapters = self.contributor.publish(
                    book = book,
                    number = i,
                    techniques = dict(zip(drafted_steps, techniques))))
        return book

    """ Core siMpLify Methods """

    def publish(self) -> None:
        """Finalizes Book instance, making all changes before application."""
        book = self.project[self.worker]
        # Creates 'chapters' for 'book'.
        self.project[self.worker] = self._publish_chapters(book = book)
        # Creates an 'options' attribute to allow publishing of 'techniques'.
        self.project[self.worker]  = self._publish_techniques(book = book)
        return self


@dataclass
class Researcher(Editor):
    """Creates Technique instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker and Book instances in
            'project'.

    """
    project: 'Project'
    worker: str

    def __post_init__(self) -> None:
        # Declares possible 'parameter_types'.
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            # 'search',
            'runtime',]
        super().__post_init__()
        return self

    """ Private Methods """

    def _publish_parameters(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Creates 'parameters' for a 'Technique' using 'outline'.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        # Iterates through types of 'parameter_types'.
        for parameter_type in self.parameter_types:
            if parameter_type in outline:
                technique = getattr(self, '_'.join(
                    ['_publish', parameter_type]))(
                        outline = outline,
                        technique = technique)
        return technique

    def _publish_idea(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Acquires parameters from Idea instance, if no parameters exist.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(self.project.idea['_'.join(
                [outline.name, 'parameters'])])
        except KeyError:
            try:
                technique.parameters.update(
                    self.project.idea.configuration['_'.join(
                        [technique.name, 'parameters'])])
            except AttributeError:
                pass
        return technique

    def _publish_selected(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique'.

        If 'outline.selected' is True, the keys from 'outline.defaults' are used
        to select the final returned parameters.

        If 'outline.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        if outline.selected:
            if isinstance(outline.selected, list):
                parameters_to_use = outline.selected
            else:
                parameters_to_use = list(outline.default.keys())
            new_parameters = {}
            for key, value in technique.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            technique.parameters = new_parameters
        return technique

    def _publish_required(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(outline.required)
        except TypeError:
            pass
        return technique

    def _publish_search(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        technique.space = {}
        new_parameters = {}
        for parameter, values in technique.parameters.items():
            if isinstance(values, list):
                if any(isinstance(i, float) for i in values):
                    technique.space.update(
                        {parameter: uniform(values[0], values[1])})
                elif any(isinstance(i, int) for i in values):
                    technique.space.update(
                        {parameter: randint(values[0], values[1])})
            else:
                new_parameters.update({parameter: values})
        technique.parameters = new_parameters
        return technique

    def _publish_runtime(self,
            outline: 'TechniqueDefinition',
            technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            outline ('TechniqueDefinition'): instructions for creating a
                Technique instance.
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            for key, value in outline.runtime.items():
                try:
                    technique.parameters.update(
                        {key: getattr(self.project, value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return technique

    """ Core siMpLify Methods """

    def publish(self, step: str, technique: str) -> 'Technique':
        """Creates parameters and algorithms for a 'Technique' instance.

        Args:
            step (str): name of step in a 'Book' instance iterable.
            technique (str): name of the specific technique to use for a 'step'
                 in a 'Book' instance iterable.

        Returns:
            'Technique': instance with 'algorithm', 'parameters', and
                'data_dependent' attributes added.

        """
        if technique in ['none']:
            return None
        else:
            # Gets appropriate TechniqueDefinition and creates an instance.
            outline = self.project.workers[self.worker].options[step][technique]
            # outline = outline.load('algorithm')
            # outline = outline(project = self)
            # Creates a Technique instance and add attributes to it.
            technique = Technique(name = step, technique = technique)
            technique.algorithm = outline.load('algorithm')
            technique.data_dependent = outline.data_dependent
            technique.parameters = self._publish_parameters(
                outline = outline,
                technique = technique)
        return technique


@dataclass
class Contributor(Editor):
    """Creates Chapter instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        worker (str): name of the key to the Worker and Book instances in
            'project'.

    """
    project: 'Project'
    worker: str

    """ Core siMpLify Methods """

    def publish(self,
            book: 'Book',
            number: int,
            techniques: Dict[str, str]) -> 'Book':
        """Creates 'chapters' for 'book'.

        Args:
            book ('Book'): instance for chapters to be added.

        Returns:
            'Book' with chapters added.

        """
        # Creates a Chapter instance and add attributes to it.
        chapter = Chapter(
            name = '_'.join([*list(techniques.values())]),
            number = number,
            book = book,
            techniques = techniques,
            returns_data = book.returns_data)
        # for step, technique in chapter.techniques.items():
        #     chapter.techniques[step] = book.techniques[step][technique]
        return chapter



# @dataclass
# class Book(Repository):
#     """Stores and iterates Chapters.

#     Args:
#         project ('Project'): current associated project.

#     Args:
#         project ('Project'): associated Project instance.
#         options (Optional[Dict[str, 'Worker']]): Repository instance or
#             a Repository-compatible dictionary. Defaults to an empty
#             dictionary.
#         steps (Optional[Union[List[str], str]]): steps of key(s) to iterate in
#             'options'. Also, if not reset by the user, 'steps' is used if the
#             'default' property is accessed. Defaults to an empty list.

#     """
#     project: 'Project' = None
#     options: Optional[Dict[str, 'Worker']] = field(default_factory = dict)
#     steps: Optional[Union['SimpleSequence', List[str], str]] = field(
#         default_factory = list)
#     name: Optional[str] = None
#     chapter_type: Optional['Chapter'] = None
#     iterable: Optional[str] = 'chapters'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'book'

#     def __post_init__(self) -> None:
#         """Initializes class instance attributes."""
#         # Sets default 'name' attribute if none exists.
#         if self.worker is None:
#             self.worker = self.__class__.__name__.lower()
#         # Calls parent method for initialization.
#         super().__post_init__()
#         return self

#     """ Core SiMpLify Methods """

#     def apply(self,
#             options: Optional[Union[List[str], Dict[str, Any], str]] = None,
#             data: Optional[Union['Ingredients', 'Book']] = None,
#             **kwargs) -> Union['Ingredients', 'Book']:
#         """Calls 'apply' method for published option matching 'step'.

#         Args:
#             options (Optional[Union[List[str], Dict[str, Any], str]]): ordered
#                 options to be applied. If none are passed, the 'published' keys
#                 are used. Defaults to None
#             data (Optional[Union['Ingredients', 'Book']]): a siMpLify object for
#                 the corresponding 'options' to apply. Defaults to None.
#             kwargs: any additional parameters to pass to the options' 'apply'
#                 method.

#         Returns:
#             Union['Ingredients', 'Book'] is returned if data is passed;
#                 otherwise nothing is returned.

#         """
#         if isinstance(options, dict):
#             options = list(options.keys())
#         elif options is None:
#             options = self.default
#         self._change_active(new_active = 'applied')
#         for option in options:
#             if data is None:
#                 getattr(self, self.active)[option].apply(**kwargs)
#             else:
#                 data = getattr(self, self.active)[option].apply(
#                     data = data,
#                     **kwargs)
#             getattr(self, self.active)[option] = getattr(
#                 self, self.active)[option]
#         if data is None:
#             return self
#         else:
#             return data


# @dataclass
# class Chapter(Repository):
#     """Iterator for a siMpLify process.

#     Args:
#         book ('Book'): current associated Book
#         metadata (Optional[Dict[str, Any]], optional): any metadata about the
#             Chapter. Unless a subclass replaces it, 'number' is automatically a
#             key created for 'metadata' to allow for better recordkeeping.
#             Defaults to an empty dictionary.

#     """
#     book: 'Book' = None
#     name: Optional[str] = None
#     iterable: Optional[str] = 'book.steps'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'chapter'

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _apply_extra_processing(self) -> None:
#         """Extra actions to take."""
#         return self

#     """ Core siMpLify Methods """

#     def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
#         """Applies stored 'options' to passed 'data'.

#         Args:
#             data (Optional[Union['Ingredients', 'Manuscript']]): a
#                 siMpLify object for the corresponding 'step' to apply. Defaults
#                 to None.
#             kwargs: any additional parameters to pass to the step's 'apply'
#                 method.

#         """
#         if data is not None:
#             self.ingredients = data
#         for step in getattr(self, self.iterable):
#             self.book[step].apply(data = self.ingredients, **kwargs)
#             self._apply_extra_processing()
#         return self