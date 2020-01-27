"""
.. module:: editors
:synopsis: constructs books, chapters, and techniques
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.repository import Repository
from simplify.core.repository import Plan
from simplify.core.technique import Technique
from simplify.core.utilities import listify


@dataclass
class Editor(object):
    """ Drafts a basic Book instance.

    Args:
        project ('Project'): a related Project instance.
        task (str): name of the key to the Task and Book instances in
            'project'.

    """
    project: 'Project'
    task: str

    """ Private Methods """

    def _draft_book(self, book: Union['Book', str]) -> 'Book':
        """Creates initial, empty Book instance.

        Returns:
            Book: instance with only 'name' set.

        """
        if isinstance(book, Book):
            return book
        elif isinstance(book, str):
            return self.project.tasks[self.task].load('book')()
        else:
            raise TypeError('book must be Book or str')

    def _draft_options(self,
            options: Union[
                'Repository', Dict[str, Dict[str, Any]], str]) -> 'Repository':
        """Gets options for creating a Book contents.

        Returns:
            'Repository': with possible options included.

        """
        if isinstance(options, Repository):
                return options
        elif isinstance(options, str):
            return self.project.tasks[self.task].load('options')(
                project = self.project)
        elif isinstance(options, dict):
            return Repository(contents = options, project = self.project)
        else:
            raise TypeError('options must be Repository, dict, or str')

    def _get_all_techniques(self, techniques: 'Repository') -> List[List[str]]:
        """Converts 'techniques' values to a list of lists.

        Args:
            techniques ('Repository'): techniques 'Repository' with all possible
                techniques to be used.

        Return:
            List[List[str]]: all possible techniques for each step.

        """
        possible_techniques = []
        for step in techniques.keys():
            possible_techniques.append(list(techniques[step].keys()))
        return possible_techniques

    def _publish_techniques(self,
            book: 'Book',
            options = 'Repository') -> 'Book':
        """Publishes instanced 'techniques' for a Book instance.

        Args:
            book ('Book'): instance for 'techniques' to be added.
            options ('Repository'): instance containing available options with
                'Technique' instances.

        Returns:
            Book: instance, with 'techniques' added.

        """
        for step, techniques in book.techniques.items():
            new_techniques = []
            for technique in techniques.keys():
                new_techniques.append(self.expert.publish(
                    step = step,
                    technique = technique))
            book.add_techniques(step = step, techniques = new_techniques)
        return book

    def _publish_chapters(self, book: 'Book') -> 'Book':
        """Publishes instanced 'chapters' for a Book instance.

        Args:
            book ('Book'): instance for 'chapters' to be added.

        Returns:
            Book: instance, with 'chapters' added.

        """
        # Gets list of steps to pair with techniques.
        drafted_steps = self.project.tasks[self.task].book.steps
        # Creates a list of lists of possible techniques.
        possible = self._get_all_techniques(
            techniques = self.project.tasks[self.task].book.techniques)
        # Converts 'possible' to a list of the Cartesian product.
        chapters = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of step techniques.
        new_chapters = {}
        for i, techniques in enumerate(chapters):
            book.add_chapters(
                chapters = self.author.publish(
                    book = book,
                    number = i,
                    techniques = dict(zip(drafted_steps, techniques))))
        return book

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Drafts initial attributes and settings of a Book instance. """
        # Instances 'options' and 'book' for 'task'.
        self.project.tasks[self.task].options = self._draft_options(
            options = self.project.tasks[self.task].options)
        book = self._draft_book(book = self.project.tasks[self.task].book)
        # Adds 'techniques' and 'steps' from 'idea' attribute of 'project'.
        self.project.tasks[self.task].book = self.project.idea.apply(
            instance = book)
        return self

    def publish(self) -> None:
        """Finalizes Book instance, making all changes before application."""
        # Creates 'Author' and 'Expert' instances for creating attributes.
        self.author = Author(
            project = self.project,
            task = self.task)
        self.expert = Expert(
            project = self.project,
            task = self.task)
        # Retrieves 'book' from 'project'.
        book = self.project.tasks[self.task].book
        # Creates 'techniques' attribute for 'book'.
        book = self._publish_techniques(
            book = book,
            options = self.project.tasks[self.task].options)
        # Creates 'chapters' for 'book'.
        book = self._publish_chapters(book = book)
        # Stores finalized 'book' in 'library' of 'project'.
        self.project.library[self.task] = book
        return self


@dataclass
class Expert(object):
    """Creates Technique instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        task (str): name of the key to the Task and Book instances in
            'project'.

    """
    project: 'Project'
    task: str

    def __post_init__(self) -> None:
        # Declares possible 'parameter_types'.
        self.parameter_types = [
            'idea',
            'selected',
            'required',
            # 'search',
            'runtime',]
        return self

    """ Private Methods """

    def _publish_outline(self,
            step: str,
            technique: str,
            outline: 'TechniqueOutline') -> 'Technique':
        """Creates Technique instance from 'TechniqueOutline'.

        Args:
            outline ('TechniqueOutline'): instructions for creating a Technique
                instance.
            technique ('Technique'): an instance for attributes to be added to.

        Returns:
            'Technique': instance with an algorithm and parameters added.

        """
        # Creates a Technique instance and add attributes to it.
        technique = Technique(name = step, technique = technique)
        if outline.module and outline.algorithm:
            technique.algorithm = outline.load('algorithm')
        technique.data_dependent = outline.data_dependent
        technique.parameters = self._publish_parameters(
            outline = outline,
            technique = technique)
        return technique

    def _publish_parameters(self,
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Creates 'parameters' for a 'Technique' using 'outline'.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Acquires parameters from Idea instance, if no parameters exist.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique'.

        If 'outline.selected' is True, the keys from 'outline.defaults' are used
        to select the final returned parameters.

        If 'outline.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            outline: 'TechniqueOutline',
            technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            outline ('TechniqueOutline'): instructions for creating a
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
            # Gets appropriate TechniqueOutline and creates an instance.
            outline = self.project.tasks[self.task].options[step][technique]
            if isinstance(outline, (dict, Repository, Plan)):
                techniques = []
                for key, value in outline.items():
                    techniques.append(self._publish_outline(
                        step = step,
                        technique = technique,
                        outline = value))
                return techniques
            else:
                return self._publish_outline(
                        step = step,
                        technique = technique,
                        outline = outline)


@dataclass
class Author(object):
    """Creates Chapter instances for a Book instance.

    Args:
        project ('Project'): a related Project instance.
        task (str): name of the key to the Task and Book instances in
            'project'.

    """
    project: 'Project'
    task: str

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
            alters_data = book.alters_data)
        # for step, technique in chapter.techniques.items():
        #     chapter.techniques[step] = book.techniques[step][technique]
        return chapter