"""
.. module:: publisher
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
class Publisher(object):
    """ Drafts a Book instance with Chapter and Technique instances.

    Args:
        idea ('Idea'): an instance with project settings.
        task ('Task'): instance with information needed to create a Book 
            instance.
        
    """
    idea: 'Idea'
    task: 'Task'

    """ Private Methods """

    def _draft_techniques(self) -> None:
        """Drafts 'raw_techniques' from 'idea'."""
        self.raw_techniques = {}
        try:
            for key, value in self.idea[self.task.name].items():
                if key.endswith('_techniques'):
                    step = key.replace('_techniques', '')
                    if value in [None, 'none', 'None', 'NONE']:
                        self.raw_techniques[step] = 'none'
                    else:
                        self.raw_techniques[step] = listify(value)
        except (KeyError, AttributeError):
            pass
        return self

    def _draft_steps(self) -> None:
        """Drafts 'steps' from 'task' or 'idea'."""
        if self.task.steps:
            self.steps = self.task.steps
        else:
            try:
                self.steps = listify(self.idea[self.task.name]['_'.join(
                    [self.task.name, 'steps'])])
            except (KeyError, AttributeError):
                self.steps = []
        return self
    
    def _draft_options(self) -> None:
        """Creates options for creating a Book contents."""
        if isinstance(self.task.options, Repository):
            self.options = self.task.options
        elif isinstance(self.task.options, str):
            self.options = self.task.load('options')(idea = self.idea)
        elif isinstance(self.task.options, dict):
            self.options = Repository(
                contents = self.task.options, 
                idea = self.idea)
        else:
            raise TypeError('task.options must be Repository, dict, or str')
        return self
    
    def _draft_book(self) -> None:
        """Creates initial, empty Book instance."""
        if isinstance(self.task.book, (str, Book)):
            if isinstance(self.task.book, Book):
                self.book = self.task.book
            else:
                self.book = self.task.load('book')(name = self.task.book)
        else:
            raise TypeError('task.book must be Book or str')
        return self

    def _get_all_techniques(self) -> List[List[str]]:
        """Converts 'techniques' values to a list of lists.

        Return:
            List[List[str]]: all possible techniques for each step.

        """
        possible_techniques = []
        for step, techniques in self.raw_techniques.items():
            possible_techniques.append(listify(techniques))
        return possible_techniques

    def _publish_chapters(self) -> None:
        """Publishes instanced 'chapters' for a Book instance."""
        # Creates 'possible' list of lists of 'techniques'.
        possible = self._get_all_techniques()
        # Creates a list of lists of the Cartesian product of 'possible'.
        chapters = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of techniques.
        for techniques in chapters:
            contents = self.author.publish(
                techniques = dict(zip(self.steps, techniques)))
            techniques = Plan(idea = self.idea,contents = contents)
            self.book.chapters.append(Chapter(techniques = techniques))
        return self

    def _publish_techniques(self) -> None:
        """Publishes instanced 'techniques' for each 'Chapter' instance."""
        new_techniques = {}
        for chapter in self.book:
            for step, technique in chapter.techniques.items():
                new_techniques[step] = self.expert.publish(
                    step = step,
                    technique = technique)
        self.book.chapters.techniques = new_techniques
        return self

    """ Core siMpLify Methods """

    def draft(self) -> 'Task':
        """Drafts initial attributes and settings of a Book instance. """
        # Creates 'raw_techniques', 'steps', 'options, and 'book'.
        self._draft_steps()
        self._draft_raw_techniques()
        self._draft_options()
        self._draft_book()
        return self

    def publish(self) -> 'Book':
        """Finalizes Book instance, making all changes before application."""
        # Creates 'author' which is used to create 'Chapter' instances.
        self.author = Author(
            idea = self.idea, 
            task = self.task)
        # Creates 'expert' which is used to create 'Technique' instances.
        self.expert = Expert(
            idea = self.idea, 
            task = self.task, 
            options = self.options)
        # Finalizes 'chapters' attribute for 'book'.
        self._publish_chapters()
        # Finalizes 'techniques' attribute for each Chapter instance in 'book'.
        self._publish_techniques()
        return self.book


@dataclass
class Expert(object):
    """Creates Technique instances for a Book instance.

    Args:
        idea ('Idea'): an instance with project settings.
        task ('Task'): instance with information needed to create a Book 
            instance.
        options ('Repository'): available options with stored 'Technique'
            instances as values.
        
    """
    idea: 'Idea'
    task: 'Task'
    options: 'Repository'

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