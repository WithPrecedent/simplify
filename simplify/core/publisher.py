"""
.. module:: publisher
:synopsis: constructs books, chapters, and techniques
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

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
    task: Optional['Task'] = None

    def __post_init__(self) -> None:
        # Creates 'author' which is used to create 'Chapter' instances.
        self.author = Author(idea = self.idea)
        return self

    """ Private Methods """

    def _draft_steps(self, task: 'Task') -> 'Task':
        """Drafts 'steps' from 'task' or 'idea'."""
        try:
            task.steps = listify(self.idea[task.name]['_'.join(
                [task.name, 'steps'])])
        except (KeyError, AttributeError):
            task.steps = []
        return task

    def _draft_techniques(self, task: 'Task') -> 'Task':
        """Drafts 'techniques' from 'idea'."""
        task.techniques = {}
        for step in task.steps:
            key = '_'.join([step, 'techniques'])
            try:
                task.techniques[step] = listify(self.idea[task.name][key])
            except KeyError:
                task.techniques[step] = ['none']
        return task

    def _draft_options(self, task: 'Task') -> 'Task':
        """Creates options for creating a Book contents."""
        if isinstance(task.options, Repository):
            pass
        elif isinstance(task.options, str):
            task.options = task.load('options')(idea = self.idea)
        elif isinstance(task.options, dict):
            task.options = Repository(
                contents = task.options,
                idea = self.idea)
        else:
            raise TypeError('task.options must be Repository, dict, or str')
        return task

    def _draft_book(self, task: 'Task') -> 'Task':
        """Creates initial, empty Book instance."""
        if isinstance(task.book, (str, Book)):
            if isinstance(task.book, Book):
                pass
            else:
                task.book = task.load('book')(name = task.book)
        else:
            raise TypeError('task.book must be Book or str')
        return task

    """ Core siMpLify Methods """

    def draft(self, task: Optional['Task'] = None) -> 'Task':
        """Drafts initial attributes and settings of a Book instance. """
        if task is None:
            task = self.task
        # Creates 'raw_techniques', 'steps', 'options, and 'book'.
        task = self._draft_steps(task = task)
        task = self._draft_techniques(task = task)
        task = self._draft_options(task = task)
        task = self._draft_book(task = task)
        task = self.author.draft(task = task)
        return task

    def publish(self, task: Optional['Task'] = None) -> 'Book':
        """Finalizes Book instance, making all changes before application."""
        if task is None:
            task = self.task
        return self.author.publish(task = task)


@dataclass
class Author(object):
    """Creates Chapter instances for a Book instance.

    Args:
        idea ('Idea'): an instance with project settings.
        task ('Task'): instance with information needed to create a Book
            instance.

    """
    idea: 'Idea'

    def __post_init__(self) -> None:
        # Creates 'expert' which is used to create 'Technique' instances.
        self.expert = Expert(idea = self.idea)
        return self

    """ Private Methods """

    def _get_selected_techniques(self, task: 'Task') -> List[List[str]]:
        """Converts 'techniques' values to a list of lists.

        Return:
            List[List[str]]: all possible techniques for each step.

        """
        possible_techniques = []
        for step, techniques in task.techniques.items():
            if step in task.steps:
                possible_techniques.append(listify(techniques))
        return possible_techniques

    def _publish_techniques(self, techniques: Dict[str, str]) -> Dict[str, str]:
        """Publishes instanced 'techniques' for each 'Chapter' instance."""
        new_techniques = {}
        for step, technique in techniques.items():
            new_techniques[step] = self.options[step][technique]
        return new_techniques

    """ Core siMpLify Methods """

    def draft(self, task: 'Task') -> 'Task':
        """Drafts instanced 'chapters' for a Book instance."""
        # Creates 'possible' list of lists of 'techniques'.
        possible = self._get_selected_techniques(task = task)
        # Creates a list of lists of the Cartesian product of 'possible'.
        chapters = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of techniques.
        for techniques in chapters:
            contents = dict(zip(task.steps, techniques))
            new_techniques = Plan(idea = self.idea, contents = contents)
            task.book.chapters.append(Chapter(techniques = new_techniques))
        return task

    def publish(self, task: 'Task') -> 'Book':
        """Finalizes all 'Chapter' instances in a 'Book' instance.

        """
        for chapter in task.book.chapters:
            new_techniques = {}
            for step, technique in chapter.techniques.items():
                new_techniques[step] = self.expert.publish(
                    task = task,
                    step = step,
                    technique = technique)
            chapter.techniques = new_techniques
        return task.book


@dataclass
class Expert(object):
    """Creates Technique instances for Chapter instances.

    Args:
        idea ('Idea'): an instance with project settings.

    """
    idea: 'Idea'

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
        technique.fit_method = outline.fit_method
        technique.transform_method = outline.transform_method
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
        parameter_types = ['idea', 'selected', 'required', 'runtime']
        # Iterates through types of 'parameter_types'.
        for parameter_type in parameter_types:
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
        technique.parameter_space = {}
        new_parameters = {}
        for parameter, values in technique.parameters.items():
            if isinstance(values, list):
                if any(isinstance(i, float) for i in values):
                    technique.parameter_space.update(
                        {parameter: uniform(values[0], values[1])})
                elif any(isinstance(i, int) for i in values):
                    technique.parameter_space.update(
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

    def publish(self, task: 'Task', step: str, technique: str) -> 'Technique':
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
            return self._publish_outline(
                        step = step,
                        technique = technique,
                        outline = task.options[step][technique])