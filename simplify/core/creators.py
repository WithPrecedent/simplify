"""
.. module:: creators
:synopsis: constructs books, chapters, and techniques
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.book import Book
from simplify.core.book import Chapter
from simplify.core.book import Technique
from simplify.core.repository import Repository
from simplify.core.utilities import listify


@dataclass
class Creator(ABC):

    worker: 'Worker'
    idea: ClassVar['Idea'] = None

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self, project: 'Project') -> 'Project':
        """Subclasses must provide their own methods."""
        return project

    @abstractmethod
    def publish(self, project: 'Project') -> 'Project':
        """Subclasses must provide their own methods."""
        return project


@dataclass
class Publisher(Creator):

    worker: 'Worker'
    idea: ClassVar['Idea'] = None

    def __post_init__(self) -> None:
        # Creates 'author' which is used to create 'Chapter' instances.
        self.author = Author(worker = self.worker)
        return self

    """ Core siMpLify Methods """

    def draft(self, project: 'Project') -> 'Project':
        """Drafts 'Book' instance and deposits it in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be added.

        Returns:
            'Project': with 'Book' instance added.

        """
        project[self.worker.name] = self.worker.load('book')()
        return self.author.draft(project = project)

    def publish(self, project: 'Project') -> 'Project':
        """Finalizes 'Book' instance and deposits it in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be added.

        Returns:
            'Project': with 'Book' instance added.

        """
        return self.author.publish(project = project)


@dataclass
class Author(Creator):

    worker: 'Worker'
    idea: ClassVar['Idea'] = None

    def __post_init__(self) -> None:
        # Creates 'expert' which is used to create 'Technique' instances.
        self.expert = Expert(worker = self.worker)
        return self

    """ Private Methods """

    def _draft_parallel(self, project: 'Project') -> 'Project':
        """Drafts 'Book' instance with a parallel chapter structure.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        # Creates list of steps from 'project'.
        steps = list(project.overview[self.worker.name].keys())
        # Creates 'possible' list of lists of 'techniques'.
        possible = list(project.overview[self.worker.name].values())
        # Creates a list of lists of the Cartesian product of 'possible'.
        combinations = list(map(list, product(*possible)))
        # Creates Chapter instance for every combination of techniques.
        for techniques in combinations:
            steps = zip(steps, techniques)
            chapter = self.worker.load('chapter')(steps = steps)
            project[self.worker.name].chapters.append(chapter)
        return project

    def _draft_serial(self, project: 'Project') -> 'Project':
        """Drafts 'Book' instance with a serial 'steps' structure.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        new_steps = []
        for step, techniques in project.overview[self.worker.name].items():
            for technique in techniques:
                new_steps.append((step, technique))
        project[self.worker.name].steps.extend(new_steps)
        return project

    def _publish_parallel(self, project: 'Project') -> 'Project':
        """Finalizes 'Book' instance in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        new_chapters = []
        for chapter in project[self.worker.name].chapters:
            new_chapters.append(self._publish_techniques(instance = chapter))
        project[self.worker.name].chapters = new_chapters
        return project

    def _publish_serial(self, project: 'Project') -> 'Project':
        """Finalizes 'Book' instance in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        project[self.worker.name] = self._publish_techniques(
            instance = project[self.worker.name])
        return project

    def _publish_techniques(self,
            instance: Union['Book', 'Chapter']) -> Union['Book', 'Chapter']:
        """Finalizes 'techniques' in 'Book' or 'Chapter' instance.

        Args:
            instance (Union['Book', 'Chapter']): an instance with 'steps' to be
                converted to 'techniques'.

        Returns:
            Union['Book', 'Chapter']: with 'techniques' added.

        """
        techniques = []
        for step in instance.steps:
            techniques.extend(self.expert.publish(step = step))
        instance.techniques = techniques
        return instance

    """ Core siMpLify Methods """

    def draft(self, project: 'Project') -> 'Project':
        """Drafts 'Chapter' instances and deposits them in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be added.

        Returns:
            'Project': with 'Book' instance added.

        """
        if hasattr(project[self.worker.name], 'steps'):
            return self._draft_serial(project = project)
        else:
            return self._draft_parallel(project = project)

    def publish(self, project: 'Project') -> 'Project':
        """Finalizes 'Book' instance in 'project'.

        Args:
            project ('Project'): an instance for a 'Book' instance to be
                modified.

        Returns:
            'Project': with 'Book' instance modified.

        """
        if hasattr(project[self.worker.name], 'steps'):
            return self._publish_serial(project = project)
        else:
            return self._publish_parallel(project = project)


@dataclass
class Expert(Creator):

    worker: 'Worker'
    idea: ClassVar['Idea'] = None

    """ Private Methods """

    def _publish_technique(self,
            technique: 'Technique',
            step: Tuple[str, str]) -> 'Technique':
        """Finalizes 'technique'.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        technique.step = step[0]
        if technique.module and technique.algorithm:
            technique.algorithm = technique.load('algorithm')
        return self._publish_parameters(technique = technique)

    def _publish_parameters(self, technique: 'Technique') -> 'Technique':
        """Finalizes 'parameters' for 'technique'.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        parameter_types = ['idea', 'selected', 'required', 'runtime']
        # Iterates through types of 'parameter_types'.
        for parameter_type in parameter_types:
            try:
                technique = getattr(self, '_'.join(
                    ['_publish', parameter_type]))(technique = technique)
            except TypeError:
                pass
        return technique

    def _publish_idea(self, technique: 'Technique') -> 'Technique':
        """Acquires parameters from 'Idea' instance.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(
                self.idea['_'.join([technique.name, 'parameters'])])
        except KeyError:
            try:
                technique.parameters.update(
                    self.idea['_'.join([technique.step, 'parameters'])])
            except AttributeError:
                pass
        return technique

    def _publish_selected(self, technique: 'Technique') -> 'Technique':
        """Limits parameters to those appropriate to the 'technique'.

        If 'technique.selected' is True, the keys from 'technique.defaults' are
        used to select the final returned parameters.

        If 'technique.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        if technique.selected:
            if isinstance(technique.selected, list):
                parameters_to_use = technique.selected
            else:
                parameters_to_use = list(technique.default.keys())
            new_parameters = {}
            for key, value in technique.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            technique.parameters = new_parameters
        return technique

    def _publish_required(self, technique: 'Technique') -> 'Technique':
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            technique.parameters.update(technique.required)
        except TypeError:
            pass
        return technique

    def _publish_search(self, technique: 'Technique') -> 'Technique':
        """Separates variables with multiple options to search parameters.

        Args:
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

    def _publish_runtime(self, technique: 'Technique') -> 'Technique':
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        Args:
            technique ('Technique'): an instance for parameters to be added to.

        Returns:
            'Technique': instance with parameters added.

        """
        try:
            for key, value in technique.runtime.items():
                try:
                    technique.parameters.update(
                        {key: getattr(self.idea['general'], value)})
                except AttributeError:
                    raise AttributeError(' '.join(
                        ['no matching runtime parameter', key, 'found']))
        except (AttributeError, TypeError):
            pass
        return technique

    """ Core siMpLify Methods """

    def draft(self, project: 'Project') -> 'Project':
        return project

    def publish(self, step: Tuple[str, str]) -> 'Technique':
        """Finalizes 'Technique' instance from 'step'.

        Args:
            step (Tuple[str, str]): information needed to create a 'Technique'
                instance.

        Returns:
            'Technique': instance ready for application.

        """
        if step[1] in ['none']:
            return [self.worker.technique(name = 'none', step = step[0])]
        elif step[1] in ['all', 'default']:
            final_techniques = []
            techniques = self.worker.options[step[0]][step[1]]
            for technique in techniques:
                final_techniques.append(
                    self._publish_technique(
                        step = step,
                        technique = technique))
            return final_techniques
        else:
            # Gets appropriate Technique and creates an instance.
            technique = self.worker.options[step[0]][step[1]]
            return [self._publish_technique(
                step = step,
                technique = technique)]