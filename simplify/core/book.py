"""
.. module:: book
:synopsis: siMpLify project deliverable
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import abc
import collections.abc
import dataclasses
import importlib
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclasses.dataclass
class Book(base.SimpleRepository, base.SimpleProxy):
    """Top-level siMpLify iterable for a 'Worker'.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        chapters (Optional['SimpleRepository']): iterable collection of
            'Chapter' instances. Defaults to an empty 'SimpleRepository'.

    """
    name: Optional[str] = None
    chapters: Optional['SimpleRepository'] = dataclasses.field(
        default_factory = base.SimpleRepository)

    def __post_init__(self) -> None:
        """Converts 'chapters' to a property for 'contents' attribute."""
        self.contents = self.chapters
        self.proxify(proxy = 'chapters', name = 'contents')
        return self


@dataclasses.dataclass
class Chapter(base.SimpleRepository, base.SimpleProxy):
    """Standard class for siMpLify nested iterable storage.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        techniques (Optional['SimpleRepository']): iterable collection of
            'Technique' instances. Defaults to an empty 'SimpleRepository'.

    """
    name: Optional[str] = None
    techniques: Optional['SimpleRepository'] = dataclasses.field(
        default_factory = base.SimpleRepository)

    def __post_init__(self) -> None:
        """Converts 'techniques' to a property for 'contents' attribute."""
        self.contents = self.techniques
        self.proxify(proxy = 'techniques', name = 'contents')
        return self


@dataclasses.dataclass
class Technique(base.SimpleComponent):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class
            instance needs settings from the shared 'Idea' instance, 'name'
            should match the appropriate section name in that 'Idea' instance.
            When subclassing, it is a good idea to use the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to None or __class__.__name__.lower().
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[Union[str, object]]): name of object in 'module' to
            load or the process object which executes the primary method of
            a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    module: Optional[str] = dataclasses.field(
        default_factory = lambda: 'simplify.core')
    algorithm: Optional[Union[str, object]] = None
    parameters: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (
            f'siMpLify {self.__class__.__name__} '
            f'technique: {self.name} '
            f'parameters: {str(self.parameters)} ')


@dataclasses.dataclass
class Parameters(structure.SimpleRepository):
    """Base class for constructing and storing 'Technique' parameters.

    Args:
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.

    """
    contents: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)


@dataclasses.dataclass
class Expert(base.SimpleCreator):

    worker: 'Worker'

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
            except (TypeError, AttributeError):
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