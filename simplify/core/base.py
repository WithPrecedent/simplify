"""
.. module:: base
:synopsis: abstract base classes for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractclassmethod
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)


@dataclass
class SimpleFlow(ABC):
    """Base class for the core siMpLify workflow."""

    self.stages: List[str] = field(
        default = lambda: ['initialize', 'draft', 'publish', 'apply'])

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Creates core siMpLify stages and initial stage.
        self.stage = self.stages[0]
        return self

    """ Required Subclass Methods """

    @abstractclassmethod
    def create(cls, *args, **kwargs) -> 'SimpleFlow':
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def add(self, item: Union['SimpleContainer', 'SimpleLoader']) -> None:
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def draft(self) -> None:
        """Subclasses must provide their own methods."""
        return self

    @abstractmethod
    def publish(self) -> None:
        """Subclasses must provide their own methods."""
        return self

    @abstractmethod
    def apply(self, **kwargs) -> None:
        """Subclasses must provide their own methods."""
        return self

    """ Stage Management Methods """

    def advance(self) -> None:
        """Advances to next stage in 'stages'."""
        self.previous_stage = self.stage
        try:
            self.stage = self.stages[self.stages.index(self.stage) + 1]
        except IndexError:
            pass
        return self

    def change_stage(self, new_stage: str) -> None:
        """Manually changes 'stage' to 'new_stage'.

        Args:
            new_stage(str): name of new stage matching a string in 'stages'.

        Raises:
            ValueError: if new_stage is not in 'stages'.

        """
        if new_stage in self.stages:
            self.previous_stage = self.stage
            self.stage = new_stage
        else:
            raise ValueError(' '.join([new_stage, 'is not a recognized stage']))
        return self


@dataclass
class SimpleCreator(ABC):
    """Base class for creating 'Book', 'Chapter', and 'Technique' instances.

    Args:
        worker ('Worker'): instance with information needed to create a 'Book'
            instance.
        idea (Optional['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Required Subclass Methods """

    @abstractmethod
    def draft(self, project: 'Project') -> 'Project':
        """Subclasses must provide their own methods."""
        return project

    @abstractmethod
    def publish(self, project: 'Project') -> 'Project':
        """Subclasses must provide their own methods."""
        return project


@dataclass
class SimpleEngineer(ABC):
    """Base class for applying 'Book' instances to data."""

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Required Subclass Methods """

    @abstractmethod
    def apply(self, project: 'Project', **kwargs) -> 'Project':
        """Subclasses must provide their own methods."""
        return project


@dataclass
class SimpleContainer(ABC):
    """Base class for core siMpLify container classes."""

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Required Subclass Methods """

    @abstractclassmethod
    def create(cls, *args, **kwargs) -> 'SimpleContainer':
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def add(self, item: Union['SimpleContainer', 'SimpleLoader']) -> None:
        """Subclasses must provide their own methods."""
        pass


@dataclass
class SimpleLoader(ABC):
    """Base class for lazy loaders for low-level siMpLify objects.

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

    """
    name: str
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')

    """ Core siMpLify Methods """

    def load(self, component: str) -> object:
        """Returns 'component' from 'module'.

        If 'component' is not a str, it is assumed to have already been loaded
        and is returned as is.

        Args:
            component (str): name of local attribute to load from 'module'.

        Returns:
            object: from 'module'.

        """
        if isinstance(component, str):
            return getattr(import_module(self.module), getattr(self, component))
        else:
            return getattr(self, component)