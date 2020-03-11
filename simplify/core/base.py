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
class SimpleSystem(ABC):
    """Base class for the core siMpLify workflow.

    A 'SimpleSystem' subclass maintains a progress state stored in the attribute
    'stage'. The 'stage' corresponds to whether one of the core workflow
    methods has been called. The string stored in 'stage' can then be used by
    subclasses to alter instance behavior, call methods, or change access
    methods.

    """

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Creates core siMpLify stages and initial stage.
        self.stages = ['initialize', 'draft', 'publish', 'apply']
        self.stage = self.stages[0]
        return self

    """ Required Construction Subclass Methods """

    @abstractclassmethod
    def create(cls, *args, **kwargs) -> 'SimpleSystem':
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def add(self, item: Union['SimpleContainer', 'SimpleComponent']) -> None:
        """Subclasses must provide their own methods."""
        return self

    """ Required Workflow Subclass Methods """

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

    """ Dunder Methods """

    def __getattribute__(self, attribute):
        if attribute in self.stages:
            self.stage = attribute
        return super().__getattribute__(attribute)


    """ Stage Management Methods """

    def advance(self) -> None:
        """Advances to next stage in 'stages'."""
        self.previous_stage = self.stage
        try:
            self.stage = self.stages[self.stages.index(self.stage) + 1]
        except IndexError:
            pass
        return self

    def change(self, stage: str) -> None:
        """Manually changes 'stage' attribute to 'stage'.

        Args:
            stage(str): name of new stage matching a string in 'stages'.

        Raises:
            ValueError: if 'stage' is not in 'stages'.

        """
        if stage in self.stages:
            self.previous_stage = self.stage
            self.stage = stage
        else:
            raise ValueError(' '.join([stage, 'is not a recognized stage']))
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
    def apply(self, project: 'Project', **kwargs) -> 'Project':
        """Subclasses must provide their own methods."""
        return project


@dataclass
class SimpleContainer(ABC):
    """Base class for core siMpLify container classes."""

    """ Required Subclass Methods """

    @abstractclassmethod
    def create(cls, *args, **kwargs) -> 'SimpleContainer':
        """Subclasses must provide their own methods."""
        pass

    @abstractmethod
    def add(self, item: Union['SimpleContainer', 'SimpleComponent']) -> None:
        """Subclasses must provide their own methods."""
        pass


@dataclass
class SimpleComponent(ABC):
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

    """ Required Subclass Methods """

    @abstractclassmethod
    def create(cls, *args, **kwargs) -> 'SimpleContainer':
        """Subclasses must provide their own methods."""
        pass


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