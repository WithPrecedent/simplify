"""
.. module:: base
:synopsis: abstract base classes for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)


@dataclass
class SimpleSettings(ABC):
    """Provides shared configuration settings and logger to subclasses.

    Args:
        idea (ClassVar['Idea']): instance with general project settings.
        filer (ClassVar['Filer']): instance with settings and methods for file
            managerment.
        journal (ClassVar['Journal']): instance which logs activity, errors,
            and timing for a siMpLify 'Project'.

    """
    idea: ClassVar['Idea']
    filer: ClassVar['Filer']
    journal: ClassVar['Journal']

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self


@dataclass
class SimpleCreator(SimpleSettings, ABC):
    """Base class for creating 'Book', 'Chapter', and 'Technique' instances.

    Args:
        worker ('Worker'): instance with information needed to create a 'Book'
            instance.
        idea (ClassVar['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: ClassVar['Idea']

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

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
class SimpleEngineer(SimpleSettings, ABC):
    """Base class for applying 'Book' instances to data.

    Args:
        worker ('Worker'): instance with information needed to apply a 'Book'
            instance.
        idea (ClassVar['Idea']): instance with project settings.

    """
    worker: 'Worker'
    idea: ClassVar['Idea']

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def apply(self,
            project: 'Project',
            data: Optional['Dataset'] = None) -> 'Project':
        """Subclasses must provide their own methods."""
        return project


@dataclass
class SimpleManuscript(ABC):
    """Base class for 'Book' and 'Chapter'."""

    def __post_init__(self) -> None:
        """Initializes attributes and settings."""
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        try:
            if self.iterable:
                self.proxify(name = self.iterable)
        except AttributeError:
            pass
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable of attribute named in 'iterable' attribute.

        Returns:
            Iterable: of attribute named in 'iterable' attribute.

        """
        return iter(getattr(self, self.iterable))

    def __len__(self) -> int:
        """Returns length of attribute named in 'iterable' attribute.

        Returns:
            Integer: length of attribute named in 'iterable' attribute.

        """
        return len(getattr(self, self.iterable))

    """ Proxy Property Methods """

    def _proxy_getter(self) -> List['Technique']:
        """Proxy getter for 'techniques'.

        Returns:
            List['Technique'].

        """
        return self.techniques

    def _proxy_setter(self, value: List['Technique']) -> None:
        """Proxy setter for 'techniques'.

        Args:
            value (List['Technique']): list of 'Technique' instances to store.

        """
        self.techniques = value
        return self

    def _proxy_deleter(self) -> None:
        """Proxy deleter for 'techniques'."""
        self.techniques = []
        return self

    """ Public Methods """

    def proxify(self, name: str) -> None:
        """Adds a proxy property to refer to class iterable.

        Args:
            name (str): name of proxy property.

        """
        setattr(self, name, property(
            fget = self._proxy_getter,
            fset = self._proxy_setter,
            fdel = self._proxy_deleter))
        return self


@dataclass
class SimpleLoader(ABC):
    """Lazy loader for low-level objects used in a siMpLify project.

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
        default_module (Optional[str]): name of a backup module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.core'. Subclasses should not generally
            override this attribute. It allows the 'load' method to use generic
            classes if the specified one is not found.

    """
    name: str
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = field(
         default_factory = lambda: 'simplify.core')

    """ Public Methods """

    def load(self, component: str) -> object:
        """Returns 'component' from 'module' or 'default_module.

        If 'component' is not a str, it is assumed to have already been loaded
        and is returned as is.

        Args:
            component (str): name of object to load from 'module' or
                'default_module'.

        Raises:
            ImportError: if 'component' is not found in 'module' or
                'default_module'.

        Returns:
            object: from 'module' or 'default_module'.

        """
        # If 'component' is a string, attempts to load from 'module' or, if not
        # found there, 'default_module'.
        if isinstance(getattr(self, component), str):
            try:
                return getattr(
                    import_module(self.module),
                    getattr(self, component))
            except (ImportError, AttributeError):
                try:
                    return getattr(
                        import_module(self.default_module),
                        getattr(self, component))
                except (ImportError, AttributeError):
                    try:
                        return getattr(
                            import_module(self.default_module),
                            self.default_components[component])
                    except (ImportError, AttributeError):
                        raise ImportError(' '.join(
                            [getattr(self, component), 'is neither in',
                                self.module, 'nor', self.default_module]))
        # If 'component' is not a string, it is returned as is.
        else:
            return getattr(self, component)


@dataclass
class SimpleStage(ABC):
    """Base class for shared state of a siMpLify 'Project' instance."""

    def _post_init__(self) -> None:
        """Creates core siMpLify stages and initial stage."""
        self.stages = ['outline', 'draft', 'publish', 'apply']
        self.stage = 'outline'
        return self

    """ Stage Management Methods """

    def advance(self) -> None:
        """Advances to next stage in 'stages'."""
        self.previous_stage = self.stage
        stage_index = self.stages.index(self.stage)
        self.stage = self.stages[self.stage_index + 1]
        return self

    def change_stage(self, new_stage: str) -> None:
        """Manually changes 'stage' to 'new_stage'.

        Args:
            new_stage(str): name of new stage matching a string in 'stages'.

        Raises:
            TypeError: if new_stage is not in 'stages'.

        """
        if new_stage in self.stages:
            self.previous_stage = self.stage
            self.stage = new_stage
        else:
            raise ValueError(' '.join([new_stage, 'is not a recognized stage']))
        return self