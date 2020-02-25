"""
.. module:: book
:synopsis: primary siMpLify iterable classes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from abc import ABC
from collections.abc import Container
from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.base import SimpleLoader
from simplify.core.base import SimpleManuscript
from simplify.core.utilities import listify


@dataclass
class Book(SimpleManuscript):
    """Standard class for top-level siMpLify package iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        chapters (Optional[List[str]]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.

    """
    name: Optional[str] = None
    chapters: Optional[List['Chapter']] = field(default_factory = list)

    """ Other Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable of 'chapters'.

        Returns:
            Iterable: of 'chapters'.

        """
        return iter(self.chapters)

    def __len__(self) -> int:
        """Returns length of 'chapters'.

        Returns:
            Integer: length of 'chapters'.

        """
        return len(self.chapters)

    """ Proxy Property Methods """

    def _proxy_getter(self) -> List['Chapter']:
        """Proxy getter for 'chapters'.

        Returns:
            List['Chapter'].

        """
        return self.chapters

    def _proxy_setter(self, value: List['Chapter']) -> None:
        """Proxy setter for 'chapters'.

        Args:
            value (List['Chapter']): list of 'Chapter' instances to store.

        """
        self.chapters = value
        return self

    def _proxy_deleter(self) -> None:
        """Proxy deleter for 'chapters'."""
        self.chapters = []
        return self

    """ Public Methods """

    def add(self,
            chapters: Union[List['Chapter'], 'Chapter']) -> None:
        """Combines 'chapters' with existing 'chapters' attribute.

        Args:
            chapters (Union['Chapter', List['Chapter']]): a 'Chapter' instance
                or list of such instances.

        """
        self.chapters.extend(listify(chapters, default_empty = True))
        return self


@dataclass
class Chapter(SimpleManuscript):
    """Standard class for bottom-level siMpLify package iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.

    """
    name: Optional[str] = None


@dataclass
class Technique(SimpleLoader):
    """Base method wrapper for applying algorithms to data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        step (Optional[str]): name of step when the class instance is to be
            applied. Defaults to None.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        algorithm (Optional[object]): callable object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (
            f'siMpLify {self.__class__.__name__} '
            f'technique: {self.name} '
            f'step: {self.step} '
            f'parameters: {str(self.parameters)} ')