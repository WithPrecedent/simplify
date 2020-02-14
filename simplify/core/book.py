"""
.. module:: book
:synopsis: primary siMpLify iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.repository import Plan
from simplify.core.repository import Repository
from simplify.core.utilities import listify


@dataclass
class Book(Repository):
    """Standard class for top-level siMpLify package iterable storage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        default (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.
        chapters (Optional[List[str]]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = field(default_factory = dict)
    defaults: Optional[List[str]] = field(default_factory = list)
    chapters: Optional[List['Chapter']] = field(default_factory = list)
    idea: ClassVar['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes attributes and settings."""
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        if hasattr(self, '_iterable'):
            self.proxify(name = self._iterable)
        super().__post_init__()
        return self

    """ Dunder Methods """

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
            key: Optional[str] = None,
            value: Optional[Any] = None,
            contents: Optional[Union['Repository', Dict[str, Any]]] = None,
            chapters: Optional[
                Union[List['Chapter'], 'Chapter']] = None) -> None:
        """Combines arguments with 'contents'.

        Args:
            key (Optional[str]): key for 'value' to use. Defaults to None.
            value (Optional[Any]): item to store in 'contents'. Defaults to
                None.
            contents (Optional[Union['Repository', Dict[str, Any]]]):
                another 'Repository' instance/subclass or a compatible
                dictionary. Defaults to None.
            chapters (Union['Chapter', List['Chapter']]: a 'Chapter' instance or
                list of such instances.

        """
        if chapters:
            self.chapters.extend(listify(chapters, default_empty = True))
        super().add(key = key, value = value, contents = contents)
        return self

    def proxify(self, name: str) -> None:
        """Adds a proxy property to refer to 'chapters'.

        Args:
            name (str): name of proxy property.

        """
        setattr(self, name, property(
            fget = self._proxy_getter,
            fset = self._proxy_setter,
            fdel = self._proxy_deleter))
        return self


@dataclass
class Chapter(Plan):
    """Standard class for bottom-level siMpLify package iterable storage.

    Args:
        steps (Optional[List[str]]): an ordred set of steps. Defaults to an
            empty list. All items in 'steps' should correspond to keys in
            'repository' before iterating.
        repository ('Book'): instance with options for 'steps'.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    steps: Union[List[str], str]
    repository: 'Book'
    idea: ClassVar['Idea'] = None

    """ Proxy Property Methods """

    def _proxy_getter(self) -> List['Technique']:
        """Proxy getter for 'steps'.

        Returns:
            List['Technique'].

        """
        return self.steps

    def _proxy_setter(self, value: List['Technique']) -> None:
        """Proxy setter for 'steps'.

        Args:
            value (List['Technique']): list of 'Technique' instances to store.

        """
        self.steps = value
        return self

    def _proxy_deleter(self) -> None:
        """Proxy deleter for 'steps'."""
        self.steps = []
        return self

    """ Public Methods """

    def proxify(self, name: str) -> None:
        """Adds a proxy property to refer to 'chapters'.

        Args:
            name (str): name of proxy property.

        """
        setattr(self, name, property(
            fget = self._proxy_getter,
            fset = self._proxy_setter,
            fdel = self._proxy_deleter))
        return self


@dataclass
class Technique(Outline):
    """Core iterable for sequences of methods to apply to passed data.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        technique (Optional[str]): name of particular technique to be used. It
            should correspond to a key in the related 'book' instance. Defaults
            to None.

    """
    name: Optional[str] = None
    technique: Optional[str] = None
    algorithm: Optional[object] = None
    module: Optional[str]
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)

    """ Core siMpLify Methods """

    def apply(self, data: Union['Dataset', 'Book']) -> Union['Dataset', 'Book']:
        return data