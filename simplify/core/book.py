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
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.utilities import listify


@dataclass
class Book(object):
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
        chapters (Optional[List[str]]): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.

    """
    name: Optional[str] = None
    chapters: Optional[List['Chapter']] = field(default_factory = list)

    def __post_init__(self) -> None:
        """Initializes attributes and settings."""
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        try:
            self.proxify(name = self._iterable)
        except AttributeError:
            pass
        return self

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
            chapters (Union['Chapter', List['Chapter']]: a 'Chapter' instance or
                list of such instances.

        """
        self.chapters.extend(listify(chapters, default_empty = True))
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
class Chapter(object):
    """Standard class for bottom-level siMpLify package iterable storage.

    Args:
        steps (Optional[List[str]]): an ordred set of steps. Defaults to an
            empty list. All items in 'steps' should correspond to keys in
            'repository' before iterating.
        repository ('Book'): instance with options for 'steps'.
        idea (ClassVar['Idea']): shared 'Idea' instance with project settings.

    """
    steps: Union[List[Tuple[str, str]], List['Technique']]

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
class Technique(Container):
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
        step (Optional[str]): name of step where the class isntance is to
            be applied. Defaults to None.

    """
    name: Optional[str] = None
    step: Optional[str] = None
    module: Optional[str] = None
    default_module: Optional[str] = field(
        default_factory = lambda: 'simplify.core')
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)

    """ Required ABC Methods """

    def __contains__(self, key: str) -> bool:
        """Returns whether 'attribute' is the 'name'.

        Args:
            key (str): name of item to check.

        Returns:
            bool: whether the 'key' is equivalent to 'name'.

        """
        return key == self.name

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return ' '.join(
            ['technique', self.name,
            'step', self.step,
            'parameters', str(self.parameters)])

    """ Public Methods """

    def load(self, component: str) -> object:
        """Returns 'component' from 'module'.

        Args:
            component (str): name of object to load from 'module'.

        Returns:
            object: from 'module'.

        """
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
                raise ImportError(' '.join(
                    [getattr(self, component), 'is neither in', self.module,
                        'nor', self.default_module]))


    """ Core siMpLify Methods """

    def apply(self, data: Union['Dataset', 'Book']) -> Union['Dataset', 'Book']:
        return data