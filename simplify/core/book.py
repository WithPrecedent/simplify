"""
.. module:: book
:synopsis: standard iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.base import SimpleCatalog
from simplify.core.base import SimpleManuscript
from simplify.core.base import SimpleOutline
from simplify.core.base import SimpleProgression


@dataclass
class Book(SimpleManuscript):
    """Standard class for top-level siMpLify package iterable storage.

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
        techiques (Optional['SimpleCatalog']): a dictionary of options with
            'Technique' instances stored by step. Defaults to an empty
            'SimpleCatalog' instance.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'chapters'.
        returns_data (Optional[bool]): whether the Scholar instance's 'apply'
            expects data when the Book instance is iterated. If False, nothing
            is returned. If true, 'data' is returned. Defaults to True.

    """
    name: str = None
    iterable: Optional[str] = 'chapters'
    techniques: Optional['SimpleCatalog'] = field(
        default_factory = SimpleCatalog)
    chapters: Optional[List['Chapter']] = field(default_factory = list)
    returns_data: Optional[bool] = True

    """ Public Methods """

    def add_chapters(self, chapters: Union['Chapter', List['Chapter']]) -> None:
        if isinstance(chapters, list):
            self.chapters.extend(chapters)
        else:
            self.chapters.append(chapters)
        return self

    def add_techniques(self, techniques: 'MutableMapping') -> None:
        self.techniques.add(techniques = techniques)
        return self

    """ Iterable Proxy Property """

    @property
    def chapters(self) -> List['Chapter']:
        return getattr(self, self.iterable)

    @chapters.setter
    def chapters(self, name: str) -> None:
        self.iterable = name
        return self


@dataclass
class Chapter(SimpleManuscript):
    """Core iterable for sequences of methods to apply to passed data.

    Args:
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None or __class__.__name__.lower() if super().__post_init__ is
            called.
        techniques (Optional[Dict[str, str]]): keys are names of 'steps' in
            the related 'book'. Values are particular techniques to pass to
            Technique or subclasses when instances are created. Defaults to an
            empty dictionary.
        metadata (Optional[Dict[str, Any]]): information needed by particular
            chapter subclasses or for recordkeeping. By default, SimpleEditor
            instances add 'number' as a key and a corresponding integer.
            Defaults to an empty dictionary.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'techniques'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = 'techniques'
    book: Optional['Book'] = None
    number: Optional[int] = 0
    techniques: Optional['SimpleProgression'] = field(
        default_factory = SimpleProgression)
    returns_data: Optional[bool] = True

    """ Iterable Proxy Property """

    @property
    def techniques(self) -> 'SimpleProgression':
        return getattr(self, self.iterable)

    @techniques.setter
    def techniques(self, name: str) -> None:
        self.iterable = name
        return self


@dataclass
class Algorithm(SimpleOutline):
    """Contains settings for creating a Technique instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify object).
        algorithm: str = None
        default: Optional[Dict[str, Any]] = field(default_factory = dict)
        required: Optional[Dict[str, Any]] = field(default_factory = dict)
        runtime: Optional[Dict[str, str]] = field(default_factory = dict)
        selected: Optional[Union[bool, List[str]]] = False
        conditional: Optional[bool] = False
        data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """
    name: str
    module: str
    algorithm: str = None
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)


@dataclass
class Technique(Container):
    """Core iterable for sequences of methods to apply to passed data.

    Args:
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
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
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    technique: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)
    returns_data: Optional[bool] = True

    """ Dunder Methods """

    def __contains__(self, key: str) -> bool:
        """Returns whether 'attribute' is the 'technique'.

        Args:
            key (str): name of item to check.

        Returns:
            bool: whether the 'key' is equivalent to 'technique'.

        """
        return item == self.technique