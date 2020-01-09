"""
.. module:: book
:synopsis: standard iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.base import SimpleManuscript
from simplify.core.base import SimpleOptions
from simplify.core.base import SimpleOutline


@dataclass
class BookOutline(SimpleOutline):
    """Object construction techniques used by SimpleEditor instances.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__ dunder method.

    Users can use the idiom 'x in Option' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        component (str): name of attribute containing the name of the python
            object within 'module' to load.
        book (str): name of Book object in 'module' to load. Defaults to None.

    """
    name: str
    module: str
    component: str = 'book'
    book: str = None


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
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'contents'. If a string is passed, it is
            converted to a 1-item list. Defaults to an empty list.
        contents (Optional['Contents']): stores an Contents instance
            or subclasses which can be iterated in 'chapters'. Defaults to an
            empty dictionary.
        chapters (Optional[List['Chapter']]): a list of Chapter instances that
            include a series of Technique instances to be applied to passed
            data. Defaults to an empty list.
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'chapters'.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = 'chapters'
    returns_data: Optional[bool] = True


@dataclass
class ChapterOutline(SimpleOutline):
    """Object construction techniques used by SimpleEditor instances.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__ dunder method.

    Users can use the idiom 'x in Option' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        component (str): name of attribute containing the name of the python
            object within 'module' to load.
        chapter (str): name of Chapter object in 'module' to load. Defaults to
            None.

    """
    name: str
    module: str
    component: str = 'chapter'
    chapter: str = None
    metadata: Optional[Dict[str, Any]] = field(default_factory = dict)


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
    number: Optional[int] = 0
    iterable: Optional[str] = 'techniques'
    returns_data: Optional[bool] = True


@dataclass
class TechniqueOutline(SimpleOutline):
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
        component (str): name of attribute containing the name of the python
            object within 'module' to load. Defaults to 'technique'.
        technique: str = None
        default: Optional[Dict[str, Any]] = field(default_factory = dict)
        required: Optional[Dict[str, Any]] = field(default_factory = dict)
        runtime: Optional[Dict[str, str]] = field(default_factory = dict)
        selected: Optional[Union[bool, List[str]]] = False
        conditional: Optional[bool] = False
        data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """
    name: str
    module: str
    component: str = 'technique'
    technique: str = None
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    conditional: Optional[bool] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)


@dataclass
class Technique(SimpleManuscript):
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
    book: Optional['Book'] = None
    name: Optional[str] = None
    technique: Optional[str] = None
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

    def __iter__(self) -> NotImplementedError:
        """Technique instances cannot be iterated."""
        raise NotImplementedError('Technique instances cannot be iterated.')

    def __len__(self) -> NotImplementedError:
        """Technique instances have no length."""
        raise NotImplementedError('Technique instances have no length.')
