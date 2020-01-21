"""
.. module:: book
:synopsis: siMpLify iterable classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.repository import Sequence
from simplify.core.repository import Repository
from simplify.core.utilities import listify


@dataclass
class Manuscript(Iterable):
    """Base class for Book and Chapter iterables.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            None. If not passed, __class__.__name__.lower() is used.
        iterable (Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter__). Defaults to None.

    """
    name: Optional[str] = None
    iterable: Optional[str] = None

    def __post_init__(self) -> None:
        """Sets 'name' and 'iterable' if not passed.

        Raises:
            ValueError: if 'iterable' is not provided and no default attribute
                is found to store class instance iterables.

        """
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        if self.iterable is None:
            if hasattr(self, 'chapters'):
                self.iterable = 'chapters'
            elif hasattr(self, 'techniques'):
                self.iterable = 'techniques'
            else:
                raise ValueError(' '.join(
                    ['Iterable attribute not found in', self.name]))
        return self

    """ Required ABC Methods """

    def __iter__(self) -> Iterable:
        """Returns class instance iterable."""
        return iter(getattr(self, self.iterable))

    """ Public Methods """

    def add(self, attribute: str, options: Any) -> None:
        """Generic 'add' method' for Manuscripts.

        Users can use the specific instance methods such as 'add_techniques'.
        This method is provided in case a user wants to use a single 'add'
        method with the 'attribute' argument indicating the specific method to
        be called. This might be helpful in certain iteration scenarios.

        Args:
            attribute (str): name of type of object to add to a Manuscript
                instance. This should correspond to a method named:
                'add_[attribute]' in the Manuscript instance.
            options (Any): item(s) to add to a Manuscript instance.

        """
        getattr(self, '_'.join(['add', attribute]))(options)
        return self


@dataclass
class Book(Manuscript):
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
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'chapters'.
        techiques (Optional['Repository']): a dictionary of options with
            'Technique' instances stored by step. Defaults to an empty
            'Repository' instance.
        chapters (Optional['Sequence']): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Sequence'
            instance.
        returns_data (Optional[bool]): whether the Scholar instance's 'apply'
            expects data when the Book instance is iterated. If False, nothing
            is returned. If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = 'chapters'
    techniques: Optional['Repository'] = field(default_factory = Repository)
    chapters: Optional['Sequence'] = field(default_factory = Sequence)
    returns_data: Optional[bool] = True

    """ Public Methods """

    def add_chapters(self,
            chapters: Union[
                'Chapter', List['Chapter'], 'Sequence']) -> None:
        """Adds 'chapters' to class iterable.

        Args:
            chapters (Union['Chapter', List['Chapter'], 'Sequence']): a
                'Sequence' instance or one or more 'Chapter' instances.
                If one or more 'Chapter' instances are passed, the 'name'
                attribute of each is used as the key in the class instance
                iterable.

        """
        if isinstance(chapters, (list, Chapter)):
            for chapter in listify(chapters):
                getattr(self, self.iterable)[chapter.name] = chapter
        else:
            getattr(self, self.iterable).update(chapters)
        return self

    def add_techniques(self,
            techniques: Union[
                'Technique', List['Technique'], 'Repository']) -> None:
        """Adds 'techniques' to 'techniques' attribute.

        Args:
            techniques (Union['Technique', List['Technique'],
                'Repository']): a 'Sequence' instance or one or more
                'Technique' instances. If one or more 'Technique' instances are
                passed, the 'name' and 'technique' attributes are used as the
                keys in the 'techniques' attribute.

        """
        if isinstance(techniques, (list, str)):
            for technique in listify(techniques):
                if not technique.name in getattr(self, self.iterable):
                    getattr(self, self.iterable)[technique.name] = {}
                _name = technique.name
                _technique = technique.technique
                getattr(self, self.iterable)[_name][_technique] = technique
        else:
            getattr(self, self.iterable).update(technique)
        return self

    """ Iterable Proxy Property """

    @property
    def chapters(self) -> 'Sequence':
        """Returns attribute named in 'iterable'.

        Returns:
            'Sequence': with Chapter instances as values.

        """
        return getattr(self, self.iterable)

    @chapters.setter
    def chapters(self, chapters: 'Sequence') -> None:
        """Sets attribute named in 'iterable' to 'chapters'.

        Args:
            chapters ('Sequence'): an instance with Chapter instance
                values.

        """
        setattr(self, self.iterable, chapters)
        return self

    @chapters.deleter
    def chapters(self, chapters: Union[List[str], str]) -> None:
        """Deletes 'chapters' from attribute named in 'iterable'.

        Args:
            chapters (Union[List[str], str]): key(s) to Chapter instances to
                be removed from the attribute named in 'iterable'.
        """
        for chapter in listify(chapters):
            try:
                del getattr(self, self.iterable)[chapter]
            except KeyError:
                pass
        return self


@dataclass
class Chapter(Manuscript):
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
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'techniques'.
        book (Optional['Book']): related Book or subclass instance. Defaults to
            None.
        number (Optional[int]): number of instance in a sequence. The value is
            used for internal recordkeeping and reporting. Defaults to 0.
        techniques (Optional[Dict[str, str]]): keys are names of 'steps' in
            the related 'book'. Values are particular techniques to pass to
            Technique or subclasses when instances are created. Defaults to an
            empty dictionary.
        returns_data (Optional[bool]): whether the Book instance's 'apply'
            method returns data when iterated. If False, nothing is returned.
            If true, 'data' is returned. Defaults to True.

    """
    name: Optional[str] = None
    iterable: Optional[str] = 'techniques'
    book: Optional['Book'] = None
    number: Optional[int] = 0
    techniques: Optional['Sequence'] = field(default_factory = Sequence)
    returns_data: Optional[bool] = True

    """ Iterable Proxy Property """

    @property
    def techniques(self) -> 'Sequence':
        """Returns attribute named in 'iterable'.

        Returns:
            'Sequence': with Technique instances as values.

        """
        if self.iterable in ['techniques']:
            return self._techniques
        else:
            return getattr(self, self.iterable)

    @techniques.setter
    def techniques(self, techniques: 'Sequence') -> None:
        """Sets attribute named in 'iterable' to 'techniques'.

        Args:
            techniques ('Sequence'): an instance with Technique instance
                values.

        """
        if self.iterable in ['techniques']:
            setattr(self, '_techniques', techniques)
        else:
            setattr(self, self.iterable, techniques)
        return self

    @techniques.deleter
    def techniques(self, techniques: Union[List[str], str]) -> None:
        """Deletes 'techniques' from attribute named in 'iterable'.

        Args:
            techniques (Union[List[str], str]): key(s) to Technique instances to
                be removed from the attribute named in 'iterable'.
        """
        for technique in listify(techniques):
            try:
                if self.iterable in ['techniques']:
                    del getattr(self, '_techniques')[technique]
                else:
                    del getattr(self, self.iterable)[technique]
            except KeyError:
                pass
        return self