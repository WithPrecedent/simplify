"""
.. module:: library
:synopsis: primary siMpLify iterable classes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections.abc import Container
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.base import SimpleLoader
from simplify.core.base import SimpleManuscript
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify


@dataclass
class Library(MutableMapping):
    """Serializable object containing complete siMpLify library.

    Args:
        identification (Optional[str]): a unique identification name for this
            'Library' instance. The name is used for creating file folders
            related to the 'Library'. If not provided, a string is created
            from the date and time.
        catalog (Optional[Dict[str, Dict[str, List[str]]]]): nested dictionary
            of workers, steps, and techniques for a siMpLify library. Defaults
            to an empty dictionary. An catalog is not strictly needed for
            object serialization, but provides a good summary of the various
            options selected in a particular 'Library'. As a result, it is
            used by the '__repr__' and '__str__' methods.
        books (Optional[Dict[str, 'Book']]): stored 'Book' instances. Defaults
            to an empty dictionary.

    """
    identification: Optional[str] = field(default_factory = datetime_string)
    catalog: Optional[Dict[str, Dict[str, List[str]]]] = field(
        default_factory = dict)
    books: Optional[Dict[str, 'Book']] = field(default_factory = dict)

    """ Factory Method """

    @classmethod
    def create(cls, manager: 'Manager') -> 'Library':
        """Creates a 'Library' instance from 'manager'.

        Args:
            manager ('Manager'): an instance with stored 'workers'.

        Returns:
            'Library': instance, properly configured.

        """
        books = {}
        for name, worker in manager.workers.items():
            books[name] = worker.load('book')()
        return cls(books = books)

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> 'Book':
        """Returns key from 'books'.
        Args:
            key (str): key to item in 'books'.

        Returns:
            'Book': from 'books'.

        """
        return self.books[key]

    def __setitem__(self, key: str, value: 'Book') -> None:
        """Sets 'key' in 'books' to 'value'.

        Args:
            key (str): key to item in 'books' to set.
            value ('Book'): instance to place in 'books'.

        """
        self.books[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'books'.

        Args:
            key (str): key in 'books'.

        """
        try:
            del self.books[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'books'.

        Returns:
            Iterable: of 'books'.

        """
        return iter(self.books)

    def __len__(self) -> int:
        """Returns length of 'books'.

        Returns:
            int: length of 'books'.
        """
        return len(self.books)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return f'Project {self.identification}: {str(self.catalog)}'

    """ Core siMpLify Methods """

    def add(self, book: 'Book') -> None:
        """Adds 'book' to 'books'.

        Args:
            book ('Book'): an instance to be added.

        Raises:
            ValueError: if 'book' is not a 'Book' instance.

        """
        # Validates 'book' type as 'Book' before adding to 'books'.
        if isinstance(book, Book):
            self.books[book.name] = book
        else:
            raise ValueError('book must be a Book instance to add to a Library')
        return self


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
        default_module (Optional[str]): name of a backup module where object to
            use is located (can either be a siMpLify or non-siMpLify module).
            Defaults to 'simplify.core'. Subclasses should not generally
            override this attribute. It allows the 'load' method to use generic
            classes if the specified one is not found.
        algorithm (Optional[object]): callable object which executes the primary
            method of a class instance. Defaults to None.
        parameters (Optional[Dict[str, Any]]): parameters to be attached to
            'algorithm' when 'algorithm' is instanced. Defaults to an empty
            dictionary.

    """
    name: str
    step: Optional[str] = None
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = field(
        default_factory = lambda: 'simplify.core')
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
                    raise ImportError(' '.join(
                        [getattr(self, component), 'is neither in',
                            self.module, 'nor', self.default_module]))
        # If 'component' is not a string, it is returned as is.
        else:
            return getattr(self, component)