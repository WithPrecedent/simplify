"""
.. module:: repository
:synopsis: siMpLify base mapping classes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core import utilities


@dataclasses.dataclass
class Overview(base.SimpleRepository):
    """Stores outline of a siMpLify project.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().
        contents (Optional[Dict[str, Dict[str, List[str]]]]): dictionary
            storing the outline of a siMpLify project. Defaults to an empty
            dictionary.


    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Dict[str, List[str]]]] = dataclasses.field(
        default_factory = dict)

    """ Factory Method """

    @classmethod
    def create(cls, manager: 'Manager') -> 'Overview':
        """Creates an 'Overview' instance from 'workers'.

        Args:
            manager ('Manager'): an instance with stored 'workers'.

        Returns:
            'Overview': instance, properly configured.

        """
        contents = {
            name: worker.outline() for name, worker in manager.workers.items()}
        return cls(contents = contents)

    """ Required ABC Methods """

    def __getitem__(self,
            key: Union[str, Tuple[str, str]]) -> Union[
                Dict[str, List[str]], List[str]]:
        """Returns key from 'contents'.

        Args:
            key (Union[str, Tuple[str, str]]): key to item in 'contents'. If
                'key' is a tuple, the method attempts to return [key[0]][key[1]]
                from 'contents'.

        Returns:
            Union[Dict[str, List[str]], List[str]]]:man overview of either
                one package of a siMpLify project (if 'key' is a str) or one
                step in one package of a siMpLify project (if 'key' is a tuple).

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            return self.contents[key]
        elif isinstance(key, tuple):
            return self.contents[key[0]][key[1]]
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')

    def __setitem__(self,
                key: Union[str, Tuple[str, str]],
                value: Union[Dict[str, List[str]], List[str]]) -> None:
        """Sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[str, Tuple[str, str]]): key to item in 'contents' to set.
            value (Union[Dict[str, List[str]], List[str]]): dictionary or list
                to place in 'contents'.

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            self.contents[key] = value
        elif isinstance(key, tuple):
            if key[0] not in self.contents:
                self.contents[key[0]] = {}
            self.contents[key[0]][key[1]] = value
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')
        return self

    def __delitem__(self, key: Union[str, Tuple[str, str]]) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Union[str, Tuple[str, str]]): key in 'contents'.

        Raises:
            TypeError: if 'key' is neither a str nor tuple type.

        """
        if isinstance(key, str):
            try:
                del self.contents[key]
            except KeyError:
                pass
        elif isinstance(key, tuple):
            try:
                del self.contents[key[0]][key[1]]
            except KeyError:
                pass
        else:
            raise TypeError(
                f'{self.__class__.__name__} requires str or tuple type')
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'contents'.

        Returns:
            Iterable: of 'contents'.

        """
        return iter(self.contents)

    def __len__(self) -> int:
        """Returns length of 'contents'.

        Returns:
            int: length of 'contents'.
        """
        return len(self.contents)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return (f'Project {self.identification}:',
                f'{self.contents}')


@dataclasses.dataclass
class Manager(base.SimpleRepository):
    """Stores and manages 'Worker' instances for a 'Project'.

    Args:
        name (Optional[str]): designates the name of the class instance used
            for internal referencing throughout siMpLify. If the class instance
            needs settings from the shared 'Idea' instance, 'name' should match
            the appropriate section name in that 'Idea' instance. When
            subclassing, it is a good idea to use the same 'name' attribute as
            the base class for effective coordination between siMpLify classes.
            'name' is used instead of __class__.__name__ to make such
            subclassing easier. Defaults to None or __class__.__name__.lower().
        contents (Optional[str, Any]): stored dictionary. Defaults to an empty
            dictionary.
        wildcards (Optional[List[str]]): a list of wildcard keys which return
            lists of values. Defaults to ['all', 'default', 'none'].
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        workers (Optional[Union[Dict[str, 'Worker'], 'Manager', List[str]]]):
            dictionary with keys as strings and values of 'Worker' instances, a
            'Manager' instance, or a list of workers corresponding to keys in
            'default_packages' to use. Defaults to an empty dictionary. If
            nothing is provided, Project attempts to construct workers from
            'idea' and 'default_packages'.
        idea (Optional['Idea']): shared project configuration settings.

    """
    name: Optional[str] = None
    contents: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)
    wildcards: Optional[List[str]] = dataclasses.field(
        default_factory = lambda: ['all', 'default', 'none'])
    workers: Optional[Union[Dict[str, 'Worker'], List[str]]] = dataclasses.field(
        default_factory = dict)
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes 'workers' and 'default_packages'."""
        super().__post_init()
        self.workers = {k: w(idea = self.idea) for k, w in workers.items()}
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            packages: Union[
                Dict[str,
                     Union['Worker', 'Package']],
                'SimpleRepository', 'Manager'],
            idea: 'Idea') -> 'Manager':
        """Creates a 'Manager' instance from 'packages'.

        Args:
            workers (Optional[Union[Dict[str, 'Worker'], 'SimpleRepository',
                'Manager']]): 'Manager' instance or a collections.abc.MutableMapping containing
                'Worker' instances. Defaults to None.
            idea ('Idea'): shared project configuration settings.


        Returns:
            'Manager': instance, properly configured.

        Raises:
            TypeError: if 'workers' are not a collections.abc.MutableMapping or if the values
                in that Mutable Mapping are not 'Worker' instances.

        """
        if isinstance(packages, Manager):
            return packages
        elif isinstance(packages, collections.abc.MutableMapping):
            if all(isinstance(value, Worker) for value in packages.values()):
                return cls(workers = packages, idea = idea)
            else:
                try:
                    workers = {key: package.load() for k, p in packages.items()}
                    return cls(workers = workers, idea = idea)
                except AttributeError:
                    raise TypeError(
                        'workers values must be Worker or Package type')
        else:
            raise TypeError(
                'workers must be dict, SimpleRepository, or Manager type')

    """ Core siMpLify Methods """

    def add(self, worker: 'Worker') -> None:
        """Adds 'worker' to 'workers'.

        Args:
            worker ('Worker'): an instance to be added.

        Raises:
            ValueError: if 'worker' is not a 'Worker' instance.

        """
        # Validates 'worker' type as 'Worker' before adding to 'workers'.
        if isinstance(worker, Worker) or issubclass(worker, Worker):
            self.workers[worker.name] = worker
        else:
            raise ValueError('worker must be a Worker or subclass instance')
        return self


@dataclasses.dataclass
class Library(base.SimpleRepository):
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
    name: Optional[str] = dataclasses.field(default_factory = utilities.datetime_string)
    contents: Optional[Dict[str, Any]] = dataclasses.field(default_factory = dict)
    defaults: Optional[List[str]] = dataclasses.field(default_factory = list)
    catalog: Optional[Dict[str, Dict[str, List[str]]]] = dataclasses.field(
        default_factory = dict)
    books: Optional[Dict[str, 'Book']] = dataclasses.field(default_factory = dict)

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