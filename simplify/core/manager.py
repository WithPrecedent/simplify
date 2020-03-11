"""
.. module:: manager
:synopsis: management of project workers
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from inspect import isclass
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.base import SimpleComponent
from simplify.core.creators import Publisher
from simplify.core.library import Book
from simplify.core.library import Chapter
from simplify.core.library import Technique
from simplify.core.repository import Repository
from simplify.core.scholar import Scholar
from simplify.core.utilities import listify


@dataclass
class Manager(MutableMapping):
    """Stores and manages 'Worker' instances for a 'Project'.

    Args:
        workers (Optional[Union[Dict[str, 'Worker'], 'Manager', List[str]]]):
            dictionary with keys as strings and values of 'Worker' instances, a
            'Manager' instance, or a list of workers corresponding to keys in
            'default_packages' to use. Defaults to an empty dictionary. If
            nothing is provided, Project attempts to construct workers from
            'idea' and 'default_packages'.
        idea (Optional['Idea']): shared project configuration settings.

    """
    workers: Optional[Union[Dict[str, 'Worker'], List[str]]] = field(
        default_factory = dict)
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Initializes 'workers' and 'default_packages'."""
        self.workers = self._initialize_workers(workers = self.workers)
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            packages: Union[Dict[str, Union['Worker', 'Package']],
                'Repository',
                'Manager'],
            idea: 'Idea') -> 'Manager':
        """Creates a 'Manager' instance from 'packages'.

        Args:
            workers (Optional[Union[Dict[str, 'Worker'], 'Repository',
                'Manager']]): 'Manager' instance or a MutableMapping containing
                'Worker' instances. Defaults to None.

        Returns:
            'Manager': instance, properly configured.

        Raises:
            TypeError: if 'workers' are not a MutableMapping or if the values
                in that Mutable Mapping are not 'Worker' instances.

        """
        if isinstance(packages, Manager):
            return packages
        elif isinstance(packages, MutableMapping):
            if all(isinstance(value, Worker) for value in packages.values()):
                return cls(workers = packages, idea = idea)
            else:
                try:
                    workers = {}
                    for key, package in packages.items():
                        workers[key] = package.load()
                    return cls(workers = workers, idea = idea)
                except AttributeError:
                    raise TypeError(
                        'workers values must be Worker or Package type')
        else:
            raise TypeError('workers must be dict, Repository, or Manager type')

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> 'Book':
        """Returns key from 'workers'.
        Args:
            key (str): key to item in 'workers'.

        Returns:
            'Book': from 'workers'.

        """
        return self.workers[key]

    def __setitem__(self, key: str, value: 'Book') -> None:
        """Sets 'key' in 'workers' to 'value'.

        Args:
            key (str): key to item in 'workers' to set.
            value ('Book'): instance to place in 'workers'.

        """
        self.workers[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'workers'.

        Args:
            key (str): key in 'workers'.

        """
        try:
            del self.workers[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'workers'.

        Returns:
            Iterable: of 'workers'.

        """
        return iter(self.workers)

    def __len__(self) -> int:
        """Returns length of 'workers'.

        Returns:
            int: length of 'workers'.
        """
        return len(self.workers)

    """ Core siMpLify Methods """

    def add(self, worker: 'Worker') -> None:
        """Adds 'worker' to 'workers'.

        Args:
            worker ('Worker'): an instance to be added.

        Raises:
            ValueError: if 'worker' is not a 'Worker' instance.

        """
        # Validates 'worker' type as 'Worker' before adding to 'workers'.
        if isinstance(worker, Worker):
            self.workers[worker.name] = worker
        else:
            raise ValueError(
                'worker must be a Worker instance to add to a Manager')
        return self

    """ Private Methods """

    def _initialize_workers(self,
            workers: Dict[str, 'Worker']) -> Dict[str, 'Worker']:
        """Instances 'Worker' subclasses stored in 'workers'.

        Args:
            workers (Dict[str, 'Worker']): stored Worker subclasses.

        Returns:
            Dict[str, 'Worker']: dictionary with instances added.

        """
        new_workers = {}
        for key, worker in workers.items():
            new_workers[key] = worker(idea = self.idea)
        return new_workers


@dataclass
class Worker(SimpleComponent):
    """Object construction instructions used by a Project instance.

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
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to 'Publisher'.
        scholar (Optional[str]): name of Scholar class in 'module' to load.
            Defaults to 'Scholar'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[str]): name of a 'Repository' instance with various
            options available to a particular 'Worker' instance. Defaults to
            an empty 'Repository'.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'books' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.
        idea (Optional['Idea']): shared project configuration settings.

    """
    name: Optional[str] = None
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')
    book: Optional[str] = field(default_factory = lambda: 'Book')
    chapter: Optional[str] = field(default_factory = lambda: 'Chapter')
    technique: Optional[str] = field(default_factory = lambda: 'Technique')
    publisher: Optional[str] = field(default_factory = lambda: 'Publisher')
    scholar: Optional[str] = field(default_factory = lambda: 'Scholar')
    steps: Optional[List[str]] = field(default_factory = list)
    options: Optional[str] = field(default_factory = Repository)
    data: Optional[str] = field(default_factory = lambda: 'dataset')
    import_folder: Optional[str] = field(default_factory = lambda: 'processed')
    export_folder: Optional[str] = field(default_factory = lambda: 'processed')
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        self._to_load = ['chapter', 'technique']
        self._to_instance = ['book', 'publisher', 'scholar', 'options']
        self.draft()
        return self

    """ Core siMpLify Methods """

    def outline(self) -> Dict[str, List[str]]:
        """Creates dictionary with techniques for each step.

        Returns:
            Dict[str, Dict[str, List[str]]]: dictionary with keys of steps and
                values of lists of techniques.

        """
        catalog = {}
        steps = self._get_settings(
            section = self.name,
            prefix = self.name,
            suffix = 'steps')
        for step in steps:
            catalog[step] = self._get_settings(
                section = self.name,
                prefix = step,
                suffix = 'techniques')
        return catalog

    def draft(self) -> None:
        for attribute in self._to_load + self._to_instance:
            setattr(self, attribute, self.load(attribute))
            if attribute in self._to_instance:
                if attribute in ['scholar', 'publisher']:
                    setattr(self, attribute, getattr(self, attribute)(
                        worker = self))
                else:
                    setattr(self, attribute, getattr(self, attribute)())
        return self

    """ Private Methods """

    def _get_settings(self,
            section: str,
            prefix: str,
            suffix: str) -> List[str]:
        """Returns settings from 'idea' based on 'name' and 'suffix'.

        Args:
            section (str): outer key name in 'idea' section.
            prefix (str); prefix for an inner key name.
            suffix (str): suffix to inner key name in 'idea'.

        Returns:
            List[str]: names of matching workers, steps, or techniques.

        """
        return listify(self.idea[section]['_'.join([prefix, suffix])])

