"""
.. module:: workers
:synopsis: controller for siMpLify projects
:editor: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.base import SimpleCatalog
from simplify.core.book import Book
from simplify.core.base import SimpleProgression
from simplify.core.editors import Author
from simplify.core.editors import Publisher
from simplify.core.scholars import Scholar
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify


@dataclass
def Workers(SimpleProgression):

    options: Optional[Dict[str, Any]] = field(default_factory = dict)
    order: Optional[List[str]] = field(default_factory = list)

    """ Other Dunder Methods """

    def __add__(self, other: 'Worker') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Worker': a 'Worker' instance.

        """
        self.add(worker = other)
        return self

    def __iadd__(self, other: 'Worker') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Worker': a 'Worker' instance.

        """
        self.add(worker = other)
        return self

    """ Public Methods """

    def add(self, worker: 'Worker', key: Optional[str] = None) -> None:
        """Combines arguments with 'options'.

        Args:
            worker ('Worker'): a 'Worker' instance.
            key (Optional[str]): key name to link to 'worker'. If not passed,
                 the 'name' attribute of 'worker' will be used.

        """
        if key:
            self.options[key] = worker
        else:
            self.options[worker.name] = worker
        return self


@dataclass
def Worker(SimpleOutline):

    name: str
    module: str
    book: Optional['Book'] = Book
    chapters: Optional['SimpleProgression'] = SimpleProgression
    author: Optional['Author'] = Author
    publisher: Optional['Publisher'] = Publisher
    scholar: Optional['Scholar'] = Scholar
    steps: Optional['SimpleProgression'] = SimpleProgression
    techniques: Optional['SimpleCatalog'] = SimpleCatalog


def create_workers(
        workers: Union['Workers', List[str], str],
        project: 'Project') -> 'Workers':
    """Creates or validates 'workers'.

    Args:
        workers: (Union['Workers', List[str], str]): either a 'Workers' instance,
            a list of workers, or a single worker.
        project ('Project'): a related 'Project' instance with a
            'default_workers' dictionary.

    Returns:
        'Workers': an instance derived from 'workers' and/or 'project'.

    """
    if isinstance(workers, Workers):
        return workers
    elif isinstance(workers, (list, str)):
        new_workers = {}
        for worker in listify(workers):
            new_workers[worker] = project.default_workers[worker]
        return Workers(options = new_workers)
    else:
        return Workers(options = project.default_workers)