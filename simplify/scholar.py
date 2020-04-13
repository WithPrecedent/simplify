"""
.. module:: scholar
:synopsis: generic siMpLify application classes
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import collections.abc
import dataclasses
import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pathos.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

import numpy as np
import pandas as pd

from simplify.core import base
from simplify.core import utilities


@dataclasses.dataclass
class Parallelizer(SimpleHandler):
    """Applies techniques using one or more CPU or GPU cores.

    Args:
        idea ('Idea'): shared 'Idea' instance with project settings.

    """
    idea: 'Idea'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        return self

    """ Private Methods """

    def _apply_gpu(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies objects in 'book' to 'data'

        Args:
            book ('Book'): siMpLify class instance to be
                modified.
            data (Optional[Union['Dataset', 'Book']]): an
                Dataset instance containing external data or a published
                Book. Defaults to None.
            kwargs: any additional parameters to pass to a related
                Book's 'apply' method.

        Raises:
            NotImplementedError: until dynamic GPU support is added.

        """
        raise NotImplementedError(
            'GPU support outside of modeling is not yet supported')

    def _apply_multi_core(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data' using multiple CPU cores.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Dataset', 'Book']): an instance containing data to
                be modified.
            method (Callable): method to parallelize.

        Returns:
            'Book': with its iterable applied to data.

        """
        with mp.Pool() as pool:
            pool.starmap(method, arguments)
        pool.close()
        return self

    """ Core siMpLify Methods """

    def apply_chapters(self,
            book: 'Book',
            data: Union['Dataset', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data'.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Dataset', 'Book']): an instance containing data to
                be modified.
            method (Callable): method to parallelize.

        Returns:
            'Book': with its iterable applied to data.

        """
        arguments = []
        for key, chapter in book.chapters.items():
            arguments.append((chapter, data))
        results = []
        chapters_keys = list(book.chapters.keys())
        with mp.Pool() as pool:
            results.append[pool.map(method, arguments)]
        pool.close()
        pool.join()
        pool.clear()
        book.chapters = dict(zip(chapters_keys, results))
        return book

    def apply_data(self,
            data: 'Data',
            method: Callable) -> 'Data':
        """Applies 'method' to 'data' across several cores.

        Args:
            data ('Data'): instance with a stored pandas DataFrame.
            method (Callable): process method or function to apply to 'data'.

        Returns:
            'Data': with 'method' applied.

        """
        dfs = np.array_split(data.data, mp.cpu_count(), axis = 0)
        pool = mp.Pool()
        data.data = np.vstack(pool.map(method, dfs))
        pool.close()
        pool.join()
        pool.clear()
        return data
