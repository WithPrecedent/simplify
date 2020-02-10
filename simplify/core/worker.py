"""
.. module:: worker
:synopsis: applies collections of techniques to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
import multiprocessing as mp
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
try:
    from pathos.multiprocessing import PlaningPool as Pool
except ImportError:
    from multiprocessing import Pool

from simplify.core.repository import Repository
from simplify.core.repository import Plan
from simplify.core.utilities import listify
from simplify.core.validators import DataValidator


@dataclass
class Worker(object):
    """Base class for applying Book instances to data.

    Args:
        idea ('Idea'): an instance with project settings.
        task ('Task'): instance with information needed to create a Book
            instance.

    """
    idea: 'Idea'
    task: Optional['Task'] = None

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self = self.idea.apply(instance = self)
        self.parallelizer = Parallelizer(idea = self.idea)
        return self

    """ Private Methods """

    def _add_conditionals(self,
            book: 'Book',
            technique: 'Technique',
            data: Union['Dataset', 'Book']) -> 'Technique':
        """Adds any conditional parameters to a 'Technique' instance.

        Args:
            book ('Book'): Book instance with algorithms to apply to 'data'.
            technique ('Technique'): instance with parameters which can take
                new conditional parameters.
            data (Union['Dataset', 'Book']): a data source which might
                contain information for condtional parameters.

        Returns:
            'technique': instance with any conditional parameters added.

        """
        try:
            if technique is not None:
                return getattr(book, '_'.join(
                    ['_add', technique.name, 'conditionals']))(
                        technique = technique,
                        data = data)
        except AttributeError:
            return technique

    def _add_data_dependents(self,
            technique: 'Technique',
            data: Union['Dataset', 'Book']) -> 'Technique':
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            technique ('Technique'): instance with information about data
                dependent parameters to add.
            data (Union['Dataset', 'Book']): a data source which contains
                'data_dependent' variables.

        Returns:
            'Technique': with any data dependent parameters added.

        """
        if technique is not None and technique.data_dependents is not None:
            for key, value in technique.data_dependents.items():
                try:
                    technique.parameters.update({key, getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in data')
        return technique

    def _add_parameters_to_algorithm(self,
            technique: 'Technique') -> 'Technique':
        """Instances 'algorithm' with 'parameters' in 'technique'.

        Args:
            technique ('Technique'): with completed 'algorith' and 'parameters'.

        Returns:
            'Technique': with 'algorithm' instanced with 'parameters'.

        """
        if technique is not None:
            try:
                technique.algorithm = technique.algorithm(
                    **technique.parameters)
            except AttributeError:
                try:
                    technique.algorithm = technique.algorithm(
                        technique.parameters)
                except AttributeError:
                    technique.algorithm = technique.algorithm()
            except TypeError:
                try:
                    technique.algorithm = technique.algorithm()
                except TypeError:
                    pass
        return technique

    def _iterate_chapter(self,
            book: 'Book',
            chapter: 'Chapter',
            data: Union['Dataset', 'Book']) -> 'Chapter':
        """Iterates a single chapter and applies 'techniques' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'techniques' to apply to 'data'.
            data (Union['Dataset', 'Book']): object for 'chapter'
                'techniques' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        for step, techniques in chapter.techniques.items():
            if techniques is not None:
                for technique in listify(techniques, default_empty = True):
                    technique = self._add_conditionals(
                        book = book,
                        technique = technique,
                        data = data)
                    technique = self._add_data_dependents(
                        technique = technique,
                        data = data)
                    technique = self._add_parameters_to_algorithm(
                        technique = technique)
                    data = technique.apply(data = data)
        if book.alters_data:
            setattr(chapter, data.name, data)
        return chapter

    """ Core siMpLify Methods """

    def apply(self,
            book: 'Book',
            data: Optional[Union['Dataset', 'Book']] = None,
            **kwargs) -> Union['Dataset', 'Book']:
        """Applies objects in 'book' to 'data'.

        Args:
            book ('Book'): Book instance with algorithms to apply to 'data'.
            data (Optional[Union['Dataset', 'Book']]): a data source for
                the 'book' methods to be applied.
            kwargs: any additional parameters to pass to a related
                Book's options' 'apply' method.

        Returns:
            Union['Dataset', 'Book']: data object with modifications
                possibly made.

        """
        if self.parallelize:
            self.parallelizer.apply_chapters(
                book = book,
                data = data,
                method = self._iterate_chapter)
        else:
            new_chapters = []
            for chapter in book.chapters:
                new_chapters.append(self._iterate_chapter(
                    book = book,
                    chapter = chapter,
                    data = data))
            book.chapters = new_chapters
        return book



# @dataclass
# class DataProxies(MutableMapping):

#     dataset: 'Dataset'
#     test_suffixes: Dict[str, str] = field(default_factory = dict)
#     train_suffixes: Dict[str, str] = field(default_factory = dict)

#     def __post_init__(self) -> None:
#         if not self.test_suffixes:
#             self.test_suffixes = {
#                 'unsplit': None,
#                 'xy': '',
#                 'train_test': '_test',
#                 'train_val': '_test',
#                 'full': '_train'}
#         if not self.train_suffixes:
#             self.train_suffixes = {
#                 'unsplit': None,
#                 'xy': '',
#                 'train_test': '_train',
#                 'train_val': '_train',
#                 'full': '_train'}
#         return self

#     """ Required ABC Methods """

#     def __getitem__(self, key: str) -> 'Data':
#         """Returns 'Data' based upon current 'state'.

#         Args:
#             key (str): name of key in 'dataset'.

#         Returns:
#             'Data': an 'Data' instance stored in 'dataset'
#                 based on 'state' in 'dataset'

#         Raises:
#             ValueError: if access to train or test data is sought before data
#                 has been split.

#         """
#         try:
#             contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
#             if getattr(self, dictionary)[self.dataset.state] is None:
#                 raise ValueError(''.join(['Train and test data cannot be',
#                     'accessed until data is split']))
#             else:
#                 new_key = ''.join(
#                     [key[0], getattr(self, dictionary)[self.dataset.state]])
#                 return self.dataset.dataset[new_key]
#         except TypeError:
#             return self.dataset.dataset[key]

#     def __setitem__(self, key: str, value: 'Data') -> None:
#         """Sets 'key' to 'Data' based upon current 'state'.

#         Args:
#             key (str): name of key to set in 'dataset'.
#             value ('Data'): 'Data' instance to be added to
#                 'dataset'.

#         """
#         try:
#             contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
#             new_key = ''.join(
#                 [key[0], getattr(self, dictionary)[self.dataset.state]])
#             self.dataset.dataset[new_key] = value
#         except ValueError:
#             self.dataset.dataset[key] = value

#     def __delitem__(self, key: str) -> None:
#         """Deletes 'key' in the 'dataset' dictionary.

#         Args:
#             key (str): name of key in the 'dataset' dictionary.

#         """
#         try:
#             contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
#             new_key = ''.join(
#                 [key[0], getattr(self, dictionary)[self.dataset.state]])
#             self.dataset.dataset[new_key] = value
#         except ValueError:
#             try:
#                 del self.dataset.dataset[key]
#             except KeyError:
#                 pass

#     def __iter__(self) -> NotImplementedError:
#         raise NotImplementedError('DataProxies does not implement an iterable')

#     def __len__(self) -> int:
#         raise NotImplementedError('DataProxies does not implement length')



@dataclass
class Parallelizer(object):
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
        with Pool() as pool:
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
        with Pool() as pool:
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
            method (Callable): callable method or function to apply to 'data'.

        Returns:
            'Data': with 'method' applied.

        """
        dfs = np.array_split(data.data, mp.cpu_count(), axis = 0)
        pool = Pool()
        data.data = np.vstack(pool.map(method, dfs))
        pool.close()
        pool.join()
        pool.clear()
        return data
