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
    from pathos.multiprocessing import ProcessPool as Pool
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

    def _finalize_chapters(self, book: 'Book', data: 'Dataset') -> 'Book':
        for chapter in book.chapters:
            for step, techniques in chapter.techniques.items():
                for technique in listify(techniques):
                    if technique is not 'none':
                        technique = self._add_conditionals(
                            book = book,
                            technique = technique,
                            data = data)
                        technique = self._add_data_dependents(
                            technique = technique,
                            data = data)
                        technique = self._add_parameters_to_algorithm(
                            technique = technique)
        return book

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
            data: Union['Dataset']) -> 'Chapter':
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
            data = self._iterate_techniques(
                techniques = techniques, 
                data = data)
        setattr(chapter, 'data', data)
        return chapter

    def _iterate_techniques(self,
                techniques: Union[List['Technique'], 'Technique'],
                data: Union['Dataset', 'Book']) -> Union['Dataset', 'Book']:
            for technique in listify(techniques):
                if not technique in ['none', None]:
                    data = technique.apply(data = data)
            return data
        
    """ Core siMpLify Methods """

    def apply(self,
            book: 'Book',
            data: 'Dataset',
            library: 'Repository',
            **kwargs) -> ('Book', 'Dataset'):
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
        self._finalize_chapters(book = book, data = data)
        if self.parallelize:
            self.parallelizer.apply_chapters(
                data = data,
                method = self._iterate_chapter)
        else:
            new_chapters = []
            for i, chapter in enumerate(book.chapters):
                if self.verbose:
                    print('Applying chapter', str(i + 1), 'to data')
                new_chapters.append(self._iterate_chapter(
                    chapter = chapter,
                    data = data))
            book.chapters = new_chapters
        return book, data


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
