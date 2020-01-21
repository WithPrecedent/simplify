"""
.. module:: scholar
:synopsis: applies collections of techniques to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    from pathos.multiprocessing import SequenceingPool as Pool
except ImportError:
    from multiprocessing import Pool

from simplify.core.repository import Repository
from simplify.core.repository import Sequence
from simplify.core.utilities import listify
from simplify.core.validators import DataValidator


@dataclass
class Scholar(object):
    """Base class for applying Book instances to data.

    Args:
        project ('Project'): a related Project instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self.parallelizer = Parallelizer(project = self.project)
        return self

    """ Private Methods """

    def _add_conditionals(self,
            book: 'Book',
            technique: 'Technique',
            data: Union['Ingredients', 'Book']) -> 'Technique':
        """Adds any conditional parameters to a 'Technique' instance.

        Args:
            book ('Book'): Book instance with algorithms to apply to 'data'.
            technique ('Technique'): instance with parameters which can take
                new conditional parameters.
            data (Union['Ingredients', 'Book']): a data source which might
                contain information for condtional parameters.

        Returns:
            'technique': instance with any conditional parameters added.

        """
        try:
            return getattr(book, '_'.join(
                ['_add', technique.name, 'conditionals']))(
                    technique = technique,
                    data = data)
        except AttributeError:
            return technique

    def _add_data_dependents(self,
            technique: 'Technique',
            data: Union['Ingredients', 'Book']) -> 'Technique':
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            technique ('Technique'): instance with information about data
                dependent parameters to add.
            data (Union['Ingredients', 'Book']): a data source which contains
                'data_dependent' variables.

        Returns:
            'Technique': with any data dependent parameters added.

        """
        if technique.data_dependents is not None:
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
        try:
            technique.algorithm = technique.algorithm(**technique.parameters)
        except AttributeError:
            try:
                technique.algorithm = technique.algorithm(technique.parameters)
            except AttributeError:
                technique.algorithm = technique.algorithm()
        except TypeError:
            technique.algorithm = technique.algorithm()
        return self

    def _iterate_chapter(self,
            book: 'Book',
            chapter: 'Chapter',
            data: Union['Ingredients', 'Book']) -> 'Chapter':
        """Iterates a single chapter and applies 'techniques' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'techniques' to apply to 'data'.
            data (Union['Ingredients', 'Book']): object for 'chapter'
                'techniques' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        for step, technique in chapter.techniques.items():
            instance = book.techniques[step][technique]
            instance = self._add_conditionals(
                book = book,
                technique = instance,
                data = data)
            instance = self._add_data_dependents(
                technique = instance,
                data = data)
            self._add_parameters_to_algorithm(technique = instance)
            data = self._apply_technique(
                technique = technique,
                data = data,
                **kwargs)
        setattr(chapter, data.name, data)
        return chapter

    def _apply_technique(self,
            technique: 'Technique',
            data: Union['Book', 'Ingredients']) -> Union['Book', 'Ingredients']:

        return data

    """ Core siMpLify Methods """

    def apply(self,
            book: 'Book',
            data: Optional[Union['Ingredients', 'Book']] = None,
            **kwargs) -> Union['Ingredients', 'Book']:
        """Applies objects in 'book' to 'data'.

        Args:
            book ('Book'): Book instance with algorithms to apply to 'data'.
            data (Optional[Union['Ingredients', 'Book']]): a data source for
                the 'book' methods to be applied.
            kwargs: any additional parameters to pass to a related
                Book's options' 'apply' method.

        Returns:
            Union['Ingredients', 'Book']: data object with modifications
                possibly made.

        """
        if data is None:
            data = self.project.ingredients
        if self.project.parallelize:
            self.parallelizer.apply_chapters(
                book = book,
                data = data,
                method = self._iterate_chapter)
        else:
            new_chapters = {}
            for key, chapter in book.chapters.items():
                new_chapters[key] = self._iterate_chapter(
                    book = book,
                    chapter = chapter,
                    data = data)
        return book


@dataclass
class Parallelizer(object):
    """Applies techniques using one or more CPU or GPU cores.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        return self

    """ Private Methods """

    def _apply_gpu(self,
            book: 'Book',
            data: Union['Ingredients', 'Book'],
            method: Callable) -> 'Book':
        """Applies objects in 'book' to 'data'

        Args:
            book ('Book'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'Book']]): an
                Ingredients instance containing external data or a published
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
            data: Union['Ingredients', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data' using multiple CPU cores.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Ingredients', 'Book']): an instance containing data to
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
            data: Union['Ingredients', 'Book'],
            method: Callable) -> 'Book':
        """Applies 'method' to 'data'.

        Args:
            book ('Book'): siMpLify class instance with Chapter instances to
                parallelize.
            data (Union['Ingredients', 'Book']): an instance containing data to
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

    def apply_ingredient(self,
            data: 'Ingredient',
            method: Callable) -> 'Ingredient':
        """Applies 'method' to 'data' across several cores.

        Args:
            data ('Ingredient'): instance with a stored pandas DataFrame.
            method (Callable): callable method or function to apply to 'data'.

        Returns:
            'Ingredient': with 'method' applied.

        """
        dfs = np.array_split(ingredient.data, mp.cpu_count(), axis = 0)
        pool = Pool()
        ingredient.data = np.vstack(pool.map(method, dfs))
        pool.close()
        pool.join()
        pool.clear()
        return ingredient

    def apply(self, data: object, **kwargs) -> object:
        """

        """
        self._add_data_dependent(data = data)
        self._add_parameters_to_algorithm()
        try:
            self.algorithm.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
            setattr(
                data, ''.join(['x_', data.state]),
                self.algorithm.transform(getattr(
                    data, ''.join(['x_', data.state]))))
        except AttributeError:
            try:
                data = self.algorithm.apply(data = data)
            except AttributeError:
                pass
        return data

    """ Scikit-Learn Compatibility Methods """

    @DataValidator
    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if x is not None:
            try:
                if y is None:
                    self.algorithm.process.fit(x)
                else:
                    self.algorithm.process.fit(x, y)
            except AttributeError:
                error = ' '.join([self.design.name,
                                  'algorithm has no fit method'])
                raise AttributeError(error)
        elif data is not None:
            self.algorithm.process.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
        else:
            error = ' '.join([self.worker, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @DataValidator
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.algorithm.process.fit(x = x, y = y, data = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.algorithm.process.transform(x = x, y = y)
        elif data is not None:
            return self.algorithm.process.transform(data = ingredients)
        else:
            error = ' '.join([self.worker,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @DataValidator
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional[object] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if hasattr(self.algorithm.process, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.process.transform(x)
                else:
                    return self.algorithm.process.transform(x, y)
            elif data is not None:
                return self.algorithm.process.transform(
                    X = getattr(data, 'x_' + data.state),
                    Y = getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.worker, 'algorithm has no transform method'])
            raise AttributeError(error)

