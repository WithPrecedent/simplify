
"""
.. module:: workers
:synopsis: applies siMpLify objects to data
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.book import Contents
from simplify.core.base import BookOutline
from simplify.core.utilities import listify
from simplify.core.validators import DataConformer


@dataclass
class Worker(object):
    """Applies methods to siMpLify class instances.

    Args:
        project ('Project'): a related director class instance.

    """
    project: 'Project'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets initial index location for iterable.
        self._position = 0
        return self

    """ Private Methods """

    def _apply_gpu(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> NotImplementedError:
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Raises:
            NotImplementedError: until dynamic GPU support is added.

        """
        raise NotImplementedError(
            'GPU support outside of modeling is not yet supported')

    def _apply_multi_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients',
                'SimpleManuscript']] = None) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        with Pool() as pool:
            pool.imap(manuscript.apply, data)
        pool.close()
        return self

    def _apply_single_core(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        manuscript.apply(data = data, **kwargs)
        return self

    def _apply_technique(technique: )

    """ Core siMpLify Methods """

    def apply(self,
            manuscript: 'SimpleManuscript',
            data: Optional[Union['Ingredients', 'SimpleManuscript']] = None,
            **kwargs) -> 'SimpleManuscript':
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's options' 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        if self.parallelize and not kwargs:
            self._apply_multi_core(
                manuscript = manuscript,
                data = data)
        else:
            self._apply_single_core(
                manuscript = manuscript,
                data = data,
                **kwargs)
        return manuscript

    def apply(self,
            book: 'Book',
            data: Optional[Union['Ingredients', 'Book']] = None,
            **kwargs) -> Union['Ingredients', 'Book']:
        """Applies objects in 'manuscript' to 'data'

        Args:
            manuscript ('SimpleManuscript'): siMpLify class instance to be
                modified.
            data (Optional[Union['Ingredients', 'SimpleManuscript']]): an
                Ingredients instance containing external data or a published
                SimpleManuscript. Defaults to None.
            kwargs: any additional parameters to pass to a related
                SimpleManuscript's options' 'apply' method.

        Returns:
            manuscript ('SimpleManuscript'): siMpLify class instance with
                modifications made.

        """
        for chapter in book:
            for step, technique in chapter:
                data = technique.apply(data = data,**kwargs)
        return book


    """ Core siMpLify Methods """

    def _add_data_dependents(self, data: object) -> None:
        """Completes parameter dictionary by adding data dependent parameters.

        Args:
            data (object): data object with attributes for data dependent
                parameters to be added.

        Returns:
            parameters with any data dependent parameters added.

        """
        if self.outline.data_dependents is not None:
            for key, value in self.outline.data_dependents.items():
                try:
                    self.parameters.update({key, getattr(data, value)})
                except KeyError:
                    print('no matching parameter found for', key, 'in',
                        data.name)
        return self

    def _add_parameters_to_algorithm(self) -> None:
        """Attaches 'parameters' to the 'algorithm'."""
        try:
            self.algorithm = self.algorithm(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm(self.parameters)
            except AttributeError:
                pass
        except TypeError:
            pass
        return self

    """ Public Methods """

    def draft(self):
        """Creates 'algorithm' and 'outline' attributes."""
        # Injects attributes from Idea instance, if values exist.
        self = self.library.idea.apply(instance = self)
        self.outline = self.library[self.technique]
        self.algorithm = self.outline.load()
        return self

    def publish(self) -> None:
        """Finalizes 'algorithm' and 'parameters' attributes."""
        self.algorithm = self.algorithm.publish()
        self.parameters = self.parameters.publish()
        return self

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

    @DataConformer
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
            error = ' '.join([self.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @DataConformer
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
            error = ' '.join([self.name,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @DataConformer
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
            error = ' '.join([self.name, 'algorithm has no transform method'])
            raise AttributeError(error)



# @dataclass
# class Book(Contents):
#     """Stores and iterates Chapters.

#     Args:
#         project ('Project'): current associated project.

#     Args:
#         project ('Project'): associated Project instance.
#         options (Optional[Dict[str, 'BookOutline']]): Contents instance or
#             a Contents-compatible dictionary. Defaults to an empty
#             dictionary.
#         steps (Optional[Union[List[str], str]]): steps of key(s) to iterate in
#             'options'. Also, if not reset by the user, 'steps' is used if the
#             'default' property is accessed. Defaults to an empty list.

#     """
#     project: 'Project' = None
#     options: Optional[Dict[str, 'BookOutline']] = field(default_factory = dict)
#     steps: Optional[Union['SimpleSequence', List[str], str]] = field(
#         default_factory = list)
#     name: Optional[str] = None
#     chapter_type: Optional['Chapter'] = None
#     iterable: Optional[str] = 'chapters'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'book'

#     def __post_init__(self) -> None:
#         """Initializes class instance attributes."""
#         # Sets default 'name' attribute if none exists.
#         if self.name is None:
#             self.name = self.__class__.__name__.lower()
#         # Calls parent method for initialization.
#         super().__post_init__()
#         return self

#     """ Core SiMpLify Methods """

#     def apply(self,
#             options: Optional[Union[List[str], Dict[str, Any], str]] = None,
#             data: Optional[Union['Ingredients', 'Book']] = None,
#             **kwargs) -> Union['Ingredients', 'Book']:
#         """Calls 'apply' method for published option matching 'step'.

#         Args:
#             options (Optional[Union[List[str], Dict[str, Any], str]]): ordered
#                 options to be applied. If none are passed, the 'published' keys
#                 are used. Defaults to None
#             data (Optional[Union['Ingredients', 'Book']]): a siMpLify object for
#                 the corresponding 'options' to apply. Defaults to None.
#             kwargs: any additional parameters to pass to the options' 'apply'
#                 method.

#         Returns:
#             Union['Ingredients', 'Book'] is returned if data is passed;
#                 otherwise nothing is returned.

#         """
#         if isinstance(options, dict):
#             options = list(options.keys())
#         elif options is None:
#             options = self.default
#         self._change_active(new_active = 'applied')
#         for option in options:
#             if data is None:
#                 getattr(self, self.active)[option].apply(**kwargs)
#             else:
#                 data = getattr(self, self.active)[option].apply(
#                     data = data,
#                     **kwargs)
#             getattr(self, self.active)[option] = getattr(
#                 self, self.active)[option]
#         if data is None:
#             return self
#         else:
#             return data


# @dataclass
# class Chapter(Contents):
#     """Iterator for a siMpLify process.

#     Args:
#         book ('Book'): current associated Book
#         metadata (Optional[Dict[str, Any]], optional): any metadata about the
#             Chapter. Unless a subclass replaces it, 'number' is automatically a
#             key created for 'metadata' to allow for better recordkeeping.
#             Defaults to an empty dictionary.

#     """
#     book: 'Book' = None
#     name: Optional[str] = None
#     iterable: Optional[str] = 'book.steps'
#     metadata: Optional[Dict[str, Any]] = field(default_factory = dict)
#     file_format: Optional[str] = 'pickle'
#     export_folder: Optional[str] = 'chapter'

#     def __post_init__(self) -> None:
#         super().__post_init__()
#         return self

#     """ Private Methods """

#     def _apply_extra_processing(self) -> None:
#         """Extra actions to take."""
#         return self

#     """ Core siMpLify Methods """

#     def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
#         """Applies stored 'options' to passed 'data'.

#         Args:
#             data (Optional[Union['Ingredients', 'SimpleManuscript']]): a
#                 siMpLify object for the corresponding 'step' to apply. Defaults
#                 to None.
#             kwargs: any additional parameters to pass to the step's 'apply'
#                 method.

#         """
#         if data is not None:
#             self.ingredients = data
#         for step in getattr(self, self.iterable):
#             self.book[step].apply(data = self.ingredients, **kwargs)
#             self._apply_extra_processing()
#         return self