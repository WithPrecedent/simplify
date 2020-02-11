
"""
.. module:: technique
:synopsis: siMpLify algorithms and parameters
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from inspect import signature
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y

from simplify.core.definitions import Outline


@dataclass
class TechniqueOutline(Outline):
    """Contains settings for creating a Technique instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify object).
        algorithm: str = None
        default: Optional[Dict[str, Any]] = field(default_factory = dict)
        required: Optional[Dict[str, Any]] = field(default_factory = dict)
        runtime: Optional[Dict[str, str]] = field(default_factory = dict)
        selected: Optional[Union[bool, List[str]]] = False
        conditional: Optional[bool] = False
        data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)

    """
    name: str
    module: str
    algorithm: str = None
    default: Optional[Dict[str, Any]] = field(default_factory = dict)
    required: Optional[Dict[str, Any]] = field(default_factory = dict)
    runtime: Optional[Dict[str, str]] = field(default_factory = dict)
    selected: Optional[Union[bool, List[str]]] = False
    data_dependent: Optional[Dict[str, str]] = field(default_factory = dict)
    fit_method: Optional[str] = field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = field(
        default_factory = lambda: 'transform')


def numpy_shield(callable: Callable) -> Callable:
    """
    """
    @wraps(callable)
    def wrapper(*args, **kwargs):
        call_signature = signature(callable)
        arguments = dict(call_signature.bind(*args, **kwargs).arguments)
        try:
            x_columns = list(arguments['x'].columns.values)
            result = callable(*args, **kwargs)
            if isinstance(result, np.ndarray):
                result = pd.DataFrame(result, columns = x_columns)
        except KeyError:
            result = callable(*args, **kwargs)
        return result
    return wrapper


@dataclass
class Technique(Container):
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
        technique (Optional[str]): name of particular technique to be used. It
            should correspond to a key in the related 'book' instance. Defaults
            to None.

    """
    name: Optional[str] = None
    technique: Optional[str] = None
    algorithm: Optional[object] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory = dict)
    parameter_space: Optional[Dict[str, List[Union[int, float]]]] = field(
        default_factory = dict)
    data_dependents: Optional[Dict[str, str]] = field(default_factory = dict)
    fit_method: Optional[str] = field(default_factory = lambda: 'fit')
    transform_method: Optional[str] = field(
        default_factory = lambda: 'transform')

    """ Required ABC Methods """

    def __contains__(self, key: str) -> bool:
        """Returns whether 'attribute' is the 'technique'.

        Args:
            key (str): name of item to check.

        Returns:
            bool: whether the 'key' is equivalent to 'technique'.

        """
        return item == self.technique

    """ Private Methods """

    def _apply_once(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        print('test fit method', self.fit)
        if self.fit_method:
            self.fit(x = x, y = y)
        if self.transform_method:
            x = self.transform(x = x, y = y)
        return x

    """ Core siMpLify Methods """

    def apply(self, data: 'Dataset') -> 'Dataset':
        if data.stages.current in ['full']:
            data.x = self._apply_once(x = data.x, y = data.y)
        else:
            data.x_train = self._apply_once(x = data.x_train, y = data.y_train)
            data.x_test = self._apply_once(x = data.x_test, y = data.y_test)
        return data

    """ Scikit-Learn Compatibility Methods """

    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Raises:
            AttributeError if no 'fit' method exists for 'technique'.

        """
        print('test fit data', y)
        print('test fit technique', self.technique)
        x, y = check_X_y(X = x, y = y, accept_sparse = True)
        try:
            if y is None:
                getattr(self.algorithm, self.fit_method)(x)
            else:
                getattr(self.algorithm, self.fit_method)(x, y)
        except AttributeError:
            raise AttributeError(' '.join(
                [self.technique, 'has no fit method']))
        return self

    @numpy_shield
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or dataset is not passed to
                the method.

        """
        self.fit(x = x, y = y, data = dataset)
        return self.transform(x = x, y = y)

    @numpy_shield
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if self.transform_method:
            return getattr(self.algorithm, self.transform_method)(x)
        else:
            raise AttributeError(' '.join(
                [self.technique, 'has no transform method']))