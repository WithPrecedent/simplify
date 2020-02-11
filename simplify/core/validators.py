"""
.. module:: validators
:synopsis: validation and validator decoraters and methods
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC
from functools import wraps
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd

from simplify.core.idea import Idea
from simplify.core.dataset import Dataset
from simplify.core.inventory import Inventory
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


class SimpleValidator(ABC):
    """Base class decorator to convert arguments to proper types."""

    def __init__(self,
            callable: Callable,
            validators: Optional[Dict[str, Callable]] = None) -> None:
        """Sets initial validator options.

        Args:
            callable (Callable): wrapped method, function, or callable class.
            validators Optional[Dict[str, Callable]]: keys are names of
                parameters and values are functions to convert or validate
                passed arguments. Those functions must return a completed
                object and take only a single passed passed argument. Defaults
                to None.

        """
        self.callable = callable
        update_wrapper(self, self.callable)
        if self.validators is None:
            self.validators = {}
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self.apply(arguments = arguments)
            return self.callable(self, **arguments)
        return wrapper

    """ Core siMpLify Methods """

    def apply(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts values of 'arguments' to proper types.

        Args:
            arguments (Dict[str, Any]): arguments with values to be converted.

        Returns:
            Dict[str, Any]: arguments with converted values.

        """
        for argument, validator in self.validators.items():
            try:
                arguments[argument] = validator(arguments[argument])
            except KeyError:
                pass
        return arguments



""" Validator Decorators """

def SimplifyValidator(SimpleValidator):
    """Decorator for converting siMpLify objects to proper types.

    By default, this decorator checks the following parameters:
        idea
        data
        dataset
        inventory

    """

    def __init__(self, callable: Callable) -> None:
        """Sets initial validator options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.validators = {
            'idea': create_idea,
            'data': create_data,
            'dataset': create_dataset,
            'inventory': create_inventory}
        super().__init__()
        return self


def DataValidator(SimpleValidator):
    """Decorator for converting data objects to proper types.

    By default, this decorator checks any arguments that begin with 'x_', 'X_',
    'y_', or 'Y_' as wll as 'Y'. In all cases, the arguments are converted to
    either 'x' or 'y' so that wrapped objects need only include generic 'x' and
    'y' in their parameters.

    The decorator also preserves pandas data objects with feature names even
    when the wrapped object converts the data object to a numpy array.

    """

    def __init__(self, callable: Callable) -> None:
        """Sets initial validator options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.validators = {
            'x': self._create_df,
            'y': self._create_series}
        super().__init__()
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.

        All passed parameter names are converted to lower case to avoid issues
        with arguments passed with 'X' and 'Y' instead of 'x' and 'y'.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self._convert_names(arguments = arguments)
            self._store_names(arguments = arguments)
            result = self.callable(self, **arguments)
            result = self.apply(result = result)
            return result
        return wrapper

    """ Private Methods """

    def _create_df(self, x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(x, np.ndarray):
            return pd.DataFrame(x, columns = self.x_columns)
        else:
            return x

    def _create_series(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        if isinstance(y, np.ndarray):
            try:
                return pd.Series(y, name = self.y_name)
            except (AttributeError, KeyError):
                return pd.Series(y)
        else:
            return y

    def _convert_names(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts data arguments to truncated, lower-case names.

        Args:
            arguments (Dict[str, Any]): passed arguments to 'callable'.

        Returns:
            Dict[str, Any]: with any data arguments changed to either 'x' or
                'y'.

        """
        new_arguments = {}
        for parameter, value in arguments.items():
            if parameter.startswith(('X', 'Y')):
                new_arguments[parameter.lower()] = value
            else:
                new_arguments[parameter] = value
            if parameter.startswith(('x_', 'y_')):
                new_arguments[parameter[0]] = value
            else:
                new_arguments[parameter] = value
        return new_arguments

    def _store_names(self, arguments: Dict[str, Any]) -> None:
        try:
            self.x_columns = arguments['x'].columns.values
            self.y_name = arguments['y'].name
        except KeyError:
            pass
        return self

    """ Core siMpLify Methods """

    def apply(self,
            result: Union[
                pd.DataFrame,
                pd.Series,
                np.ndarray,
                Tuple[pd.DataFrame, pd.Series],
                Tuple[np.ndarray, np.ndarray]]) -> (
                    Union[
                        pd.DataFrame,
                        pd.Series,
                        Tuple[pd.DataFrame, pd.Series]]):
        """Converts values of 'result' to proper type.

        Args:
            result: Union[pd.DataFrame, pd.Series, np.ndarray,
                tuple[pd.DataFrame, pd.Series], tuple[np.ndarray, np.ndarray]]:
                result of data analysis in several possible permutations.

        Returns:
            Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, pd.Series]]:
                result returned with all objects converted to pandas datatypes.

        """
        if isinstance(result, tuple):
            return tuple(
                self.validators['x'](x = result[0]),
                self.validators['y'](y = result[1]))
        elif isinstance(result, np.ndarray):
            if result.ndim == 1:
                return self.validators['y'](y = result)
            else:
                return self.validators['x'](x = result)
        else:
            return result

def ColumnsValidator(SimpleValidator):
    """Decorator for creating column lists for wrapped methods."""

    def __init__(self, callable: Callable) -> None:
        """Sets initial validator options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.validators = {
            'columns': self._create_columns,
            'prefixes': self._create_prefixes,
            'suffixes': self._create_suffixes,
            'mask': self._create_mask}
        super().__init__()
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self.apply(arguments = arguments)
            return self.callable(self, **arguments)
        return wrapper

    """ Private Methods """

    def _create_columns(self,
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = []
        return arguments

    def _create_prefixes(self,
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = []
        return arguments


    def _create_suffixes(self,
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = []
        return arguments


    def _create_mask(self,
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = []
        return arguments

    def make_columns(method: Callable, *args, **kwargs) -> Callable:
        """Decorator which creates a complete column list from passed arguments.

        If 'prefixes', 'suffixes', or 'mask' are passed to the wrapped method, they
        are combined with any passed 'columns' to form a list of 'columns' that are
        ultimately passed to the wrapped method.

        Args:
            method (Callable): wrapped method.

        Returns:
            Callable: with 'columns' parameter that combines items from 'columns',
                'prefixes', 'suffixes', and 'mask' parameters into a single list
                of column names using the 'make_column_list' method.

        """
        call_signature = signature(method)
        @wraps(method)
        def wrapper(*args, **kwargs):
            new_arguments = {}
            parameters = dict(call_signature.parameters)
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            unpassed = list(parameters.keys() - arguments.keys())
            if 'columns' in unpassed:
                columns = []
            else:
                columns = listify(arguments['columns'])
            try:
                columns.extend(
                    make_column_list(prefixes = arguments['prefixes']))
                del arguments['prefixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    make_column_list(suffixes = arguments['suffixes']))
                del arguments['suffixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    make_column_list(mask = arguments['mask']))
                del arguments['mask']
            except KeyError:
                pass
            if not columns:
                columns = list(columns.keys())
            arguments['columns'] = deduplicate(columns)
            # method.__signature__ = Signature(arguments)
            return method(**arguments)
        return wrapper

    def make_column_list(
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            prefixes: Optional[Union[List[str], str]] = None,
            suffixes: Optional[Union[List[str], str]] = None,
            mask: Optional[Union[List[bool]]] = None) -> None:
        """Dynamically creates a new column list from a list of columns, lists
        of prefixes, and/or boolean mask.

        This method serves as the basis for the 'column_lists' decorator which
        allows users to pass 'prefixes', 'columns', and 'mask' to a wrapped
        method with a 'columns' argument. Those three arguments are then
        combined into the final 'columns' argument.

        Args:
            df (DataFrame): pandas object.
            columns (list or str): column names to be included.
            prefixes (list or str): list of prefixes for columns to be included.
            suffixes (list or str): list of suffixes for columns to be included.
            mask (numpy array, list, or Series, of booleans): mask for columns
                to be included.

        Returns:
            column_names (list): column names created from 'columns',
                'prefixes', and 'mask'.

        """
        column_names = []
        try:
            for boolean, feature in zip(mask, list(df.columns)):
                if boolean:
                    column_names.append(feature)
        except TypeError:
            pass
        try:
            temp_list = []
            for prefix in listify(prefixes, default_null = True):
                temp_list = [col for col in df if col.startswith(prefix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            temp_list = []
            for prefix in listify(suffixes, default_null = True):
                temp_list = [col for col in df if col.endswith(suffix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            column_names.extend(listify(columns, default_null = True))
        except TypeError:
            pass
        return deduplicate(iterable = column_names)
