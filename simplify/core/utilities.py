"""
.. module:: utilities
:synopsis: tasks made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import wraps
from inspect import getfullargspec, signature
import time
from types import FunctionType
from typing import Any, Dict, List, Union

from more_itertools import unique_everseen
import numpy as np
import pandas as pd


""" Functions """

def add_prefix(iterable: Union[Dict, List], prefix: str) -> Union[Dict, List]:
    """Adds prefix to each item in a list or keys in a dict.

    An underscore is automatically added after the string prefix.

    Args:
        iterable (list(str) or dict(str: any)): iterable to be modified.
        prefix (str): prefix to be added.
    Returns:
        list or dict with prefixes added.

    """
    try:
        return {prefix + '_' + k: v for k, v in iterable.items()}
    except TypeError:
        return [prefix + '_' + item for item in iterable]

def add_suffix(iterable: Union[Dict, List], suffix: str) -> Union[Dict, List]:
    """Adds suffix to each item in a list or keys in a dict.

    An underscore is automatically added after the string suffix.

    Args:
        iterable (list(str) or dict(str: any)): iterable to be modified.
        suffix (str): suffix to be added.
    Returns:
        list or dict with suffixes added.

    """
    try:
        return {k + '_' + suffix: v for k, v in iterable.items()}
    except TypeError:
        return [item + '_' + suffix for item in iterable]

def deduplicate(iterable: Union[List, pd.DataFrame, pd.Series]) -> (
    Union[List, pd.DataFrame, pd.Series]):
    """Deduplicates list, pandas DataFrame, or pandas Series.

    Args:
        iterable (list, DataFrame, or Series): iterable to have duplicate
            entries removed.

    Returns:
        iterable (list, DataFrame, or Series, same as passed type):
            iterable with duplicate entries removed.

    """
    try:
        return list(unique_everseen(iterable))
    except TypeError:
        return iterable.drop_duplicates(inplace = True)

def is_nested(dictionary: Dict):
    """Returns if passed 'dictionary' is nested at least one-level.

    Args:
        dictionary (dict): dict to be tested.

    Returns:
        bool: indicating whether any value in the 'dictionary' is also a
            dict (meaning that 'dictionary' is nested).

    """
    return any(isinstance(d, dict) for d in dictionary.values())

def listify(variable: Any, use_null: bool = False) -> Union[list, None]:
    """Stores passed variable as a list (if not already a list).

    Args:
        variable (any): variable to be transformed into a list to allow proper
            iteration.
        use_null (boolean): whether to return None (True) or ['none']
            (False).

    Returns:
        variable (list): either the original list, a string converted to a
            list, None, or a list containing 'none' as its only item.

    """
    if not variable:
        if use_null:
            return None
        else:
            return ['none']
    elif isinstance(variable, list):
        return variable
    else:
        return [variable]

def stringify(variable: Union[str, List]) -> str:
    """Converts one item list to a string (if not already a string).

    Args:
        variable (str, list): variable to be transformed into a string.

    Returns:
        variable (str): either the original str, a string pulled from a
            one-item list, or the original list.

    """
    if variable is None:
        return 'none'
    elif isinstance(variable, str):
        return variable
    else:
        try:
            return variable[0]
        except TypeError:
            return variable

""" Decorators """

def convert_time(seconds):
    """Function that converts seconds into hours, minutes, and seconds.

    Args:
        seconds: an int containing a nubmer of seconds.
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def timer(process = None):
    """Decorator for computing the length of time a process takes.

    Args:
        process: string containing name of class or method to be used in the
            output describing time elapsed.

    """
    if not process:
        if isinstance(process, FunctionType):
            process = process.__name__
        else:
            process = process.__class__.__name__
    def shell_timer(_function):
        def decorated(*args, **kwargs):
            implement_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - implement_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result
        return decorated
    return shell_timer

def localize(method):
    """Converts passed keyword arguments into local attributes in the class
    instance.

    The created attributes use the same names as the keyword parameters.

    Args:
        method: wrapped method within a class instance.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        return method
    return wrapper

def local_backups(method, excludes = None, includes = None):
    """Decorator which uses class instance attribute of the same name as a
    passed parameter if no argument is passed for that parameter and the
    parameter is not listed in excludes.

    Args:
        method: wrapped method within a class instance.
        excludes: list or string of parameters for which a local attribute
            should not be used.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        if includes:
            for unpassed in unpassed_args:
                if unpassed in includes and hasattr(self, unpassed):
                    kwargs.update({unpassed: getattr(self, unpassed)})
        elif excludes:
            for unpassed in unpassed_args:
                if unpassed not in excludes and hasattr(self, unpassed):
                    kwargs.update({unpassed: getattr(self, unpassed)})
        else:
            for unpassed in unpassed_args:
                if hasattr(self, unpassed):
                    kwargs.update({unpassed: getattr(self, unpassed)})
        return method(self, *args, **kwargs)
    return wrapper

def XxYy(truncate = False):
    """Converts 'X' and 'Y' to 'x' and 'y' in arguments.

    Because different packages use upper and lower case names for the core
    independent and dependent variable names, this decorator converts passed
    uppercase parameter names to their lowercase versions (used by siMpLify).

    If 'truncate' is True, the named parameter is reduced to just 'x' or 'y'.
    This is particularly useful for scikit-learn compatibile methods.

    Args:
        truncate (bool): whether to discard the suffixes to the variable names
            and just use the first character ('x' or 'y').
        method (method): wrapped method accepting lowercase versions of the
            variables.

    Returns:
        method (method): method with arguments properly adjusted.

    """
    def shell_converter(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            arguments = signature(method).bind(self, *args, **kwargs).arguments
            new_arguments = {}
            for parameter, argument in arguments.items():
                if parameter in ['X', 'Y', 'X_train', 'Y_train', 'X_test',
                                 'Y_test', 'X_val', 'Y_val']:
                    new_arguments[parameter.lower()] = argument
                else:
                    new_arguments[parameter] = argument
            return method(self, **new_arguments)
        return wrapper
    return shell_converter

def choose_df(method):
    """Substitutes the default DataFrame or Series if one is not passed to the
    decorated method.

    Args:
        method(method): wrapped method.

    Returns:
        df(DataFrame or Series): if the passed 'df' parameter was None,
            the attribute named by 'default_df' will be passed. Otherwise,
            df will be passed to the wrapped method intact.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        if 'df' in unpassed_args:
            kwargs.update({'df': getattr(self, self.default_df)})
        return method(self, *args, **kwargs)
    return wrapper

def combine_lists(method, arguments_to_check = None):
    """Decorator which creates a complete column list from kwargs passed
    to wrapped method.

    Args:
        method (method): wrapped method.

    Returns:
        new_kwargs (dict): 'columns' parameter has items from 'columns',
            'prefixes', and 'mask' parameters combined into a single list
            of column names using the 'create_column_list' method.

    """
    # kwargs names to use to create publishd 'columns' argument
    if not arguments_to_check:
        arguments_to_check = ['columns', 'prefixes', 'mask']
    new_kwargs = {}
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        argspec = getfullargspec(method)
        unpassed_args = argspec.args[len(args):]
        if ('columns' in unpassed_args
                and 'prefixes' in unpassed_args
                and 'mask' in unpassed_args):
            columns = list(self.datatypes.keys())
        else:
            for argument in arguments_to_check:
                if argument in kwargs:
                    new_kwargs[argument] = kwargs[argument]
                else:
                    new_kwargs[argument] = None
                if argument in ['prefixes', 'mask'] and argument in kwargs:
                    del kwargs[argument]
            columns = self.create_column_list(**new_kwargs)
            kwargs.update({'columns': columns})
        return method(self, **kwargs)
    return wrapper

def numpy_shield(method):
    """Stores and then reapplies feature names to passed pandas DataFrames.

    If the Algorithm subclass 'step' attribute is 'none', the
    Ingredients instance is returned unaltered.

    If, however, there is a step other than 'none', the decorator allows
    the passing of pandas DataFrame attributes to Ingredients even when the
    algorithm used transforms those DataFrames to numpy ndarrays. The decorator
    allows Ingredients attributes to be pandas DataFrames to be passed to a
    method, have those DataFrames converted to numpy ndarrays and then restored
    to pandas DataFrames with the original column names when the wrapped method
    is complete.

    Args:
        method (method): wrapped method.

    Returns:
        result (Ingredients): with all transformed numpy ndarrays restored to
            pandas DataFrames with the same column names.
    """
    dataframes_to_check = ['x_train', 'x_test', 'x', 'x_val']
    series_to_restore = ['y_train', 'y_test', 'y', 'y_val']
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        arguments = signature(method).bind(self, *args, **kwargs).arguments
        result = arguments['ingredients']
        if self.step != 'none':
            for df_attr in dataframes_to_check:
                if not getattr(result, df_attr) is None:
                    x_columns = list(getattr(result, df_attr).columns.values)
                    break
            result = method(self, *args, **kwargs)
            for df_attr in dataframes_to_check:
                if isinstance(getattr(result, df_attr), np.ndarray):
                    if not getattr(result, df_attr) is None:
                        setattr(result, df_attr, pd.DataFrame(
                                getattr(result, df_attr),
                                columns = x_columns))
            for series in series_to_restore:
                if isinstance(getattr(result, series), np.ndarray):
                    if isinstance(getattr(result, series), np.ndarray):
                        setattr(result, series, pd.Series(
                                getattr(result, series),
                                name = self.label))
        return result
    return wrapper


def columns_shield(method):
    """Checks conditions of Cookbook step and adjusts arguments and return
    value accordingly.

    If the Page subclass 'step' attribute is 'none', the Ingredients
    instance is returned unaltered.

    If, however, there is a step other than 'none', the decorator allows
    the passing of pandas DataFrame attributes to Ingredients even when the
    algorithm used transforms those DataFrames to numpy ndarrays. The decorator
    allows Ingredients attributes to be pandas DataFrames to be passed to a
    method, have those DataFrames converted to numpy ndarrays and then restored
    to pandas DataFrames with the original column names when the wrapped method
    is complete.

    Args:
        method(method): wrapped method.

    Returns:
        result(Ingredients instance): with all transformed numpy ndarrays
            restored to pandas DataFrames with the same column names.
    """
    dataframes_to_check = ['x_train', 'x_test', 'x', 'x_val']
    series_to_restore = ['y_train', 'y_test', 'y', 'y_val']
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        arguments = signature(method).bind(self, *args, **kwargs).arguments
        result = arguments['ingredients']
        if hasattr(self, 'step') and self.step != 'none':
            for df_attr in dataframes_to_check:
                if not getattr(result, df_attr) is None:
                    x_columns = list(getattr(result, df_attr).columns.values)
                    break
            result = method(self, *args, **kwargs)
            for df_attr in dataframes_to_check:
                if isinstance(getattr(result, df_attr), np.ndarray):
                    if not getattr(result, df_attr) is None:
                        setattr(result, df_attr, pd.DataFrame(
                                getattr(result, df_attr),
                                columns = x_columns))
            for series in series_to_restore:
                if isinstance(getattr(result, series), np.ndarray):
                    if isinstance(getattr(result, series), np.ndarray):
                        setattr(result, series, pd.Series(
                                getattr(result, series),
                                name = self.label))
        return result
    return wrapper