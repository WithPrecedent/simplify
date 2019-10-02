"""
.. module:: decorators
:synopsis: decorators for methods and functions throughout siMpLify
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import wraps
from inspect import getfullargspec, signature
import time
from types import FunctionType

import numpy as np
import pandas as pd


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
            produce_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - produce_time
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

def choose_df(method):
    """Substitutes the default DataFrame or Seriesif one is not passed to the
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
        if 'df' in argspec.args and 'df' in unpassed_args:
            kwargs.update({'df': getattr(self, self.default_df)})
        return method(self, *args, **kwargs)
    return wrapper

def combine_lists(method, arguments_to_check = None):
    """Decorator which creates a complete column list from kwargs passed
    to wrapped method.

    Args:
        method(method): wrapped method.

    Returns:
        new_kwargs(dict): 'columns' parameter has items from 'columns',
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
    """Checks conditions of Cookbook step and adjusts arguments and return
    value accordingly.

    If the SimpleStep subclass 'technique' attribute is 'none', the Ingredients
    instance is returned unaltered.

    If, however, there is a technique other than 'none', the decorator allows
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
        if hasattr(self, 'technique') and self.technique != 'none':
            for df in dataframes_to_check:
                if not getattr(arguments['ingredients'], df) is None:
                    x_columns = list(getattr(
                            arguments['ingredients'], df).values)
                    break
            result = method(self, *args, **kwargs)
            for df in dataframes_to_check:
                if isinstance(getattr(result, df), np.ndarray):
                    if not getattr(result, df) is None:
                        setattr(result, df, pd.DataFrame(
                                getattr(result, df), columns = x_columns))
            for series in series_to_restore:
                if isinstance(getattr(result, series), np.ndarray):
                    if isinstance(getattr(result, series), np.ndarray):
                        setattr(result, series, pd.Series(
                                getattr(result, series), name = self.label))
        return result
    return wrapper