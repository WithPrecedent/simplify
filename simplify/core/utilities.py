"""
.. module:: utilities
:synopsis: tasks made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import wraps
from inspect import signature
from pathlib import Path
import time
from types import FunctionType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from more_itertools import unique_everseen
import numpy as np
import pandas as pd


""" Functions """

def add_prefix(
    iterable: Union[Dict[str, Any], List],
    prefix: str) -> Union[Dict[str, Any], List]:
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

def add_suffix(
    iterable: Union[Dict[str, Any], List],
    suffix: str) -> Union[Dict[str, Any], List]:
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

def is_nested(dictionary: Dict[Any, Any]) -> bool:
    """Returns if passed 'dictionary' is nested at least one-level.

    Args:
        dictionary (dict): dict to be tested.

    Returns:
        bool: indicating whether any value in the 'dictionary' is also a
            dict (meaning that 'dictionary' is nested).

    """
    return any(isinstance(d, dict) for d in dictionary.values())

def listify(
    variable: Any,
    default_null: Optional[bool]  = False) -> Union[list, None]:
    """Stores passed variable as a list (if not already a list).

    Args:
        variable (any): variable to be transformed into a list to allow proper
            iteration.
        default_null (boolean): whether to return None (True) or ['none']
            (False).

    Returns:
        variable (list): either the original list, a string converted to a
            list, None, or a list containing 'none' as its only item.

    """
    if not variable:
        if default_null:
            return None
        else:
            return ['none']
    elif isinstance(variable, list):
        return variable
    else:
        return [variable]

def _numify(variable: str) -> Union[int, float, str]:
    """Attempts to convert 'variable' to a numeric type.

    Args:
        variable (str): variable to be converted.

    Returns
        variable (int, float, str) converted to numeric type, if possible.

    """
    try:
        return int(variable)
    except ValueError:
        try:
            return float(variable)
        except ValueError:
            raise TypeError

def proxify(instance: object, proxies: Dict[str, str]) -> object:
    """Creates proxy names for attributes, properties, and methods.

    Args:
        instance (object): instance for proxy attributes to be added.
        proxies (Dict[str, str]): dictionary of proxy values to be added. Keys
            are the original attribute, method, or property name (or partial
            name). Values are the replacement strings.

    Returns:
        object with proxy attributes added.

    """
    try:
        proxy_attributes = {}
        for name, proxy in proxies.items():
            for attribute in dir(instance):
                if name in attribute and not attribute.startswith('__'):
                    proxy_attributes[attribute.replace(name, proxy)] = (
                        getattr(instance, attribute))
        instance.__dict__.update(proxy_attributes)
    except AttributeError:
        pass
    return instance

def stringify(
    variable: Union[str, List],
    default_null: Optional[bool] = False,
    default_empty: Optional[bool] = False) -> str:
    """Converts one item list to a string (if not already a string).

    Args:
        variable (str, list): variable to be transformed into a string.
        default_null (boolean): whether to return None (True) or ['none']
            (False).

    Returns:
        variable (str): either the original str, a string pulled from a
            one-item list, or the original list.

    """
    if variable is None:
        if default_null:
            return None
        elif default_empty:
            return []
        else:
            return ['none']
    elif isinstance(variable, str):
        return variable
    else:
        try:
            return variable[0]
        except TypeError:
            return variable

def typify(variable: str) -> Union[List, int, float, bool, str]:
    """Converts stingsr to appropriate, supported datatypes.

    The method converts strings to list (if ', ' is present), int, float,
    or bool datatypes based upon the content of the string. If no
    alternative datatype is found, the variable is returned in its original
    form.

    Args:
        variable (str): string to be converted to appropriate datatype.

    Returns:
        variable (str, list, int, float, or bool): converted variable.
    """
    try:
        variable = variable.split(', ')
        return [numify(v) for v in variable]
    except (AttributeError, TypeError):
        pass
    try:
        return numify(variable)
    except TypeError:
        pass
    if variable in ['True', 'true', 'TRUE']:
        return True
    elif variable in ['False', 'false', 'FALSE']:
        return False
    elif variable in ['None', 'none', 'NONE']:
        return None
    else:
        return variable

""" Decorators """

# def convert_time(seconds: int) -> Tuple(int, int, int):
#     """Function that converts seconds into hours, minutes, and seconds.

#     Args:
#         seconds: an int containing a nubmer of seconds.

#     """
#     minutes, seconds = divmod(seconds, 60)
#     hours, minutes = divmod(minutes, 60)
#     return hours, minutes, seconds

def timer(process: Optional[str] = None) -> Callable:
    """Decorator for computing the length of time a process takes.

    Args:
        process (Optional[str]): name of class or method to be used in the
            output describing time elapsed.

    """
    if not process:
        if isinstance(process, FunctionType):
            process = process.__name__
        else:
            process = process.__class__.__name__
    def shell_timer(_function):
        def decorated(*args, **kwargs):
            def convert_time(seconds: int) -> tuple(int, int, int):
                minutes, seconds = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                return hours, minutes, seconds
            implement_time = time.time()
            result = _function(*args, **kwargs)
            total_time = time.time() - implement_time
            h, m, s = convert_time(total_time)
            print(f'{process} completed in %d:%02d:%02d' % (h, m, s))
            return result
        return decorated
    return shell_timer

def localize_arguments(
    override: Optional[bool] = True,
    excludes: Optional[List[str]] = None,
    includes: Optional[List[str]] = None) -> Callable:
    """Converts passed arguments into local attributes in the class instance.

    The created attributes use the same names as the keyword parameters.

    Args:
        method (Callable): wrapped method within a class instance.
        override (Optional[bool]): whether to override local attributes that
            already exist. Defaults to True.
        excludes (Optional[List[str]]): names of parameters for which a local
            attribute should not be created.
        includes (Optional[List[str]]): names of parameters for which a local
            attribute can be created.

    Return:
        Callable, unchanged.

    """
    def shell_localize_arguments(method: Callable, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            arguments = dict(signature(method).bind(*args, **kwargs).arguments)
            for argument, value in arguments.items():
                if argument not in self.__dict__ or override:
                    if ((includes and argument in includes)
                            or (excludes and argument not in excludes)):
                        self.__dict__[argument] = value
            return method(self, *args, **kwargs)
        return wrapper
    return shell_localize_arguments

def use_local_backups(
    excludes: Optional[List[str]] = None,
    includes: Optional[List[str]] = None) -> Callable:
    """Decorator which uses class instance attributes for unpassed parameters.

    If an optional parameter is not passed, the decorator looks for an attribute
    of the same name. If excludes is passed, the name of the parameter cannot
    be on that list. If includes is passed, the name of the parameter must be on
    that list. Only includes or excludes should be passed - if both are passed,
    excludes is ignored.

    Args:
        method (Callable): wrapped method within a class instance.
        excludes (Optional[List[str]]): names of parameters for which a local
            attribute should not be used.
        includes (Optional[List[str]]): names of parameters for which a local
            attribute can be used.

    Return:
        Callable with passed arguments in addition to replaced unpassed
            parameters.

    """
    def shell_use_local_backups(method: Callable, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            call_signature = signature(method)
            parameters = dict(call_signature.parameters)
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            unpassed = list(parameters.keys() - arguments.keys())
            if includes:
                for argument in unpassed:
                    if argument in includes:
                        try:
                            arguments.update({argument: getattr(self, argument)})
                        except AttributeError:
                            pass
            elif excludes:
                for argument in unpassed:
                    if argument not in excludes:
                        try:
                            arguments.update({argument: getattr(self, argument)})
                        except AttributeError:
                            pass
            else:
                for argument in unpassed:
                    try:
                        arguments.update({argument: getattr(self, argument)})
                    except AttributeError:
                        pass
            return method(self, **arguments)
        return wrapper
    return shell_use_local_backups

def XxYy(truncate: Optional[bool] = True) -> Callable:
    """Converts 'X' and 'Y' to 'x' and 'y' in arguments with optional
    truncation.

    Because different packages use upper and lower case names for the core
    independent and dependent variable names, this decorator converts passed
    uppercase parameter names to their lowercase versions (used by siMpLify).

    If 'truncate' is True, the named parameter is reduced to just 'x' or 'y'
    and the '_train', '_test', and '_val' suffixes are dropped. This is
    particularly useful for scikit-learn compatible methods.

    Args:
        method (Callable): wrapped method accepting lowercase versions of the
            variables.
        truncate (Optional[bool]): whether to discard the suffixes to the
            variable names and just use the first character ('x' or 'y').
            Defaults to True.

    Returns:
        method (Callable): method with arguments properly adjusted.

    """
    def shell_XxYy(method: Callable, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            arguments = signature(method).bind(self, *args, **kwargs).arguments
            new_arguments = {}
            for parameter, value in arguments.items():
                if parameter in ['X', 'Y', 'X_train', 'Y_train', 'X_test', 'Y_test',
                        'X_val', 'Y_val', 'x_train', 'y_train', 'x_test', 'y_test',
                        'x_val', 'y_val']:
                    if truncate:
                        new_parameter = parameter[0]
                    else:
                        new_parameter = parameter
                    new_arguments[new_parameter.lower()] = value
                else:
                    new_arguments[parameter] = value
            return method(self, **new_arguments)
        return wrapper
    return shell_XxYy

def numpy_shield(method: Callable) -> Callable:
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


# def columns_shield(method: Callable):
#     """Checks conditions of Cookbook step and adjusts arguments and return
#     value accordingly.

#     If the Page subclass 'step' attribute is 'none', the Ingredients
#     instance is returned unaltered.

#     If, however, there is a step other than 'none', the decorator allows
#     the passing of pandas DataFrame attributes to Ingredients even when the
#     algorithm used transforms those DataFrames to numpy ndarrays. The decorator
#     allows Ingredients attributes to be pandas DataFrames to be passed to a
#     method, have those DataFrames converted to numpy ndarrays and then restored
#     to pandas DataFrames with the original column names when the wrapped method
#     is complete.

#     Args:
#         method(method): wrapped method.

#     Returns:
#         result(Ingredients instance): with all transformed numpy ndarrays
#             restored to pandas DataFrames with the same column names.
#     """
#     dataframes_to_check = ['x_train', 'x_test', 'x', 'x_val']
#     series_to_restore = ['y_train', 'y_test', 'y', 'y_val']
#     @wraps(method)
#     def wrapper(self, *args, **kwargs):
#         arguments = signature(method).bind(self, *args, **kwargs).arguments
#         result = arguments['ingredients']
#         if hasattr(self, 'step') and self.step != 'none':
#             for df_attr in dataframes_to_check:
#                 if not getattr(result, df_attr) is None:
#                     x_columns = list(getattr(result, df_attr).columns.values)
#                     break
#             result = method(self, *args, **kwargs)
#             for df_attr in dataframes_to_check:
#                 if isinstance(getattr(result, df_attr), np.ndarray):
#                     if not getattr(result, df_attr) is None:
#                         setattr(result, df_attr, pd.DataFrame(
#                                 getattr(result, df_attr),
#                                 columns = x_columns))
#             for series in series_to_restore:
#                 if isinstance(getattr(result, series), np.ndarray):
#                     if isinstance(getattr(result, series), np.ndarray):
#                         setattr(result, series, pd.Series(
#                                 getattr(result, series),
#                                 name = self.label))
#         return result
#     return wrapper