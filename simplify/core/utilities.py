"""
.. module:: utilities
:synopsis: tasks made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
import time
from types import FunctionType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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

def datetime_string() -> str:
    """Creates a string from current date and time.

    Returns:
        str with current date and time in Y/M/D/H/M format.

    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M')

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
    """Returns if passed 'contents' is nested at least one-level.

    Args:
        dictionary (dict): dict to be tested.

    Returns:
        bool: indicating whether any value in the 'contents' is also a
            dict (meaning that 'contents' is nested).

    """
    return any(isinstance(d, dict) for d in dictionary.values())

def listify(
        variable: Any,
        default_null: Optional[bool]  = False,
        default_empty: Optional[bool] = False) -> Union[list, None]:
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
        elif default_empty:
            return []
        else:
            return ['none']
    elif isinstance(variable, list):
        return variable
    else:
        return [variable]

def numify(variable: str) -> Union[int, float, str]:
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
            return variable

def pathlibify(path: Union[str, Path]) -> Path:
    """Converts string 'path' to pathlib Path object.

    Args:
        path (Union[str, Path]): either a string representation of a path or a
            Path object.

    Returns:
        Path object.

    Raises:
        TypeError if 'path' is neither a str or Path type.

    """
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise TypeError('path must be str or Path type')

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

def subsetify(dictionary: Dict[Any, Any], subset: List[Any]) -> Dict[Any, Any]:
    """Returns a subset of a dictionary.

    The returned subset is a dictionary with keys in 'subset'.

    Args:
        dictionary (Dict[Any, Any]): dict to be subsetted.
        subset (List[Any]): list of keys to get key/values from dictionary.

    Returns:
        Dict[Any, Any]: with only keys in 'subset'

    """
    return {key: dictionary[key] for key in subset}

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
        return int(variable)
    except ValueError:
        try:
            return float(variable)
        except ValueError:
            if variable.lower() in ['true', 'yes']:
                return True
            elif variable.lower() in ['false', 'no']:
                return False
            elif ', ' in variable:
                variable = variable.split(', ')
                return [numify(v) for v in variable]
            else:
                variable = numify(variable)
                if variable in ['True', 'true', 'TRUE']:
                    return True
                elif variable in ['False', 'false', 'FALSE']:
                    return False
                elif variable in ['None', 'none', 'NONE']:
                    return 'none'
                else:
                    return variable

""" Decorators """

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
