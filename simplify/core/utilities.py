"""
.. module:: utilities
:synopsis: tasks made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from typing import Any, Dict, List, Union

import pandas as pd


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