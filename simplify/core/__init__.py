"""
.. module:: siMpLify core
:synopsis: siMpLify core class factory
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from importlib import import_module
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.idea import Idea


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = _get_supported_types()


def create_idea(idea: Union[Dict[str, Dict[str, Any]], str, 'Idea']) -> 'Idea':
    """Creates an Idea instance from passed argument.

    Args:
        idea (Union[Dict[str, Dict[str, Any]], str, 'Idea']): can either be a
            dict, a str file path to an ini, csv, or py file with settings, or
            an Idea instance with a configuration attribute.

    Returns:
        Idea instance, published.

    Raises:
        TypeError: if 'idea' is neither a dict, str, nor Idea instance.

    """
    if isinstance(idea, Idea):
        return idea
    elif isinstance(idea, dict):
        return Idea(configuration = dict)
    elif isinstance(idea, str):
        extension = str(Path(idea).suffix)[1:]
        configuration = globals()['_'.join(['_load_from', extension])](
            file_path = idea)
        return Idea(configuration = configuration)
    else:
        error = 'idea must be Idea, str, or nested dict type'
        raise TypeError(error)

def _load_from_csv(file_path: str) -> Dict[str, Any]:
    """Creates a configuration dictionary from a .csv file.

    Args:
        file_path (str): path to siMpLify-compatible .csv file.

    Returns:
        Dict[str, Any] of settings.

    Raises:
        FileNotFoundError: if the file_path does not correspond to a file.

    """
    configuration = pd.read_csv(file_path, dtype = 'str')
    return configuration.to_dict(orient = 'list')

def _load_from_ini(file_path: str) -> Dict[str, Any]:
    """Creates a configuration dictionary from an .ini file.

    Args:
        file_path (str): path to configparser-compatible .ini file.

    Returns:
        Dict[str, Any] of configuration.

    Raises:
        FileNotFoundError: if the file_path does not correspond to a file.

    """
    try:
        configuration = ConfigParser(dict_type = dict)
        configuration.optionxform = lambda option: option
        configuration.read(file_path)
        configuration = dict(configuration._sections)
    except FileNotFoundError:
        error = ' '.join(['configuration file ', file_path, ' not found'])
        raise FileNotFoundError(error)
    return configuration

def _load_from_py(file_path: str) -> Dict[str, Any]:
    """Creates a configuration dictionary from a .py file.

    Args:
        file_path (str): path to python module with '__dict__' dict defined.

    Returns:
        Dict[str, Any] of configuration.

    Raises:
        FileNotFoundError: if the file_path does not correspond to a file.

    """
    try:
        return getattr(import_module(file_path), '__dict__')
    except FileNotFoundError:
        error = ' '.join(['configuration file ', file_path, ' not found'])
        raise FileNotFoundError(error)
