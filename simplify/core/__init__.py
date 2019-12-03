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

import simplify
from simplify.core.book import Book
from simplify.core.idea import Idea


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = _get_supported_types()


def create_book(
        idea: Union[Dict[str, Dict[str, Any]], str, 'Idea'],
        library: Optional[Union['Library', str]],
        ingredients: Optional[Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]] = None,
        steps: Optional[Union[List[str], str]] = None,
        name: Optional[str] = None,
        auto_publish: Optional[bool] = True) -> 'Book':
    """
    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        library (Optional[Union['Library', str]]): an instance of
            library or a string containing the full path of where the root
            folder should be located for file output. A library instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Library instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.

    """
    idea, library, ingredients = startup(
        idea = idea,
        library = library,
        ingredients = ingredients)
    return Book(
        idea = idea,
        library = library,
        ingredients = ingredients,
        steps = steps,
        name = name,
        auto_publish = auto_publish)

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
