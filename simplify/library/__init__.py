"""
.. module:: siMpLify data
:synopsis: siMpLify data class factory
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = []

from importlib import import_module
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.creator import _get_supported_types
from simplify.data.ingredients import Ingredients


def make_ingredients(
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
                           str],
        idea: 'Idea',
        filer: 'Filer') -> 'Ingredients':
    """Creates an Ingredients instance.

    If 'ingredients' is an Ingredients instance, it is returned unchanged.
    If 'ingredients' is a pandas data container, an Ingredients is created
        with that data container as the 'df' attribute which is returned.
    If 'ingredients' is a file path, the file is loaded into a DataFrame and
        assigned to 'df' in an Ingredients instance which is returned.
    If 'ingredients' is a file folder, a glob in the shared Filer is
        created and an Ingredients instance is returned with 'df' as None.
    If 'ingredients' is a numpy array, it is converted to a pandas
        DataFrame at the 'df' attribute of an Ingredients instance and
        returned
    If 'ingredients' is None, a new Ingredients instance is returned with
        'df' assigned to None.

    Args:
        ingredients (Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str]): Ingredients instance or information needed to create one.
        idea ('Idea'): an Idea instance.
        filer ('Filer'): a Filer instance.

    Returns:
        Ingredients instance, published.

    Raises:
        TypeError: if 'ingredients' is neither a file path, file folder,
            None, DataFrame, Series, numpy array, or Ingredients instance.

    """
    if isinstance(ingredients, Ingredients):
        return ingredients
    elif isinstance(ingredients, (pd.Series, pd.DataFrame)):
        return Ingredients(
            idea = idea,
            filer = filer,
            df = ingredients)
    elif isinstance(ingredients, np.ndarray):
        return Ingredients(
            idea = idea,
            filer = filer,
            df =  pd.DataFrame(data = getattr(self, ingredients)))
    elif isinstance(ingredients, None):
        return Ingredients(
            idea = idea,
            filer = filer)
    elif isinstance(ingredients, str):
        try:
            df = filer.load(
                folder = filer.data,
                file_name = ingredients)
            return Ingredients(
                idea = idea,
                filer = filer,
                df = df)
        except FileNotFoundError:
            try:
                filer.make_batch(
                    folder = getattr(self, ingredients))
                return Ingredients(
                    idea = idea,
                    filer = filer)
            except FileNotFoundError:
                error = ' '.join(
                    ['ingredients must be a file path, file folder',
                        'DataFrame, Series, None, Ingredients, or numpy',
                        'array'])
                raise TypeError(error)


def make_filer(filer: Union['Filer', str], idea: 'Idea') -> 'Filer':
    """Creates an Filer instance from passed arguments.

    Args:
        filer: Union['Filer', str]: Filer instance or root folder for one.
        idea ('Idea'): an Idea instance.

    Returns:
        Filer instance, published.

    Raises:
        TypeError if filer is not Filer or str folder path.

    """
    if isinstance(filer, Filer):
        return filer
    elif os.path.isdir(filer):
        return Filer(idea = idea, root_folder = filer)
    else:
        error = 'filer must be Filer type or folder path'
        raise TypeError(error)
