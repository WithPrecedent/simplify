"""
.. module:: creator
:synopsis: functional data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.idea import create_idea
from simplify.core.ingredients import create_ingredients
from simplify.core.inventory import create_inventory
from simplify.core.workers import create_workers


def startup(
        idea: Union['Idea', Dict[str, Dict[str, Any]], str],
        inventory: Union['Inventory', str],
        ingredients: Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str,
            List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
            Dict[str, Union[
                'Ingredient', pd.DataFrame, pd.Series, np.ndarray, str]]],
        project: 'Project') -> None:
    """Creates and/or conforms Idea, Inventory, and Ingredients instances.

    Args:
        idea (Union['Idea', Dict[str, Dict[str, Any]], str]): an instance of
            Idea, a nested Idea-compatible nested dictionary, or a string
            containing the file path where a file of a supoorted file type with
            settings for an Idea instance is located.
        inventory (Union['Inventory', str]): an instance of Inventory or a
            string containing the full path of where the root folder should be
            located for file output. A Inventory instance contains all file path
            and import/export methods for use throughout the siMpLify package.
        ingredients (Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str, List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
            Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]): an
            instance of Ingredients, a string containing the full file
            path where a data file for a pandas DataFrame or Series is located,
            a string containing a file name in the default data folder, as
            defined in the shared Inventory instance, a DataFrame, a Series,
            numpy ndarray, a list of data objects, or dict with data objects as
            values. If a DataFrame, ndarray, or string is provided, the
            resultant DataFrame is stored at the 'df' attribute in a new
            Ingredients instance. If a list is provided, each data object is
            stored as 'df' + an integer based upon the order of the data
            objct in the list.
        project ('Project'): a related Project instance.

    Returns:
        Idea, Inventory, Ingredients instances.

    """
    idea = create_idea(idea = idea)
    idea.project = project
    inventory = create_inventory(
        inventory = inventory,
        idea = idea)
    inventory.project = project
    ingredients = create_ingredients(
        ingredients = ingredients,
        inventory = inventory)
    ingredients.project = project
    workers = create_workers(
        workers = workers,
        project = project)
    return idea, inventory, ingredients, workers
