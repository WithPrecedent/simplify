"""
.. module:: creator
:synopsis: functional data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import numpy as np
import pandas as pd
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

from simplify.core.book import Book
from simplify.core.dataset import Dataset
from simplify.core.idea import Idea
from simplify.core.inventory import Inventory
from simplify.core.project import Project
from simplify.core.project import Task
from simplify.core.publisher import Publisher
from simplify.core.worker import Worker

__all__ = [
    'Book',
    'Publisher',
    'Publisher',
    'Worker',
    'create_idea',
    'create_dataset',
    'create_inventory',
    'Project',
    'Task']


def startup(
        idea: Union['Idea', Dict[str, Dict[str, Any]], str],
        inventory: Union['Inventory', str],
        dataset: Union[
            'Dataset',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str,
            List[Union[
                'Data',
                pd.DataFrame,
                pd.Series,
                np.ndarray,
                str]],
            Dict[str, Union[
                'Data',
                pd.DataFrame,
                pd.Series,
                np.ndarray,
                str]]],
        project: 'Project') -> None:
    """Creates and/or conforms Idea, Inventory, and Dataset instances.

    Args:
        idea (Union['Idea', Dict[str, Dict[str, Any]], str]): an instance of
            Idea, a nested Idea-compatible nested dictionary, or a string
            containing the file path where a file of a supoorted file type with
            settings for an Idea instance is located.
        inventory (Union['Inventory', str]): an instance of Inventory or a
            string containing the full path of where the root folder should be
            located for file output. A Inventory instance contains all file path
            and import/export methods for use throughout the siMpLify package.
        dataset (Union['Dataset', pd.DataFrame, pd.Series, np.ndarray,
            str, List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
            Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]): an
            instance of Dataset, a string containing the full file
            path where a data file for a pandas DataFrame or Series is located,
            a string containing a file name in the default data folder, as
            defined in the shared Inventory instance, a DataFrame, a Series,
            numpy ndarray, a list of data objects, or dict with data objects as
            values. If a DataFrame, ndarray, or string is provided, the
            resultant DataFrame is stored at the 'df' attribute in a new
            Dataset instance. If a list is provided, each data object is
            stored as 'df' + an integer based upon the order of the data
            objct in the list.
        project ('Project'): a related Project instance.

    Returns:
        Idea, Inventory, Dataset instances.

    """
    idea = Idea.create(idea = idea)
    Inventory.idea = idea
    inventory = Inventory.create(root_folder = inventory)
    Dataset.idea = idea
    Dataset.inventory = inventory
    dataset = Dataset.create(dataset = dataset)
    return idea, inventory, dataset