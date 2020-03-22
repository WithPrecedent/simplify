"""
.. module:: creator
:synopsis: functional data science made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.idea import Idea
from simplify.core.library import Book
from simplify.core.creators import Publisher
from simplify.core.dataset import Dataset
from simplify.core.filer import Filer
from simplify.core.manager import Package
from simplify.core.project import Project

from simplify.core.scholar import Scholar

__all__ = [
    'Idea',
    'Book',
    'Dataset',
    'Filer',
    'Publisher',
    'Scholar',
    'Project',
    'Package']


# def startup(
#         idea: Union['Idea', Dict[str, Dict[str, Any]], str],
#         filer: Union['Filer', str],
#         dataset: Union[
#             'Dataset',
#             pd.DataFrame,
#             pd.Series,
#             np.ndarray,
#             str,
#             List[Union[
#                 'Data',
#                 pd.DataFrame,
#                 pd.Series,
#                 np.ndarray,
#                 str]],
#             Dict[str, Union[
#                 'Data',
#                 pd.DataFrame,
#                 pd.Series,
#                 np.ndarray,
#                 str]]],
#         project: 'Project') -> None:
#     """Creates and/or conforms Idea, Filer, and Dataset instances.

#     Args:
#         idea (Union['Idea', Dict[str, Dict[str, Any]], str]): an instance of
#             Idea, a nested Idea-compatible nested dictionary, or a string
#             containing the file path where a file of a supoorted file type with
#             settings for an Idea instance is located.
#         filer (Union['Filer', str]): an instance of Filer or a
#             string containing the full path of where the root folder should be
#             located for file output. A Filer instance contains all file path
#             and import/export methods for use throughout the siMpLify package.
#         dataset (Union['Dataset', pd.DataFrame, pd.Series, np.ndarray,
#             str, List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
#             Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]): an
#             instance of Dataset, a string containing the full file
#             path where a data file for a pandas DataFrame or Series is located,
#             a string containing a file name in the default data folder, as
#             defined in the shared Filer instance, a DataFrame, a Series,
#             numpy ndarray, a list of data objects, or dict with data objects as
#             values. If a DataFrame, ndarray, or string is provided, the
#             resultant DataFrame is stored at the 'df' attribute in a new
#             Dataset instance. If a list is provided, each data object is
#             stored as 'df' + an integer based upon the order of the data
#             objct in the list.
#         project ('Project'): a related Project instance.

#     Returns:
#         Idea, Filer, Dataset instances.

#     """
#     idea = Idea.create(idea = idea)
#     Filer.idea = idea
#     filer = Filer.create(root_folder = filer)
#     Dataset.idea = idea
#     Dataset.filer = filer
#     dataset = Dataset.create(data = dataset)
#     return idea, filer, dataset