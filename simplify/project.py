"""
.. module:: siMpLify project
:synopsis: entry point for implementing multiple siMpLify subpackages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.core.author import Author
from simplify.core.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class Project(Author):
    """Controller class for siMpLify projects.

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
    idea: Union['Idea', str]
    library: Optional[Union['Library', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Finalizes 'idea', 'library', and 'ingredients instances.
        self.idea, self.library, self.ingredients = simplify.startup(
            idea = self.idea,
            library = self.library,
            ingredients = self.ingredients)
        # Injects SimpleOptions with shared Idea and Library.
        SimpleOptions.idea = self.idea
        SimpleOptions.library = self.library
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_options(self) -> None:
        """Sets step options with information for module importation."""
        self._options = SimpleOptions(
            options = {
                'farmer': ('simplify.farmer', 'Almanac'),
                'chef': ('simplify.chef', 'Cookbook'),
                'actuary': ('simplify.actuary', 'Ledger'),
                'critic': ('simplify.critic', 'Collection'),
                'artist': ('simplify.artist', 'Canvas')},
            _manuscript = self)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        super().draft(manuscript = self)
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Finalizes"""
        super().publish(manuscript = self, data = data)
        return self

    def apply(self, data: Optional[object] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        super().apply(manuscript = self, data = data, **kwargs)
        return self