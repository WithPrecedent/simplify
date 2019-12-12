"""
.. module:: siMpLify project
:synopsis: entry point for implementing multiple siMpLify steps
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from simplify import core
from simplify.core.author import Author
from simplify.core.library import Library
from simplify.core.manuscript import Manuscript
from simplify.core.options import ManuscriptOptions
from simplify.core.outline import Outline
from simplify.core.utilities import listify


DEFAULT_OPTIONS = {
    'farmer': Outline(
        name = 'farmer',
        module = 'simplify.farmer',
        component = 'Almanac'),
    'chef': Outline(
        name = 'chef',
        module = 'simplify.chef',
        component = 'Cookbook'),
    'actuary': Outline(
        name = 'actuary',
        module = 'simplify.actuary',
        component = 'Ledger'),
    'critic': Outline(
        name = 'critic',
        module = 'simplify.critic',
        component = 'Collection'),
    'artist': Outline(
        name = 'artist',
        module = 'simplify.artist',
        component = 'Canvas')}


@dataclass
class Project(Manuscript):
    """Controller class for siMpLify projects.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supported file type with settings for an Idea instance is
            located.
        inventory (Optional[Union['Inventory', str]]): an instance of Inventory
            or a string containing the full path of where the root folder should
            be located for file output. A inventory instance contains all file
            path and import/export methods for use throughout the siMpLify
            package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Inventory instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'options'. Defaults to an empty list.
        options (Optional[Union['ManuscriptOptions', Dict[str, 'Book']]]): 
            allows setting of 'options' property with an argument. Defaults to 
            an empty dictionary.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: Union['Idea', str]
    inventory: Optional[Union['Inventory', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = field(default_factory = list)
    options: Optional[Union['ManuscriptOptions', Dict[str, 'Book']]] = field(
        default_factory = dict)
    auto_publish: Optional[bool] = True
    name: Optional[str] = 'simplify'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Finalizes 'idea', 'inventory', and 'ingredients instances.
        self.idea, self.inventory, self.ingredients = core.startup(
            idea = self.idea,
            inventory = self.inventory,
            ingredients = self.ingredients)
        super()._post_init__()
        return self
    
    """ Private Methods """

    def _draft_options(self) -> None:
        """Converts 'options' to proper ManuscriptOptions form."""
        if not self.options:
            self.options = ManuscriptOptions(
                options = globals()['DEFAULT_OPTIONS'],
                parent = self)
        elif isinstance(self.options, Dict):
            self.options = ManuscriptOptions(
                options = self.options, 
                parent = self)
        else:
            self.options = ManuscriptOptions(options = {}, parent = self)
        # Creates private '_options' to allow usage of 'options' property.
        self._options = self.options
        return self

    def _draft_steps(self) -> None:
        """Attempts to find 'steps' in 'idea' if not were provided."""
        if not self.steps:
            try:
                return listify(self.idea[name]['_'.join([name, 'steps'])])
            except KeyError:
                pass
        return self
    
    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        super().draft()
        # Creates Author instance for building Manuscript objects.
        self.author = Author(
            idea = self.idea,
            inventory = self.inventory,
            auto_publish = self.auto_publish)
        # Creates Library instance for storing built objects.
        self.library = Library()
        Manuscript.library = self.library
        return self

    def publish(self) -> None:
        """Finalizes steps by creating Book instances in options.

        Args:
            data (Optional[object]): an optional object needed for a Book's
                'publish' method.

        """
        for step in self.steps:
            book = self.author.create(name = step, options = self.options)
            self.library[step] = book
            setattr(self, step, self.library[step])
        return self

    def apply(self, data: Optional[object] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        if data:
            self.ingredients = data
        for step in self.steps:
            self.ingredients = getattr(self, step).apply(
                data = self.ingredients, 
                **kwargs)
        return self