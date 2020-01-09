"""
.. module:: siMpLify project
:synopsis: controller for siMpLify projects
:editor: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from simplify.core import startup
from simplify.core.base import book import Contents
from simplify.core.ingredients import validate_ingredients
from simplify.core.editors import Author
from simplify.core.editors import Publisher
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify
from simplify.core.workers import Worker


@dataclass
class Project(MutableMapping):
    """Controller class for siMpLify projects.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supported file type with settings for an Idea instance is
            located. Even though a default value is provided (None) to lessen
            subclass MRO problems, 'idea' is required for a Project to work.
        inventory (Optional[Union['Inventory', str]]): an instance of Inventory
            or a string containing the full path of where the root folder should
            be located for file output. A inventory instance contains all file
            path and import/export methods for use throughout the siMpLify
            package. Default is None.
        ingredients (Optional[Union['Ingredients', 'Ingredient', pd.DataFrame,
            np.ndarray, str]]): an instance of Ingredients, an instance of
            Ingredient, a string containing the full file path where a data file
            for a pandas DataFrame is located, a string containing a file name
            in the default data folder (as defined in the shared Inventory
            instance), a full folder path where raw files for data to be
            extracted from, a string containing a folder name which is an
            attribute in the shared Inventory instance, a DataFrame, or numpy
            ndarray. If a DataFrame, Ingredient instance, ndarray, or string is
            passed, the resultant data object is stored in the 'data' attribute
            in a new Ingredients instance as a DataFrame. Default is None.
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'library'. Defaults to an empty list. If
            no 'steps' are provided, a Project instance attempts to find them
            in 'idea'.
        library (Optional[Dict[str, Union['BookOutline', 'Book']]]): attribute
            which stores the siMpLify objects which are iterated by Project. If
            not passed or later populated, the internally stored
            'default_library' is used when 'publish' is called. Defaults to an
            empty dictionary.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'ingredients' must also
            be passed. Defaults to False.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'simplify'.

    """
    idea: Union['Idea', str] = None
    inventory: Optional[Union['Inventory', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        'Ingredient',
        pd.DataFrame,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = field(default_factory = list)
    library: Optional[Dict[str, 'BookOutline'] = field(default_factory = dict)
    corpus: Optional[Dict[str, 'Book']] = field(default_factory = dict)
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False
    name: Optional[str] = 'simplify'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Checks 'idea' to make sure it was passed.
        if self.idea is None:
            raise AttributeError('Project requires idea to be passed.')
        # Creates a unique 'project_id' from date and time.
        self.project_id = datetime_string()
        # Validates 'idea', 'inventory', and 'ingredients'.
        self.idea, self.inventory, self.ingredients = startup(
            idea = self.idea,
            inventory = self.inventory,
            ingredients = self.ingredients,
            project = self)
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        # Calls 'apply' method if 'auto_apply' is True.
        if self.auto_apply:
            self.apply()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Union['BookOutline', 'Book']:
        """Returns item in the 'library' dictionary.

        Args:
            key (str): name of key in the 'library' dictionary.

        Returns:
            Union['BookOutline', 'Book']: item stored as a the 'library'
                dictionary value.

        Raises:
            KeyError: if 'key' is not found in the 'library' dictionary.

        """
        try:
            return self.library[key]
        except KeyError:
            raise KeyError(' '.join([key, 'is not in', self.name]))

    def __setitem__(self,
            key: str,
            value: Union['BookOutline', 'Book']) -> None:
        """Sets 'key' in the 'library' dictionary to 'value'.

        Args:
            key (str): name of key in the 'library' dictionary.
            value (Union['BookOutline', 'Book']): value to be paired with 'key'
                in the 'library' dictionary.

        """
        self.library[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes item in the 'library' dictionary.

        Args:
            key (str): name of key in the 'library' dictionary.

        """
        try:
            del self.library[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'library' with keys in 'steps'.

        Returns:
            Iterable stored in 'library'.

        """
        subset = {k: self.library[k] for k in self.steps if k in self.library}
        return iter(subset.items())

    def __len__(self) -> int:
        """Returns length of 'steps'.

        Returns:
            Integer of length of 'steps'.

        """
        return len(self.steps)

    """ Other Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies Project.

        This requires idea and ingredients arguments to be passed to work
        properly.

        Calling Project as a function is compatible with and used by the
        command line interface.

        Raises:
            ValueError if 'ingredients' is not passed when Project is called as
                a function.

        """
        # Validates 'ingredients'.
        if self.ingredients is None:
            raise ValueError(
                'Calling Project as a function requires ingredients')
        else:
            self.auto_apply = True
            self.__post__init()
        return self

    """ Public Methods """

    def add_book(self, name: str, module: str, book: str) -> None:
        """Adds subpackage to 'library'.

        Args:
            name (str): name of subpackage. This is used as both the key to the
                created BookOutline in 'library' and as the 'name' attribute
                in the BookOutline.
            module (str): import path for the subpackage.
            book (str): name of 'Book' subclass in 'module'.

        """
        self.library[name] = BookOutline(
            name = name,
            module = module,
            book = book)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Sets default package options available to Project.
        self.default_library = {
            'chef': BookOutline(
                name = 'chef',
                module = 'simplify.chef.chef',
                book = 'Cookbook'),
            'farmer': BookOutline(
                name = 'farmer',
                module = 'simplify.farmer.farmer',
                book = 'Almanac'),
            'actuary': BookOutline(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                book = 'Ledger'),
            'critic': BookOutline(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Collection'),
            'artist': BookOutline(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas')}
        # Sets default iterable names.
        self._iterables = {
            Book: 'chapters',
            Chapter: 'techniques',
            Technique: None}
        # Creates 'author' to build Book instances.
        self.author = Author(project = self)
        # Iterates through 'steps' and creates a skeleton of each Book.
        self.author.draft()
        return self

    def publish(self, steps: Optional[Union[List[str], str]] = None) -> None:
        """Finalizes iterable by creating Book instances.

        Args:
            steps (Optional[Union[List[str], str]]): option(s) to publish.

        """
        # Assigns 'steps' argument to 'steps' attribute, if passed.
        if steps is not None:
            self.steps = steps
        # Converts 'steps' attribute to a list, if necessary.
        self.steps = listify(steps, default_empty = True)
        # Injects attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Sets 'library' to 'default_library' if 'library' not passed.
        if not self.library:
            self.library = self.default_library
        # Iterates through 'steps' and finalizes each Book instance.
        self.author.publish()
        return self

    def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Optional['Ingredients']): data object for methods to be
                applied. If not passed, data stored in the 'ingredients' is
                used.

        """
        # Assigns 'data' to 'ingredients' attribute and validates it.
        if data:
            self.ingredients = validate_ingredients(ingredients = data)
        # Creates a 'worker' to apply Book instance to 'ingredients'.
        self.worker = Worker(project = self)
        # Iterates through each step, creating and applying needed Books,
        # Chapters, and Contents for each step in the Project.
        for step in self.steps:
            if self.library[step].returns_data:
                self.ingredients = self.worker.apply(
                    book = self.library[step],
                    data = self.ingredients,
                    **kwargs)
            else:
                self.worker.apply(
                    book = self.library[step],
                    data = self.ingredients,
                    **kwargs)
        return self