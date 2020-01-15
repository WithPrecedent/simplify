"""
.. module:: siMpLify project
:synopsis: controller class for siMpLify projects
:editor: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Iterable
from collections.abc import Iterator
from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.core import workers
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify
from simplify.core.ingredients import create_ingredients


@dataclass
class Project(Iterable):
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
        workers (Optional[Union['Workers', List[str], str]]): ordered list of
            workers to call. Each worker should match a key in 'workers'. Defaults
            to None. If no 'workers' are provided, a Project instance attempts to
            find them in 'idea' or uses 'default_workers'.
        library (Optional[Dict[str, Union['Worker', 'Book']]]): attribute
            which stores the siMpLify objects which are iterated by Project. If
            not passed or later populated, the internally stored
            'default_workers' is used when 'publish' is called. Defaults to an
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
    workers: Optional[Union['Workers', List[str], str]] = None
    library: Optional[Dict[str, 'Book']] = field(default_factory = dict)
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False
    name: Optional[str] = 'project'

    def __post_init__(self) -> None:
        """Initializes a Project and validates arguments."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Checks 'idea' to make sure it was passed.
        if self.idea is None:
            raise ValueError('Project requires an idea argument')
        # Creates a unique project 'identification' from date and time.
        self.identification = datetime_string()
        # Validates 'idea', 'inventory', and 'ingredients'.
        self.idea, self.inventory, self.ingredients, self.workers = (
            simplify.startup(
                idea = self.idea,
                inventory = self.inventory,
                ingredients = self.ingredients,
                workers = self.workers,
                project = self))
        self.position = 0
        self.draft()
        return self

    def __iter__(self) -> Iterable:
        for item in self.iterable:
            if isinstance(item, tuple):
                worker = self.workers[item[0]]
                editor = getattr(worker, [item[1]])
                method = self.editors[item[1]]
                if item[1] in 'apply':
                    yield getattr(editor, method)(
                        data = self.ingredients)
                else:
                    yield getattr(editor, method)()
            else:
                getattr(self, item)()

    def __next__(self) -> Iterable:
        """

        """
        try:
            self.position += 1
            return self.iterable[self.position - 1]
        except IndexError:
            raise StopIteration()

    def _draft_iterable(self) -> None:
        """Creates 'iterable' from 'editors' and 'workers'."""
        plans = product(self.editors.keys(), self.workers.keys())
        self.iterable = list(map(tuple, plans)))
        return self

    def _draft_workers(self) -> None:
        """Creates SimpleEditor instances and drafts Books for each Worker."""
        for name, worker in self.workers.items():
            # For each worker, create an Author, Publisher, and Scholar instance
            # to draft, publish, and apply Book instances.
            self.workers[name].author = Author(
                project = self,
                worker = worker)
            self.workers[name].publisher = Publisher(
                project = self,
                worker = worker)
            self.workers[name].scholar = Scholar(
                project = self,
                worker = worker)
            # Drafts a Book instance for each 'worker' using 'author'.
            self.workers[name].author.draft()
        return self

    def draft(self) -> None:
        self.default_workers = {
            'chef': workers.Worker(
                name = 'chef',
                module = 'simplify.chef.chef',
                book = 'Cookbook',
                scholar = 'chef'),
            'farmer': workers.Worker(
                name = 'farmer',
                module = 'simplify.farmer.farmer',
                book = 'Almanac'),
            'actuary': workers.Worker(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                book = 'Ledger'),
            'critic': workers.Worker(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Collection'),
            'artist': workers.Worker(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas')}
        self.editors = {
            'author' : 'draft',
            'publisher': 'publish',
            'scholar': 'apply'}
        self.workers = create_workers(workers = workers, project = self)
        self._draft_iterable()
        self.__next__()
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
            self.ingredients = create_ingredients(ingredients = data)
        # Iterates through each worker, publishing and applying needed Books,
        # Chapters, and SimpleCatalog for each worker in the Project.
        for i in iter(self):
            next()
        return self

