"""
.. module:: siMpLify project
:synopsis: controller for siMpLify projects
:editor: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from simplify.core.base import SimpleOutline
from simplify.core.base import SimplePublisher
from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients
from simplify.core.inventory import Inventory
from simplify.core.publishers import Author
from simplify.core.publishers import Contributor
from simplify.core.utilities import listify
from simplify.core.workers import Worker


@dataclass
class Project(SimplePublisher):
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
        options (Optional[Union['Library', Dict[str, 'Book']]]):
            allows setting of 'options' property with an argument. Defaults to
            an empty dictionary.
        steps (Optional[List[str], str]): ordered list of steps to execute. Each
            step should match a key in 'options'. Defaults to an empty list.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced.
        auto_apply (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_apply to have an effect,
            'ingredients' must also be passed. Defaults to False.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: Union['Idea', str] = None
    inventory: Optional[Union['Inventory', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = field(default_factory = list)
    library: Optional[Dict[str, 'SimpleOutline']] = field(
        default_factory = dict)
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
        # Sets 'project' to self.
        super().__post_init__()
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

    """ Dunder Methods """

    def __call__(self):
        """Drafts, publishes, and applies Project.

        This requires an Idea and Ingredients arguments to be passed in steps
        to work properly.

        Calling Project as a function is compatible with and used by the
        command line interface.

        Raises:
            ValueError if 'ingredients' not passed when Project is called as a
                function.

        """
        # Validates 'ingreidents'.
        if self.ingredients is None:
            raise ValueError('Calling Project requires ingredients')
        else:
            self.auto_apply = True
            self.__post__init()
        return self

    """ Public Methods """

    def add_package(self, name: str, module: str, book: str) -> None:
        """Adds package to 'library'.

        Args:
            name (str): name of package. This is used as both the key to the
                created SimpleOutline in 'library' and as the 'name' attribute
                in the SimpleOutline.
            module (str): import path for the package.
            book (str): name of 'Book' class in 'module'.

        """
        self.library[name] = SimpleOutline(
            name = name,
            module = module,
            book = book)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Sets default package options available to Project.
        self.default_library = {
            'chef': SimpleOutline(
                name = 'chef',
                module = 'simplify.chef.chef',
                component = 'Cookbook'),
            'farmer': SimpleOutline(
                name = 'farmer',
                module = 'simplify.farmer.farmer',
                component = 'Almanac'),
            'actuary': SimpleOutline(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                component = 'Ledger'),
            'critic': SimpleOutline(
                name = 'critic',
                module = 'simplify.critic.critic',
                component = 'Collection'),
            'artist': SimpleOutline(
                name = 'artist',
                module = 'simplify.artist.artist',
                component = 'Canvas')}
        # Injects attributes from 'idea'.
        self = self._draft_idea(manuscript = self)
        return self

    def publish(self, steps: Optional[Union[List[str], str]] = None) -> None:
        """Finalizes iterable by creating Book instances.

        Args:
            steps (Optional[Union[List[str], str]]): option(s) to publish.

        """
        # If optional 'steps' passed, they are assigned to 'steps' attribute.
        if steps is not None:
            self.steps = listify(steps)
        # Validates 'steps'.
        self = self._publish_steps(manuscript = self)
        # Sets 'library' to 'default_library' if 'library' not passed.
        if not self.library:
            self.library = default_library
        return self

    def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data ('Ingredients'): data object for methods to be applied.

        """
        # Assigns 'data' to 'ingredients' attribute, if passed.
        if data:
            self.ingredients = data
        # Creates 'author', 'contributor', and 'worker' to build and apply
        # Book, Chapter, and Page instances.
        self.author = Author(project = self)
        self.contributor = Contributor(project = self)
        self.worker = Worker(project = self)
        # Iterates through each step, creating and applying needed Books,
        # Chapters, and Pages for each step in the Project.
        for step in self.steps:
            # Drafts and publishes Book instance at 'step' in 'library'.
            self.library[step] = self.author.draft(
                outline = self.library[step])
            self.library[step] = self.author.publish(
                book = self.library[step])
            # Drafts and publishes Chapter instance(s) for Book at 'step' in
            # 'library'.
            self.library[step] = self.contributor.draft(
                book = self.library[step])
            self.library[step] = self.contributor.publish(
                book = self.library[step])
            # Applies completed Book instance with Chapter instances to
            # 'ingredients'.
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

""" Builder Functions """

def startup(
        idea: Union['Idea', Dict[str, Dict[str, Any]], str],
        inventory: Union['Inventory', str],
        ingredients: Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str],
        project: 'Project') -> None:
    """Creates and/or validates Idea, Inventory, and Ingredients instances.

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
            str]): an instance of Ingredients, a string containing the full file
            path where a data file for a pandas DataFrame or Series is located,
            a string containing a file name in the default data folder, as
            defined in the shared Inventory instance, a DataFrame, a Series, or
            numpy ndarray. If a DataFrame, ndarray, or string is provided, the
            resultant DataFrame is stored at the 'df' attribute in a new
            Ingredients instance.

    Returns:
        Idea, Inventory, Ingredients instances.

    """
    idea = create_idea(project = project, idea = idea)
    inventory = create_inventory(
        project = project,
        inventory = inventory,
        idea = idea)
    ingredients = create_ingredients(
        project = project,
        ingredients = ingredients,
        idea = idea,
        inventory = inventory)
    return idea, inventory, ingredients

def create_idea(
        idea: Union[Dict[str, Dict[str, Any]],  'Idea'],
        project: 'Project') -> 'Idea':
    """Creates an Idea instance from passed argument.

    Args:
        idea (Union[Dict[str, Dict[str, Any]],  'Idea']): can either be a
            dict, a str file path to an ini, csv, or py file with settings, or
            an Idea instance with a configuration attribute.

    Returns:
        Idea instance, published.

    Raises:
        TypeError: if 'idea' is neither a dict, str, nor Idea instance.

    """
    def _load_from_csv(file_path: str) -> Dict[str, Any]:
        """Creates a configuration dictionary from a .csv file.

        Args:
            file_path (str): path to siMpLify-compatible .csv file.

        Returns:
            Dict[str, Any] of settings.

        Raises:
            FileNotFoundError: if the file_path does not correspond to a file.

        """
        try:
            configuration = pd.read_csv(file_path, dtype = 'str')
            return configuration.to_dict(orient = 'list')
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['configuration file ',
                file_path, ' not found']))


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
            return dict(configuration._sections)
        except FileNotFoundError:
            raise FileNotFoundError(' '.join(['configuration file ',
                file_path, ' not found']))

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
            raise FileNotFoundError(' '.join(['configuration file ',
                file_path, ' not found']))

    if isinstance(idea, Idea):
        return idea
    elif isinstance(idea, dict):
        return Idea(project = project, configuration = idea)
    elif isinstance(idea, str):
        extension = str(Path(idea).suffix)[1:]
        configuration = locals()['_'.join(['_load_from', extension])](
            file_path = idea)
        return Idea(project = project, configuration = configuration)
    else:
        raise TypeError('idea must be Idea, str, or nested dict type')

def create_inventory(
        inventory: Union['Inventory', str],
        idea: 'Idea',
        project: 'Project') -> 'Inventory':
    """Creates an Inventory instance from passed arguments.

    Args:
        inventory: Union['Inventory', str]: Inventory instance or root folder
            for one.
        idea ('Idea'): an Idea instance.

    Returns:
        Inventory instance, published.

    Raises:
        TypeError if inventory is not Inventory or str folder path.

    """
    if isinstance(inventory, Inventory):
        return inventory
    elif isinstance(inventory, str):
        return Inventory(
            project = project,
            idea = idea,
            root_folder = inventory)
    else:
        raise TypeError('inventory must be Inventory type or folder path')

def create_ingredients(
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
                           str],
        idea: 'Idea',
        inventory: 'Inventory',
        project: 'Project') -> 'Ingredients':
    """Creates an Ingredients instance.

    If 'ingredients' is an Ingredients instance, it is returned unchanged.
    If 'ingredients' is a pandas data container, an Ingredients is created
        with that data container as the 'df' attribute which is returned.
    If 'ingredients' is a file path, the file is loaded into a DataFrame and
        assigned to 'df' in an Ingredients instance which is returned.
    If 'ingredients' is a file folder, a glob in the shared Inventory is
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
        inventory ('Inventory'): a Inventory instance.

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
            inventory = inventory,
            df = ingredients)
    elif isinstance(ingredients, np.ndarray):
        return Ingredients(
            idea = idea,
            inventory = inventory,
            df =  pd.DataFrame(data = getattr(self, ingredients)))
    elif isinstance(ingredients, None):
        return Ingredients(
            idea = idea,
            inventory = inventory)
    elif isinstance(ingredients, str):
        try:
            df = inventory.load(
                folder = inventory.data,
                file_name = ingredients)
            return Ingredients(
                idea = idea,
                inventory = inventory,
                df = df)
        except FileNotFoundError:
            try:
                inventory.create_batch(
                    folder = getattr(self, ingredients))
                return Ingredients(
                    idea = idea,
                    inventory = inventory)
            except FileNotFoundError:
                raise TypeError(' '.join(
                    ['ingredients must be a file path, file folder',
                        'DataFrame, Series, None, Ingredients, or numpy',
                        'array']))