"""
.. module:: creator
:synopsis: functional data science made simple
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

from simplify.core.book import Book
from simplify.core.chapter import Chapter
from simplify.core.content import Content
from simplify.core.options import ManuscriptOptions
from simplify.core.page import Page
from simplify.core.project import Project
from simplify.core.ingredients import Ingredients
from simplify.core.inventory import Inventory
from simplify.core.idea import Idea

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'


"""
This module uses a traditional builder design pattern, with this module acting
as the defacto builder class, to create algorithms for use in the siMpLify
package. The director is any outside class or function which calls the builder
to create an algorithm and corresponding parameters.

Placing the builder functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the builder just
needs:

    from simplify import creator

And then to create any new algorithm, the function calls are very
straightforward and clear. For example, to create a siMpLify object, this is
all that's required:

    design = creator.make_object(object_type, **parameters)

Or, you can call the specific object builder directly, as follows:

    book = creator.make_book(**parameters)

Putting the builder in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported. And the user does not have to use the import * anti-pattern.

Using the __init__.py file for builder functions was inspired by a blog post,
which used the file for a factory design pattern, by Luna at:
https://www.bnmetrics.com/blog/builder-pattern-in-python3-simple-version

"""

""" Private Functions """

def _check_options(
        options: Union['ManuscriptOptions',
                       Dict[str, 'Outline']]) -> 'ManuscriptOptions':
    """Checks if 'options' is a dict and prepares ManuscriptOptions instance.

    Args:
        options (Union['ManuscriptOptions', Dict[str, Any]]):

    Returns:
        completed ManuscriptOptions instance.

    """
    if options is None:
        return ManuscriptOptions(options = {})
    elif isinstance(options, Dict):
        return make_options(options = options)
    else:
        return options

""" Public Functions """

def make_object(object_type: str, *args, **kwargs) -> object:
    """Calls appropriate function based upon 'types' passed.

    This method adds nothing to calling the functions directly. It is simply
    included for easier external iteration and/or for those who prefer a generic
    function for all builder functions.

    Args:
        object_type (str): name of an object type to be created. It should
            correspond to the suffix of one of the other functions in this
            module (following the prefix 'make_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            factory function.

    Returns:
        object: instance of new class created by the builder.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance exists.

    """
    if object_type in get_supported_types():
        return locals()['make_' + object_type ](*args, **kwargs)
    else:
        raise TypeError(' '.join(
            [object_type, 'is not a valid siMpLify object']))

def startup(
        idea: Union['Idea', Dict[str, Dict[str, Any]], str],
        inventory: Union['Inventory', str],
        ingredients: Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]) -> None:
    """Creates Idea, Inventory, and Ingredients instances.

    Args:
        idea (Union['Idea', Dict[str, Dict[str, Any]], str]): an instance of
            Idea, a nested Idea-compatible nested dictionary, or a string
            containing the file path where a file of a supoorted file type with
            settings for an Idea instance is located.
        inventory (Union['Inventory', str]): an instance of Inventory or a string
            containing the full path of where the root folder should be located
            for file output. A Inventory instance contains all file path and
            import/export methods for use throughout the siMpLify package.
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
    idea = make_idea(idea = idea)
    inventory = make_inventory(inventory = inventory, idea = idea)
    ingredients = make_ingredients(
        ingredients = ingredients,
        idea = idea,
        inventory = inventory)
    return idea, inventory, ingredients

def make_project(
        idea: Union['Idea', Dict[str, Dict[str, Any]], str],
        inventory: Optional[Union['Inventory', str]] = None,
        ingredients: Optional[Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]] = None,
        options: Optional[Union['ManuscriptOptions', Dict[str, 'Outline']]] = None,
        steps: Optional[Union[List[str], str]] = None,
        name: Optional[str] = None,
        auto_publish: Optional[bool] = True) -> 'Book':
    """
    Args:
        idea (Union['Idea', Dict[str, Dict[str, Any]], str]): an instance of
            Idea, a nested Idea-compatible nested dictionary, or a string
            containing the file path where a file of a supoorted file type with
            settings for an Idea instance is located.
        inventory (Optional[Union['Inventory', str]]): an instance of Inventory or a string
            containing the full path of where the root folder should be located
            for file output. A Inventory instance contains all file path and
            import/export methods for use throughout the siMpLify package.
            Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Inventory instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[Union[List[str], str]]): ordered names of Manuscript
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        options (Optional['ManuscriptOptions', Dict[str, 'Outline']]): either
            a ManuscriptOptions instance or a dictionary compatible with a
            ManuscriptOptions instance. Defaults to None.
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
            'ingredients' and 'options' must also be passed. Defaults to True,
            but the 'publish' method will not be called without 'ingredients'
            and 'options'.

    """
    idea, inventory, ingredients = startup(
        idea = idea,
        inventory = inventory,
        ingredients = ingredients)
    return Project(
        idea = idea,
        inventory = inventory,
        ingredients = ingredients,
        options = options,
        steps = steps,
        name = name,
        auto_publish = auto_publish)

def make_idea(idea: Union[Dict[str, Dict[str, Any]],  'Idea']) -> 'Idea':
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

def make_options(options: Dict[str, 'Outline']) -> 'ManuscriptOptions':
    """Creates a ManuscriptOptions instance.

    Args:
        options: Dict[str, 'Outline']: dict compatiable with ManuscriptOptions.

    Returns:
        ManuscriptOptions instance with 'options' dict.

    """
    return ManuscriptOptions(options = options)

def make_book(
        idea: Union[Dict[str, Dict[str, Any]], 'Idea'],
        inventory: Optional[Union['Inventory', str]],
        ingredients: Optional[Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]] = None,
        options: Optional[Union['ManuscriptOptions', Dict[str, 'Outline']]] = None,
        steps: Optional[Union[List[str], str]] = None,
        name: Optional[str] = None,
        auto_publish: Optional[bool] = True) -> 'Book':
    """Creates a Book instance.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        inventory (Optional[Union['Inventory', str]]): an instance of
            inventory or a string containing the full path of where the root
            folder should be located for file output. A inventory instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Inventory instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        options (Optional['ManuscriptOptions', Dict[str, 'Outline']]): either
            a ManuscriptOptions instance or a dictionary compatible with a
            ManuscriptOptions instance. Defaults to None.
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
            'ingredients' and 'options' must also be passed. Defaults to True,
            but the 'publish' method will not be called without 'ingredients'
            and 'options'.

    """
    idea, inventory, ingredients = startup(
        idea = idea,
        inventory = inventory,
        ingredients = ingredients)
    options = _check_options(options = options)
    return Book(
        idea = idea,
        inventory = inventory,
        ingredients = ingredients,
        steps = steps,
        name = name,
        auto_publish = auto_publish,
        options = options)

def make_chapter(
        name: Optional[str] = None,
        steps: Dict[str, str] = None,
        options: Optional[Union['ManuscriptOptions', Dict[str, 'Outline']]] = None,
        metadata: Dict[str, Any] = None) -> 'Chapter':
    """Creates a Chapter instance.

    Args:
        steps (Dict[str, str]): ordered names of steps as keys and particular
            techniques as methods.
        options (Optional['ManuscriptOptions', Dict[str, 'Outline']]): either
            a ManuscriptOptions instance or a dictionary compatible with a
            ManuscriptOptions instance. Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    Returns:
        Chapter instance.

    """
    options = _check_options(options = options)
    return Chapter(
        steps = steps,
        metadata = metadata,
        name = name,
        options = options)

def make_page(
        idea: 'Idea',
        outline: 'Outline',
        name: Optional[str] = None,
        sklearn_compatiable: Optional[bool] = True) -> 'Page':
    algorithm = outline.load()
    parameters = make_parameters(idea = idea, outline = outline)
    return Page(algorithm = algorithm, parameters = parameters, name = name)

def make_parameters(
        idea: 'Idea',
        outline: 'Outline',
        parameters: Optional[Dict[str, Any]] = None) -> 'Parameters':

    def make_selected(
            parameters: Dict[str, Any],
            outline: 'Outline') -> None:
        """Limits parameters to those appropriate to the outline.

        If 'outline.selected' is True, the keys from 'outline.defaults' are
        used to select the final returned parameters.

        If 'outline.selected' is a list of parameter keys, then only those
        parameters are selected for the final returned parameters.

        Args:
            outline (Outline): settings for parameters to be built.

        """
        if outline.selected:
            if isinstance(outline.selected, list):
                parameters_to_use = outline.selected
            else:
                parameters_to_use = list(outline.default.keys())
            new_parameters = {}
            for key, value in parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key: value})
            parameters = new_parameters
        return parameters

    def make_required(
            parameters: Dict[str, Any],
            outline: 'Outline') -> None:
        """Adds required parameters (mandatory additions) to 'parameters'.

        Args:
            outline (Outline): settings for parameters to be built.

        """
        try:
            parameters.update(outline.required)
        except TypeError:
            pass
        return parameters

    def make_search(
            parameters: Dict[str, Any],
            outline: 'Outline') -> None:
        """Separates variables with multiple options to search parameters.

        Args:
            outline (Outline): settings for parameters to be built.

        """
        space = {}
        if outline.hyperparameter_search:
            new_parameters = {}
            for parameter, values in parameters.items():
                if isinstance(values, list):
                    if any(isinstance(i, float) for i in values):
                        space.update(
                            {parameter: uniform(values[0], values[1])})
                    elif any(isinstance(i, int) for i in values):
                        space.update(
                            {parameter: randint(values[0], values[1])})
                else:
                    new_parameters.update({parameter: values})
            parameters = new_parameters
        return parameters, space

    def make_runtime(self, outline: 'Outline') -> None:
        """Adds parameters that are determined at runtime.

        The primary example of a runtime parameter throughout siMpLify is the
        addition of a random seed for a consistent, replicable state.

        The runtime variables should be stored as attributes in the Manuscript
        instance so that the values listed in outline.runtimes match those
        attributes to be added to parameters.

        Args:
            outline (Outline): settings for parameters to be built.

        """
        try:
            for key, value in outline.runtime.items():
                try:
                    parameters.update({key: getattr(self.author, value)})
                except AttributeError:
                    error = ' '.join('no matching runtime parameter',
                        key, 'found')
                    raise AttributeError(error)
        except (AttributeError, TypeError):
            pass
        return parameters

    if parameters is None:
        try:
            parameters = idea[outline.name]
        except KeyError:
            parameters = {}
    parameters = make_selected(parameters = parameters, outline = outline)
    parameters = make_required(parameters = parameters, outline = outline)
    parameters = make_runtime(parameters = parameters, outline = outline)
    return parameters

def make_outline(
        name: str,
        module: str,
        algorithm: str,
        **kwargs) -> 'Outline':
    outline = Outline(name = name, module = module, algorithm = algorithm)
    for key, value in kwargs.items():
        setattr(outline, key, value)
    return outline

def make_ingredients(
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
                           str],
        idea: 'Idea',
        inventory: 'Inventory') -> 'Ingredients':
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
                inventory.make_batch(
                    folder = getattr(self, ingredients))
                return Ingredients(
                    idea = idea,
                    inventory = inventory)
            except FileNotFoundError:
                error = ' '.join(
                    ['ingredients must be a file path, file folder',
                        'DataFrame, Series, None, Ingredients, or numpy',
                        'array'])
                raise TypeError(error)


def make_inventory(inventory: Union['Inventory', str], idea: 'Idea') -> 'Inventory':
    """Creates an Inventory instance from passed arguments.

    Args:
        inventory: Union['Inventory', str]: Inventory instance or root folder for one.
        idea ('Idea'): an Idea instance.

    Returns:
        Inventory instance, published.

    Raises:
        TypeError if inventory is not Inventory or str folder path.

    """
    if isinstance(inventory, Inventory):
        return inventory
    elif os.path.isdir(inventory):
        return Inventory(idea = idea, root_folder = inventory)
    else:
        error = 'inventory must be Inventory type or folder path'
        raise TypeError(error)


def get_supported_types() -> List[str]:
    """Removes 'make_' from object names in locals() to create a list of
    supported class types.

    Returns:
        List[str]: class types which have a corresponding factory function.

    """
    types = []
    for factory_function in locals().keys():
        if factory_function.startswith('make_'):
            types.append(factory_function[len('make_'):])
    return types
