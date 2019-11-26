"""
.. module:: factory
:synopsis: the factory design pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""
from configparser import ConfigParser
from importlib import import_module
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.idea import Idea
from simplify.core.ingredients import Ingredients
from simplify.core.library import Library
from simplify.project import Project

"""
This module uses a traditional factory design pattern, with this module acting
as the defacto class, to create some of the primary class objects used in the
siMpLify packages.
Placing the factory functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the factory just
needs:

    from simplify import factory

And then to create any new classes, the function calls are very straightforward
and clear. For example, to create a new Idea, this is all that is required:

    factory.create_idea(parameters)

Putting the factory in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

Using the __init__.py file for factory functions was inspired by a blog post by
Luna at:
https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version

"""

def startup(
    idea: Union[Dict[str, Dict[str, Any]], str, 'Idea'],
    library: Union[str, 'Library'],
    ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
                       str]) -> None:
    """Creates Idea, Library, and Ingredients instances.

    Args:
        idea: Union[Dict[str, Dict[str, Any]], str, 'Idea']: Idea instance or
            needed information to create one.
        library: Union[str, 'Library']: Library instance or root folder for one.
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str]: Ingredients instance or information needed to create one.

    Returns:
        Idea, Library, Ingredients instances, published.

    """
    idea = create_idea(idea = idea)
    library = create_library(library = library, idea = idea)
    ingredients = create_ingredients(
        ingredients = ingredients,
        idea = idea,
        library = library)
    return idea, library, ingredients

def create(object_type: str, *args, **kwargs):
    """Calls appropriate function based upon 'types' passed.
    This method adds nothing to calling the functions directly. It is simply
    included for easier external iteration and/or for those who prefer a generic
    function for all factories.
    Args:
        object_type (str): name of package type to be created. It should
            correspond to the suffix of one of the other functions in this
            module (following the prefix 'create_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            factory function.

    Returns:
        instance of new class created by the factory.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance designated in 'types'.

    """
    if object_type in get_supported_types():
        return locals()['create_' + object_type ](*args, **kwargs)
    else:
        error = ' '.join([object_type, ' is not a valid class type'])
        raise TypeError(error)

def create_idea(idea: Union[Dict[str, Dict[str, Any]], str, 'Idea']) -> 'Idea':
    """Creates an Idea instance from passed argument.

    Args:
        idea (Union[Dict[str, Dict[str, Any]], str, 'Idea']): can either be a
            dict, a str file path to an ini, csv, or py file with settings, or
            an Idea instance with a configuration attribute.

    Returns:
        Idea instance, published.

    Raises:
        TypeError: if 'idea' is neither a dict, str, nor Idea instance.

    """
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

def create_library(library: Union[str, 'Library'], idea: 'Idea') -> 'Library':
    """Creates an Library instance from passed arguments.

    Args:
        library: Union[str, 'Library']: Library instance or root folder for one.
        idea ('Idea'): an Idea instance.

    Returns:
        Library instance, published.

    Raises:
        TypeError if library is not Library or str folder path.

    """
    if isinstance(library, Library):
        return library
    elif os.path.isdir(library):
        return Library(idea = idea, root_folder = library)
    else:
        error = 'library must be Library type or folder path'
        raise TypeError(error)

def create_ingredients(
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
                           str],
        idea: 'Idea',
        library: 'Library') -> 'Ingredients':
    """Creates an Ingredients instance.

    If 'ingredients' is an Ingredients instance, it is returned unchanged.
    If 'ingredients' is a pandas data container, an Ingredients is created
        with that data container as the 'df' attribute which is returned.
    If 'ingredients' is a file path, the file is loaded into a DataFrame and
        assigned to 'df' in an Ingredients instance which is returned.
    If 'ingredients' is a file folder, a glob in the shared Library is
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
        library ('Library'): a Library instance.

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
            library = library,
            df = ingredients)
    elif isinstance(ingredients, np.ndarray):
        return Ingredients(
            idea = idea,
            library = library,
            df =  pd.DataFrame(data = getattr(self, ingredients)))
    elif isinstance(ingredients, None):
        return Ingredients(
            idea = idea,
            library = library)
    elif isinstance(ingredients, str):
        try:
            df = library.load(
                folder = library.data,
                file_name = ingredients)
            return Ingredients(
                idea = idea,
                library = library,
                df = df)
        except FileNotFoundError:
            try:
                library.create_batch(
                    folder = getattr(self, ingredients))
                return Ingredients(
                    idea = idea,
                    library = library)
            except FileNotFoundError:
                error = ' '.join(
                    ['ingredients must be a file path, file folder',
                        'DataFrame, Series, None, Ingredients, or numpy',
                        'array'])
                raise TypeError(error)

def create_project(
    idea: 'Idea',
    library: 'Library',
    ingredients: 'Ingredients') -> 'Project':
    """Creates Project from idea, library, and ingredients

    Args:
        idea ('Idea'): an Idea instance.
        library ('Library'): a Library instance.
        ingredients ('Ingredients'): an Ingredients instance.

    Returns:
        Project based upon passed attributes.

    """
    if idea.configuration['general']['verbose']:
        print('Starting siMpLify Project')
    return Project(
        idea = idea,
        library = library,
        ingredients = ingredients)

def get_supported_types():
    """Removes 'create_' from object names in locals() to create a list of
    supported class types.

    Returns:
        List: class types which have a corresponding factory function.

    """
    types = []
    for factory_function in locals().keys():
        if factory_function.startswith('create_'):
            types.append(factory_function[len('create_'):])
    return types
