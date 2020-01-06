"""
.. module:: conformers
:synopsis: validation and conformer decoraters and methods
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from configparser import ConfigParser
from functools import wraps
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.base import SimpleConformer
from simplify.core.idea import Idea
from simplify.core.data import Ingredient
from simplify.core.data import Ingredients
from simplify.core.inventory import Inventory
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


""" Conformer Decorators """

def SimplifyConformer(SimpleConformer):
    """Decorator for converting siMpLify objects to proper types.

    By default, this decorator checks the following parameters:
        idea
        ingredient
        ingredients
        inventory
        
    """

    def __init__(self, callable: Callable) -> None:
        """Sets initial conformer options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.conformers = {
            'idea': conform_idea,
            'ingredient': conform_ingredient,
            'ingredients': conform_ingredients,
            'inventory': conform_inventory}
        super().__init__()
        return self
    
def DataConformer(SimpleConformer):
    """Decorator for converting data objects to proper types.

    By default, this decorator checks any arguments that begin with 'x_', 'X_',
    'y_', or 'Y_' as wll as 'Y'. In all cases, the arguments are converted to 
    either 'x' or 'y' so that wrapped objects need only include generic 'x' and
    'y' in their parameters.
    
    The decorator also preserves pandas data objects with feature names even
    when the wrapped object converts the data object to a numpy array.
        
    """

    def __init__(self, callable: Callable) -> None:
        """Sets initial conformer options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.conformers = {
            'x': self._conform_df, 
            'y': self._conform_series}
        super().__init__()
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.
        
        All passed parameter names are converted to lower case to avoid issues
        with arguments passed with 'X' and 'Y' instead of 'x' and 'y'.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self._convert_names(arguments = arguments)
            self._store_names(arguments = arguments)
            result = self.callable(self, **arguments)
            result = self.apply(result = result)
            return result
        return wrapper

    """ Private Methods """
    
    def _conform_df(self, x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(x, np.ndarray):
            return pd.DataFrame(x, columns = self.x_columns)
        else:
            return x
    
    def _conform_series(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        if isinstance(y, np.ndarray):
            try:
                return pd.Series(y, name = self.y_name)
            except (AttributeError, KeyError):
                return pd.Series(y)
        else:
            return y
        
    def _convert_names(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts data arguments to truncated, lower-case names.
        
        Args:
            arguments (Dict[str, Any]): passed arguments to 'callable'.
            
        Returns:
            Dict[str, Any]: with any data arguments changed to either 'x' or 
                'y'.
            
        """
        new_arguments = {}
        for parameter, value in arguments.items():
            if parameter.startswith(('X', 'Y')):
                new_arguments[parameter.lower()] = value
            else:
                new_arguments[parameter] = value
            if parameter.startswith(('x_', 'y_')):
                new_arguments[parameter[0]] = value
            else:
                new_arguments[parameter] = value
        return new_arguments
    
    def _store_names(self, arguments: Dict[str, Any]) -> None:
        try:
            self.x_columns = arguments['x'].columns.values
            self.y_name = arguments['y'].name
        except KeyError:
            pass
        return self          

    """ Core siMpLify Methods """

    def apply(self, 
            result: Union[
                pd.DataFrame, 
                pd.Series, 
                np.ndarray,
                tuple[pd.DataFrame, pd.Series],
                tuple[np.ndarray, np.ndarray]]) -> (
                    Union[
                        pd.DataFrame, 
                        pd.Series, 
                        tuple[pd.DataFrame, pd.Series]]):
        """Converts values of 'result' to proper type.
        
        Args:
            result: Union[pd.DataFrame, pd.Series, np.ndarray, 
                tuple[pd.DataFrame, pd.Series], tuple[np.ndarray, np.ndarray]]:
                result of data analysis in several possible permutations.

        Returns:
            Union[pd.DataFrame, pd.Series, tuple[pd.DataFrame, pd.Series]]:
                result returned with all objects converted to pandas datatypes.

        """
        if isinstance(result, tuple):
            return tuple(
                self.conformers['x'](x = result[0]), 
                self.conformers['y'](y = result[1]))
        elif isinstance(result, np.ndarray):
            if result.ndim == 1:
                return self.conformers['y'](y = result)
            else:
                return self.conformers['x'](x = result)
        else:
            return result                 

def ColumnsConformer(SimpleConformer):
    """Decorator for creating column lists for wrapped methods."""

    def __init__(self, callable: Callable) -> None:
        """Sets initial conformer options.

        Args:
            callable (Callable): wrapped method, function, or callable class.

        """
        self.conformers = {
            'columns': self._conform_columns,
            'prefixes': self._conform_prefixes,
            'suffixes': self._conform_suffixes,
            'mask': self._conform_mask}
        super().__init__()
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'callable' to appropriate type.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.callable)
        @wraps(self.callable)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self.apply(arguments = arguments)
            return self.callable(self, **arguments)
        return wrapper
    
    """ Private Methods """

    def _conform_columns(self, 
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = [] 
        return arguments

    def _conform_prefixes(self, 
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = [] 
        return arguments
    
    
    def _conform_suffixes(self, 
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = [] 
        return arguments
    
    
    def _conform_mask(self, 
        arguments: Dict[str, Union[List[str], str]]) -> Dict[str, List[str]]:
        try:
            arguments['columns'] = listify(arguments['columns'])
        except KeyError:
            arguments['columns'] = [] 
        return arguments
        
    def make_columns(method: Callable, *args, **kwargs) -> Callable:
        """Decorator which creates a complete column list from passed arguments.

        If 'prefixes', 'suffixes', or 'mask' are passed to the wrapped method, they
        are combined with any passed 'columns' to form a list of 'columns' that are
        ultimately passed to the wrapped method.

        Args:
            method (Callable): wrapped method.

        Returns:
            Callable: with 'columns' parameter that combines items from 'columns',
                'prefixes', 'suffixes', and 'mask' parameters into a single list
                of column names using the 'make_column_list' method.

        """
        call_signature = signature(method)
        @wraps(method)
        def wrapper(*args, **kwargs):
            new_arguments = {}
            parameters = dict(call_signature.parameters)
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            unpassed = list(parameters.keys() - arguments.keys())
            if 'columns' in unpassed:
                columns = []
            else:
                columns = listify(arguments['columns'])
            try:
                columns.extend(
                    make_column_list(prefixes = arguments['prefixes']))
                del arguments['prefixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    make_column_list(suffixes = arguments['suffixes']))
                del arguments['suffixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    make_column_list(mask = arguments['mask']))
                del arguments['mask']
            except KeyError:
                pass
            if not columns:
                columns = list(columns.keys())
            arguments['columns'] = deduplicate(columns)
            # method.__signature__ = Signature(arguments)
            return method(**arguments)
        return wrapper

    def make_column_list(
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            prefixes: Optional[Union[List[str], str]] = None,
            suffixes: Optional[Union[List[str], str]] = None,
            mask: Optional[Union[List[bool]]] = None) -> None:
        """Dynamically creates a new column list from a list of columns, lists
        of prefixes, and/or boolean mask.

        This method serves as the basis for the 'column_lists' decorator which
        allows users to pass 'prefixes', 'columns', and 'mask' to a wrapped
        method with a 'columns' argument. Those three arguments are then
        combined into the final 'columns' argument.

        Args:
            df (DataFrame): pandas object.
            columns (list or str): column names to be included.
            prefixes (list or str): list of prefixes for columns to be included.
            suffixes (list or str): list of suffixes for columns to be included.
            mask (numpy array, list, or Series, of booleans): mask for columns
                to be included.

        Returns:
            column_names (list): column names created from 'columns',
                'prefixes', and 'mask'.

        """
        column_names = []
        try:
            for boolean, feature in zip(mask, list(df.columns)):
                if boolean:
                    column_names.append(feature)
        except TypeError:
            pass
        try:
            temp_list = []
            for prefix in listify(prefixes, default_null = True):
                temp_list = [col for col in df if col.startswith(prefix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            temp_list = []
            for prefix in listify(suffixes, default_null = True):
                temp_list = [col for col in df if col.endswith(suffix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            column_names.extend(listify(columns, default_null = True))
        except TypeError:
            pass
        return deduplicate(iterable = column_names)


""" Conformer Functions """

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
    idea = conform_idea(idea = idea)
    idea.project = project
    inventory = conform_inventory(
        inventory = inventory,
        idea = idea)
    inventory.project = project
    ingredients = conform_ingredients(
        ingredients = ingredients,
        inventory = inventory)
    ingredients.project = project
    return idea, inventory, ingredients

def conform_idea(idea: Union[Dict[str, Dict[str, Any]], 'Idea']) -> 'Idea':
    """Creates an Idea instance from passed argument.

    Args:
        idea (Union[Dict[str, Dict[str, Any]], 'Idea']): a dict, a str file path
            to an ini, csv, or py file with settings, or an Idea instance with a
            'configuration' attribute.

    Returns:
        Idea instance, properly configured.

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
        return Idea(configuration = idea)
    elif isinstance(idea, str):
        extension = str(Path(idea).suffix)[1:]
        configuration = locals()['_'.join(['_load_from', extension])](
            file_path = idea)
        return Idea(configuration = configuration)
    else:
        raise TypeError('idea must be Idea, str, or nested dict type')

def conform_inventory(
        inventory: Union['Inventory', str],
        idea: 'Idea') -> 'Inventory':
    """Creates an Inventory instance from passed arguments.

    Args:
        inventory: Union['Inventory', str]: Inventory instance or root folder
            for one.
        idea ('Idea'): an Idea instance.

    Returns:
        Inventory instance, properly configured.

    Raises:
        TypeError if inventory is neither an Inventory instance nor string
            folder path.

    """
    if isinstance(inventory, Inventory):
        return inventory
    elif isinstance(inventory, str):
        return Inventory(idea = idea, root_folder = inventory)
    else:
        raise TypeError('inventory must be Inventory type or folder path')

def conform_ingredients(
        ingredients: Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str,
            List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
            Dict[str, Union[
                'Ingredient', pd.DataFrame, pd.Series, np.ndarray, str]]],
        inventory: Optional['Inventory'] = None) -> 'Ingredients':
    """Creates an Ingredients instance.

    Args:
        ingredients (Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str, List[Union[pd.DataFrame, pd.Series, np.ndarray, str]],
            Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]]):
            Ingredients instance or information needed to create one.
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
    elif ingredients is None:
        return Ingredients()
    elif isinstance(ingredients, list):
        dfs = {}
        for i, ingredient in enumerate(ingredients):
            dfs.update({''.join(['df'], str(i)): conform_ingredient(
                ingredient = ingredient, 
                inventory = inventory)})
        return Ingredients(ingredients = dfs)
    elif isinstance(ingredients, dict):
        dfs = {}
        for name, ingredient in ingredients.items():
            dfs[name] = conform_ingredient(
                ingredient = ingredient, 
                inventory = inventory)
        return Ingredients(ingredients = dfs)
    elif isinstance(ingredients, (pd.Series, pd.DataFrame, np.ndarray, str)):
        return Ingredients(ingredients = {
            'df': conform_ingredient(
                ingredient = ingredients, 
                inventory = inventory)})
    else:
        raise TypeError(' '.join(
            ['ingredients must be a file path, file folder, DataFrame, Series',
             'None, Ingredients, or numpy array']))

def conform_ingredient(
        ingredient: Union['Ingredient', pd.DataFrame, pd.Series, np.ndarray,
            str],
        inventory: 'Inventory') -> 'Ingredient':
    """Creates an Ingredients instance.

    Args:
        ingredient (Union['Ingredient', pd.DataFrame, pd.Series, np.ndarray,
            str]): Ingredient instance or information needed to create one.
        inventory ('Inventory'): a Inventory instance.

    Returns:
        Ingredient instance, published.

    Raises:
        TypeError: if 'ingredient' is neither a file path, file folder,
            None, DataFrame, Series, numpy array, or Ingredient instance.

    """
    def get_ingredient(ingredient: str, inventory: 'Inventory') -> 'Ingredient':
        """Creates an Ingredient instance from a source file.

        Args:
            ingredient (str):
            inventory ('Inventory'): an Inventory instance.

        Returns:
            Ingredient instance.

        """
        try:
            return Ingredient(
                df = inventory.load(
                    folder = inventory.data,
                    file_name = ingredient))
        except FileNotFoundError:
            try:
                return Ingredient(df = inventory.load(file_path = ingredient))
            except FileNotFoundError:
                try:
                    inventory.create_batch(folder = ingredient)
                    return Ingredient()
                except FileNotFoundError:
                    raise FileNotFoundError('ingredient not found')
    if isinstance(ingredient, (pd.DataFrame, pd.Series)):
        return Ingredient(df = ingredient)
    elif isinstance(ingredient, str):
        if inventory is None:
            raise ValueError('inventory needed to load a data object')
        else:
            return get_ingredient(
                ingredient = ingredient,
                inventory = inventory)
    elif isinstance(ingredient, np.ndarrray):
        return Ingredient(df = pd.DataFrame(data = ingredient))
    else:
        raise TypeError(' '.join(
            ['ingredient must be a file path, file folder, DataFrame, Series',
             'None, or numpy array']))