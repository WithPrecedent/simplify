"""
.. module:: conformer
:synopsis: validation and conformer decorater and methods
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import update_wrapper
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


""" Conformer Decorator """

def Conformer(object):
    """Wraps python objects to conform arguments to the proper type.

    By default, validator checks the following parameters:
        idea
        ingredient
        ingredients
        inventory

    Users can add more or edit the conformer options by updating or setting
    a class instance (which has all dictionary methods).

    Or, if a 'conformers' dictionary is passed when this class is instanced,
    those options will be used instead.

    """

    def __init__(self,
            method: Callable,
            conformers: Optional[Dict[str, Callable]] = None) -> None:
        """Sets initial conformer options.

        Args:
            method (Callable): wrapped method, function, or callable class.

        """
        update_wrapper(self, method)
        self.method = method
        if self.conformers is None:
            self.conformers = {
                'idea': conform_idea,
                'ingredient': conform_ingredient,
                'ingredients': conform_ingredients,
                'inventory': conform_inventory}
        return self

    """ Required Wrapper Method """

    def __call__(self) -> Callable:
        """Converts arguments of 'method' to appropriate type.


        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(self.method)
        @wraps(self.method)
        def wrapper(self, *args, **kwargs):
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            arguments = self.apply(arguments = arguments)
            return self.method(self, **arguments)
        return wrapper

    """ Core siMpLify Methods """

    def apply(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts values of 'arguments' to proper types.

        Args:
            arguments (Dict[str, Any]): arguments with values to be converted.

        Returns:
            Dict[str, Any]: arguments with converted values.

        """
        for argument, conformer in self.conformers.items():
            try:
                arguments[argument] = conformer(arguments[argument])
            except KeyError:
                pass
        return arguments


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
            Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]],
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
            Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray, str]]],
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
    if (isinstance(ingredients, Ingredients)
            or issubclass(ingredients, Ingredients)):
        return ingredients
    elif isinstance(ingredients, list):
        dfs = {}
        for i, ingredient in enumerate(ingredients):
            dfs.update({''.join(['df'], str(i)): conform_ingredient(
                ingredient = ingredient)})
        return Ingredients(ingredients = dfs)
    elif instance(ingredients, dict):
        return Ingredients(ingredients = ingredients)
    elif isinstance(ingredients, None):
        return Ingredients()
    elif isinstance(ingredients, (pd.Series, pd.DataFrame, np.ndarray, str)):
        return Ingredients(ingredients = {
            'df': conform_ingredient(ingredient = ingredients)})
    else:
        raise TypeError(' '.join(
            ['ingredients must be a file path, file folder, DataFrame, Series',
             'None, Ingredients, or numpy array']))

def conform_ingredient(
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str],
        idea: 'Idea') -> 'Ingredient':
    """Creates an Ingredients instance.

    Args:
        ingredients (Union['Ingredient', pd.DataFrame, pd.Series, np.ndarray,
            str]): Ingredient instance or information needed to create one.
        idea ('Idea'): an Idea instance.
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
            inventory ('Inventory')

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
        return ingredient

    if isinstance(ingredient, (pd.DataFrame, pd.Series)):
        return ingredient
    elif isinstance(ingredient, str):
        if inventory is None:
            raise ValueError('inventory needed to load a data object')
        else:
            return get_ingredient(
                ingredient = ingredient,
                inventory = inventory)
    elif isinstance(ingredient, np.ndarrray):
        return pd.DataFrame(data = ingredient)
    else:
        raise TypeError(' '.join(
            ['ingredient must be a file path, file folder, DataFrame, Series',
             'None, or numpy array']))