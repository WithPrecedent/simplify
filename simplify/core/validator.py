"""
.. module:: validator
:synopsis: data and variable validation methods
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


""" Validator Decorator """

@dataclass
def Validator(MutableMapping):
    """Wraps python objects to conform arguments to proper type

    By default, validator checks the following parameters:
        data
        idea
        ingredients
        inventory

    Users can add more or edit the validation options by updating or setting
    a class instance (which has all dictionary methods).

    Or, if a 'validations' dictionary is passed when this class is instanced,
    those options will be used instead.

    """

    validations: Dict[str, Callable] = field(default_factory = dict)

    def __post_init__(self) -> None:
        """Sets initial validation options."""
        if not self.validations:
            self.validations = {
                'data': validate_data,
                'idea': validate_idea,
                'ingredients': validate_ingredients,
                'inventory': validate_inventory}
        return self

    """ Required Wrapper Method """

    def __call__(method: Callable, *args, **kwargs) -> Callable:
        """Converts arguments of 'method' to appropriate type.

        Args:
            method (Callable): wrapped method, function, or callable class.

        Returns:
            Callable: with all arguments converted to appropriate types.

        """
        call_signature = signature(method)
        @wraps(method)
        def wrapper(self, *args, **kwargs):
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
                    self.make_column_list(prefixes = arguments['prefixes']))
                del arguments['prefixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    self.make_column_list(suffixes = arguments['suffixes']))
                del arguments['suffixes']
            except KeyError:
                pass
            try:
                columns.extend(
                    self.make_column_list(mask = arguments['mask']))
                del arguments['mask']
            except KeyError:
                pass
            if not columns:
                columns = list(self.columns.keys())
            arguments['columns'] = deduplicate(columns)
            # method.__signature__ = Signature(arguments)
            return method(self, **arguments)
        return wrapper

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Callable:
        """Returns key in the 'validations' dictionary.

        Args:
            key (str): name of key to find.

        Returns:
            Callable: function to validate a variable.

        Raises:
            KeyError: if 'key' is not found in 'validation' dictionary.

        """
        try:
            return self.validations[key]
        except KeyError:
            raise KeyError(' '.join(
                [key, 'is not in', self.__class__,__.name__]))

    def __setitem__(self, key: str, value: Callabe) -> None:
        """Stoes arguments in 'validations' dictionary.

        Args:
            key (str): name of key to set.
            value (Callable): function to validate data.

        """
        self.validations[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes key in the 'validations' dictionary.

        Args:
            key (str): name of key to find.

        """
        try:
            del self.validations[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the 'validations' dictionary.

        Returns:
            Iterable stored in the 'validations' dictionary.

        """
        return iter(self.validations)

    def __len__(self) -> int:
        """Returns length of the 'validations' dictionary.

        Returns:
            int: of length of 'validations' dictionary.

        """
        return len(self.validations)

    """ Core siMpLify Methods """

    def apply(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Converts values of 'arguments' to proper types.

        Args:
            arguments (Dict[str, Any]): arguments with values to be converted.

        Returns:
            Dict[str, Any]: arguments with converted values.

        """
        for argument, convertor in self.validators.items():
            try:
                arguments[argument] = converter(arguments[argument])
            except KeyError:
                pass
        return arguments

""" Validator Decorator """

def validator(method: Callable, *args, **kwargs) -> Callable:
    """Converts arguments of 'method' to appropriate type.

    Args:
        method (Callable): wrapped method.

    Returns:
        Callable: with all arguments converted to appropriate validations.

    """
    call_signature = signature(method)
    @wraps(method)
    def wrapper(self, *args, **kwargs):
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
                self.make_column_list(prefixes = arguments['prefixes']))
            del arguments['prefixes']
        except KeyError:
            pass
        try:
            columns.extend(
                self.make_column_list(suffixes = arguments['suffixes']))
            del arguments['suffixes']
        except KeyError:
            pass
        try:
            columns.extend(
                self.make_column_list(mask = arguments['mask']))
            del arguments['mask']
        except KeyError:
            pass
        if not columns:
            columns = list(self.columns.keys())
        arguments['columns'] = deduplicate(columns)
        # method.__signature__ = Signature(arguments)
        return method(self, **arguments)
    return wrapper


""" Validator Functions """

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
    idea = validate_idea(project = project, idea = idea)
    inventory = validate_inventory(
        project = project,
        inventory = inventory,
        idea = idea)
    ingredients = validate_ingredients(
        project = project,
        ingredients = ingredients,
        idea = idea,
        inventory = inventory)
    return idea, inventory, ingredients

def validate_idea(
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

def validate_inventory(
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

def validate_ingredients(
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
                inventory.validate_batch(
                    folder = getattr(self, ingredients))
                return Ingredients(
                    idea = idea,
                    inventory = inventory)
            except FileNotFoundError:
                raise TypeError(' '.join(
                    ['ingredients must be a file path, file folder',
                        'DataFrame, Series, None, Ingredients, or numpy',
                        'array']))