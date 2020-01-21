"""
.. module:: ingredients
:synopsis: data containment made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Collection
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy import datetime64
import pandas as pd
from pandas.api.types import CategoricalDtype

from simplify.core.states import SimpleState
from simplify.core.types import SimpleType
from simplify.core.utilities import listify


@dataclass
class Ingredients(MutableMapping):
    """Contains a collection of data objects (Ingredient instances).

    Args:
        project (Optional['Project']): related Project or subclass instance.
            Defaults to None.
        ingredients (Optional[Dict[str, 'Ingredient']]): keys are names of the
            data objects (e.g. 'x', 'y_train', etc.) and values are Ingredient
            instances storing pandas data objects. Defaults to an empty
            dictionary.
        default (Optional[str]): name of data object in 'proxies' to apply
            methods to by default, if a particular ingredient is not accessed.
            Defaults to 'data'.
        name (Optional[str]): this should be used to distinguish multiple sets
            of unrelated data. Ordinarily, it is not needed. Defaults to
            'proxies'.

    """
    project: Optional['Project'] = None
    ingredients: Optional[Dict[str, 'Ingredient']] = field(
        default_factory = dict)
    default: Optional[str] = 'data'
    name: Optional[str] = 'ingredient'

    def __post_init__(self) -> None:
        """Sets default attributes."""
        for key, value in self.ingredients.items():
            value.ingredients = self
        self.proxies = DataProxies(ingredients = self)
        self.wildcards = ['all', 'train', 'test', 'val']
        self.state = SimpleState(
            states = ['unsplit', 'xy', 'train_test', 'train_val', 'full'])
        self.types = SimpleType(
            types = {
                'boolean': bool,
                'float': float,
                'integer': int,
                'string': object,
                'categorical': CategoricalDtype,
                'list': list,
                'datetime': datetime64,
                'timedelta': timedelta})
        self.default_values = {}
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> 'Ingredient':
        """Returns Ingredient instance from 'ingredients' based on 'proxies'.

        If there are no matches, the method searches for a matching wildcard
        and tries to return the attribute matching that wildcard.

        Args:
            key (str): name of key in 'ingredients'.

        Returns:
            Ingredient: item stored as a 'ingredients' value.

        Raises:
            KeyError: if 'key' is not found in 'ingredients' and is not a
                wildcard.

        """
        try:
            return self.proxies[key]
        except KeyError:
            if item in self.wildcards:
                return getattr(self, key)
            else:
                raise KeyError(' '.join([key, 'is not in', self.name]))

    def __setitem__(self, key: str, value: 'Ingredient') -> None:
        """Sets 'key' in 'ingredients' to 'value' based on 'proxies'.

        Args:
            key (str): name of key in 'ingredients'.
            value ('Ingredient'): value to pair with 'key' in 'ingredients'.

        """
        self.proxies[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes item in 'ingredients' based on 'proxies'.

        Args:
            key (str): name of key in 'ingredients'.

        """
        try:
            del self.proxies[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'ingredients'.

        Returns:
            Iterable of 'ingredients'.

        """
        return iter(self.ingredients)

    def __len__(self) -> int:
        """Returns length of 'ingredients'.

        Returns:
            Integer of length of 'ingredients'.

        """
        return len(self.ingredients)

    """ Other Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Tries to find attribute inside other attributes.

        Args:
            attribute (str): attribute to look for in other attributes.

        Returns:
            Any: attribute inside specific attributes.

        Raises:
            AttributeError: if 'attribute' is not found in any of the searched
                attributes.

        """
        try:
            return getattr(self.__dict__['proxies'][self.default], attribute)
        except KeyError:
            try:
                # Looks for 'attribute' as an item in 'proxies'.
                return self.__dict__['proxies'][attribute]
            except KeyError:
                pass

    def __add__(self, other: 'Ingredient') -> None:
        """Adds 'other' to 'proxies'.

        Args:
            other ('Ingredient'): an 'Ingredient' instance.

        """
        self.add(name = other.name, ingredient = other)
        return self

    def __iadd__(self, other: 'Ingredient') -> None:
        """Adds 'other' to 'proxies'.

        Args:
            other ('Ingredient'): an 'Ingredient' instance.

        """
        self.add(name = other.name, ingredient = other)
        return self

    """ Public Methods """

    def add(self, name: str, ingredient: 'Ingredient') -> None:
        """Adds 'ingredient' to 'proxies'.

        Args:
            ingredient ('Ingredient'): an 'Ingredient' instance.

        """
        self.ingredients[name] = ingredient
        return self


@dataclass
class Ingredient(Collection):
    """A container of a pandas data object with related metadata.

    Args:
        data (Union[pd.DataFrame, pd.Series]): a stored pandas data object.
        datatypes (Optional[Dict[str, str]]): keys are column names and values
            are siMpLify proxy datatypes. Defaults to an empty dictionary.
        prefixes (Optional[Dict[str, str]]): keys are column prefixes and
            values are siMpLify proxy datatypes. Defaults to an empty
            dictionary.
        ingredients (Optional['Ingredients']): related 'Ingredients' instance.
            Defaults to None.
        name (Optional[str]): this should match the key used in a related
            'Ingredients' instance, if one exists. This is used when any of the
            'add' methods is used to add this Ingredient instance to an
            'Ingredients' instance. Defaults to 'ingredient'.

    """
    data: Union[pd.DataFrame, pd.Series]
    datatypes: Optional[Dict[str, str]] = field(default_factory = dict)
    prefixes: Optional[Dict[str, str]] = field(default_factory = dict)
    ingredients: Optional['Ingredients'] = None
    name: Optional[str] = 'ingredient'

    def _post_init__(self) -> None:
        """Sets 'state' for data exporting."""
        self.state = SimpleState(states = ['raw', 'interim', 'processed'])
        return self

    """ Required ABC Methods """

    def __contains__(self, key: str) -> bool:
        """Returns if 'key' is in 'columns' of the 'data' attribute.

        Args:
            key (str): item to check for existence.

        Returns:
            boolean value if 'key' is in 'data.columns'.

        """
        return key in self.data.columns

    def __iter__(self) -> Iterable:
        """Returns 'iterrows' method for 'data' attribute."""
        return self.data.iterrows()

    def __len__(self) -> int:
        """Returns integer length of 'data' attribute."""
        return len(self.data)

    """ Other Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Returns column lists or state-dependent variables.

        Args:
            attribute (str): attribute to look for.

        Returns:
            Any: attribute in state or List[str] of columns.

        Raises:
            AttributeError: if 'attribute' is not found in any of the searched
                attributes.

        """
        # Returns appropriate lists of columns with datatype 'attribute'.
        if attribute in self.ingredients.types:
            return self._get_columns_by_type(datatype = attribute[:-1])
        # Combines 'floats' and 'integers' into 'numerics'.
        elif attribute in ['numerics']:
            return self.floats + self.integers
        # Returns list of 'dropped_columns' by comparing to '_start_columns'.
        elif attribute in ['dropped_columns']:
            return self._start_columns - list(self.datatypes.keys())
        else:
            # Tries to apply method to pandas object itself.
            try:
                return getattr(self.data, attribute)
            except:
                raise AttributeError(' '.join(
                    [attribute, 'is not in', self.name]))

    def __add__(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with 'proxies'.

        Args:
            other (Union[pd.DataFrame, pd.Series, 'Ingredient']): a pandas data
                object or another 'Ingredient' instance.

        """
        self.add(ingredient = other)
        return self

    def __iadd__(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with 'proxies'.

        Args:
            other (Union[pd.DataFrame, pd.Series, 'Ingredient']): a pandas data
                object or another 'Ingredient' instance.

        """
        self.add(ingredient = other)
        return self

    """ Private Methods """

    def _check_columns(self, columns: Optional[List[str]] = None) -> List[str]:
        """Returns 'columns' keys if columns doesn't exist.

        Args:
            columns (Optional[List[str]]): column names.

        Returns:
            if columns is not None, returns columns, otherwise, the keys of
                the 'columns' attribute are returned.

        """
        if columns:
            return listify(columns)
        else:
            return list(self.datatypes.keys())

    def _convert_columns(self, raise_errors: Optional[bool] = False) -> None:
        """Converts column data to the datatypes in 'datatypes' dictionary.

        Args:
            raise_errors (Optional[bool]): whether errors should be raised when
                converting datatypes or ignored. Selecting False (the default)
                risks type mismatches between the datatypes listed in the
                'datatypes' dict and 'data', but it prevents the program from
                being halted if an error is encountered.

        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.datatypes.items():
            if not datatype in ['string']:
                self.data[column].astype(
                    dtype = self.ingredients.types[datatype],
                    copy = False,
                    errors = raise_errors)
        # Attempts to downcast datatypes to simpler forms if possible.
        self.downcast()
        return self

    def _crosscheck_columns(self) -> None:
        """Harmonizes 'datatypes' dictionary with 'data' columns attribute."""
        for column in list(self.data.columns.values()):
            if column not in self.datatypes:
                self.datatypes[column] = self._infer_type(column = column)
        for column in list(self.datatypes.keys()):
            if column not in self.data.columns:
                del self.datatypes[column]
        return self

    def _get_columns_by_type(self, datatype: str) -> List[str]:
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'datatypes'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [
            key for key, value in self.datatypes.items() if value == datatype]

    def _get_indices(self, columns: Union[List[str], str]) -> List[bool]:
        """Gets column indices for a list of column names.

        Args:
            columns (Union[List[str], str]): name(s) of columns for which
                indices are sought.

        Returns:
            bool mask for columns matching 'columns'.

        """
        return [
            self.data.columns.get_loc(column) for column in listify(columns)]

    def _infer_type(self, column: pd.Series) -> str:
        """Infers column datatype of a single column.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        Non-standard python datatypes cannot be inferred.

        Args:
            column (pd.Series): column for which datatype is sought.

        Returns:
            str: name of siMpLify proxy datatype name.

        """
        try:
            self.datatypes[column] = self.data.select_dtypes(
                    include = [datatype]).columns.to_list()[0]
        except AttributeError:
            pass
        return self

    def _initialize_datatypes(self) -> None:
        """Initializes datatypes for stored pandas data object."""
        if not self.datatypes:
            self.infer_datatypes()
        else:
            self._crosscheck_columns()
        self._start_columns = list(self.data.columns.values)
        return self

    """ Public Methods """

    def add(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with 'proxies'.

        Args:
            ingredient (Union[pd.DataFrame, pd.Series, 'Ingredient']): a pandas
                data object or another 'Ingredient' instance.

        ToDo:
            Add functionality for all cases.

        """
        return self

    def apply(self, callable: Callable, **kwargs) -> None:
        """Applies 'callable' to 'data' atttribute with **kwargs).

        Args:
            callable (Callable): to be applied to 'data'.
            kwargs: any arguments to be passed to 'callable'.

        """
        self.data = callable(self.data, **kwargs)
        return self

    def change_datatype(self,
            columns: [Union[List[str], str]],
            datatype: str) -> None:
        """Changes column datatypes of 'columns'.

        The 'datatype' becomes the new datatype for the columns in both the
        'datatypes' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            columns ([Union[List[str], str]]): column name(s) for datatypes to
                be changed.
            datatype (str): contains name of the datatype to convert the
                columns.

        """
        for column in listify(columns):
            self.datatypes[column] = datatype
        self._convert_columns()
        return self

    def downcast(self,
            columns: Optional[Union[List[str], str]] = None,
            allow_unsigned: Optional[bool] = True) -> None:
        """Decreases memory usage by downcasting datatypes.

        For numerical datatypes, the method attempts to cast the data to
        unsigned integers if possible when 'allow_unsigned' is True. If more
        data might be added later which, in the same column, has values less
        than zero, 'allow_unsigned' should be set to False.

        Args:
            columns (Optional[Union[List[str], str]]): columns to downcast.
            allow_unsigned (Optional[bool]): whether to allow downcasting to
                unsigned integers.

        Raises:
            KeyError: if column in 'columns' is not in 'data'.

        """
        for column in self._check_columns(columns):
            if self.datatypes[column] in ['boolean']:
                self.data[column] = self.data[column].astype(bool)
            elif self.datatypes[column] in ['integer', 'float']:
                try:
                    self.data[column] = pd.to_numeric(
                        self.data[column],
                        downcast = 'integer')
                    if min(self.data[column] >= 0) and allow_unsigned:
                        self.data[column] = pd.to_numeric(
                            self.data[column],
                            downcast = 'unsigned')
                except ValueError:
                    self.data[column] = pd.to_numeric(
                        self.data[column],
                        downcast = 'float')
            elif self.datatypes[column] in ['categorical']:
                self.data[column] = self.data[column].astype('category')
            elif self.datatypes[column] in ['list']:
                self.data[column].apply(
                    listify,
                    axis = 'columns',
                    inplace = True)
            elif self.datatypes[column] in ['datetime']:
                self.data[column] = pd.to_datetime(self.data[column])
            elif self.datatypes[column] in ['timedelta']:
                self.data[column] = pd.to_timedelta(self.data[column])
            else:
                raise KeyError(' '.join([column, ' is not in data']))
        return self

    def drop_columns(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Drops specified 'columns'.

        Args:
            columns (Optional[Union[List[str], str]]): columns to drop.

        """
        try:
            self.data.drop(columns, axis = 'columns', inplace = True)
        except TypeError:
            self.data.drop(columns, inplace = True)
        return self

    def get_series(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Creates a Series (row) with the 'datatypes' dict.

        Default values are added to each item in the series so that pandas does
        not automatically infer the datatype when a value is passed.

        Args:
            columns (Optional[Union[List[str], str]]): index names for pandas
                Series.

        Returns:
            pd.Series with index names matching 'columns' keys and datatypes
                matching 'datatypes' values.

        """
        row = pd.Series(index = self._check_columns(columns = columns))
        # Fills series with default_values based on datatype.
        for column, datatype in self.datatypes.items():
            row[column] = self.ingredients.default_values[datatype]
        return row

    def indexify(self,
            name: Optional[str] = 'index_universal',
            assign_index: Optional[bool] = False) -> None:
        """Creates a unique integer index for each row.

        Args:
            name (Optional[str]): column name for the index. Defaults to
                'index_universal'.
            assign_index (Optional[bool]): indicates whether the index
                'column' should be made the actual index of the DataFrame.
                Defaults to False.

        Raises:
            TypeError: if 'data' is not a DataFrame (usually because it is a
                pandas Series).

        """
        try:
            self.data[name] = range(1, len(self.data.index) + 1)
            self.datatypes.update({name, 'integer'})
            if assign_index:
                self.data.set_index(name, inplace = True)
        except (TypeError, AttributeError):
            raise TypeError('To add an index, data must be a pandas DataFrame')
        return self

    def infer_datatypes(self) -> None:
        """Infers column datatypes and adds those datatypes to 'datatypes'.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        This methods also allows the user to choose which datatypes to look for
        by changing the options in 'types'.

        Non-standard python datatypes cannot be inferred.

        """
        try:
            for proxy, datatype in self.types:
                type_columns = data.select_dtypes(
                    include = [datatype]).columns.to_list()
                self.datatypes.update(
                    dict.fromkeys(type_columns, proxy))
        except AttributeError:
            pass
        return self


@dataclass
class DataProxies(MutableMapping):

    ingredients: 'Ingredients'
    test_suffixes: Dict[str, str] = field(default_factory = dict)
    train_suffixes: Dict[str, str] = field(default_factory = dict)

    def __post_init__(self) -> None:
        if not self.test_suffixes:
            self.test_suffixes = {
                'unsplit': None,
                'xy': '',
                'train_test': '_test',
                'train_val': '_test',
                'full': '_train'}
        if not self.train_suffixes:
            self.train_suffixes = {
                'unsplit': None,
                'xy': '',
                'train_test': '_train',
                'train_val': '_train',
                'full': '_train'}
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> 'Ingredient':
        """Returns 'Ingredient' based upon current 'state'.

        Args:
            key (str): name of key in 'ingredients'.

        Returns:
            'Ingredient': an 'Ingredient' instance stored in 'ingredients'
                based on 'state' in 'ingredients'

        Raises:
            ValueError: if access to train or test data is sought before data
                has been split.

        """
        try:
            contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
            if getattr(self, dictionary)[self.ingredients.state] is None:
                raise ValueError(''.join(['Train and test data cannot be',
                    'accessed until data is split']))
            else:
                new_key = ''.join(
                    [key[0], getattr(self, dictionary)[self.ingredients.state]])
                return self.ingredients.ingredients[new_key]
        except TypeError:
            return self.ingredients.ingredients[key]

    def __setitem__(self, key: str, value: 'Ingredient') -> None:
        """Sets 'key' to 'Ingredient' based upon current 'state'.

        Args:
            key (str): name of key to set in 'ingredients'.
            value ('Ingredient'): 'Ingredient' instance to be added to
                'ingredients'.

        """
        try:
            contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
            new_key = ''.join(
                [key[0], getattr(self, dictionary)[self.ingredients.state]])
            self.ingredients.ingredients[new_key] = value
        except ValueError:
            self.ingredients.ingredients[key] = value

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in the 'ingredients' dictionary.

        Args:
            key (str): name of key in the 'ingredients' dictionary.

        """
        try:
            contents = '_'.join([key.rsplit('_', 1), 'suffixes'])
            new_key = ''.join(
                [key[0], getattr(self, dictionary)[self.ingredients.state]])
            self.ingredients.ingredients[new_key] = value
        except ValueError:
            try:
                del self.ingredients.ingredients[key]
            except KeyError:
                pass

    def __iter__(self) -> NotImplementedError:
        raise NotImplementedError('DataProxies does not implement an iterable')

    def __len__(self) -> int:
        raise NotImplementedError('DataProxies does not implement length')


""" Validator Functions """

def create_ingredients(
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
        inventory ('Inventory'): a Inventory instance.

    Returns:
        Ingredients instance, properly configured.

    Raises:
        TypeError: if 'ingredients' is neither a file path, file folder,
            None, DataFrame, Series, numpy array, or Ingredients instance.

    """
    if isinstance(ingredients, Ingredients):
        return ingredients
    elif ingredients is None:
        return Ingredients()
    elif isinstance(ingredients, list):
        data = {}
        for i, ingredient in enumerate(ingredients):
            data.update({''.join(['data'], str(i)): create_ingredient(
                ingredient = ingredient,
                inventory = inventory)})
        return Ingredients(ingredients = data)
    elif isinstance(ingredients, dict):
        data = {}
        for name, ingredient in ingredients.items():
            data[name] = create_ingredient(
                ingredient = ingredient,
                inventory = inventory)
        return Ingredients(ingredients = data)
    elif isinstance(ingredients, (pd.Series, pd.DataFrame, np.ndarray, str)):
        return Ingredients(ingredients = {
            'data': create_ingredient(
                ingredient = ingredients,
                inventory = inventory)})
    else:
        raise TypeError(' '.join(
            ['ingredients must be a file path, file folder, DataFrame, Series',
             'None, Ingredients, or numpy array']))

def create_ingredient(
        ingredient: Union['Ingredient', pd.DataFrame, pd.Series, np.ndarray,
            str],
        inventory: 'Inventory') -> 'Ingredient':
    """Creates an Ingredients instance.

    Args:
        ingredient (Union['Ingredient', pd.DataFrame, pd.Series, np.ndarray,
            str]): Ingredient instance or information needed to create one.
        inventory ('Inventory'): a Inventory instance.

    Returns:
        Ingredient instance, properly configured.

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
                data = inventory.load(
                    folder = inventory.data,
                    file_name = ingredient))
        except FileNotFoundError:
            try:
                return Ingredient(data = inventory.load(file_path = ingredient))
            except FileNotFoundError:
                try:
                    inventory.create_batch(folder = ingredient)
                    return Ingredient()
                except FileNotFoundError:
                    raise FileNotFoundError('ingredient not found')
    if isinstance(ingredient, (pd.DataFrame, pd.Series)):
        return Ingredient(data = ingredient)
    elif isinstance(ingredient, str):
        if inventory is None:
            raise ValueError('inventory needed to load a data object')
        else:
            return get_ingredient(
                ingredient = ingredient,
                inventory = inventory)
    elif isinstance(ingredient, np.ndarrray):
        return Ingredient(data = pd.DataFrame(data = ingredient))
    else:
        raise TypeError(' '.join(
            ['ingredient must be a file path, file folder, DataFrame, Series',
             'None, or numpy array']))
