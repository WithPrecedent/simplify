"""
.. module:: data
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
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from numpy import datetime64
import pandas as pd
from pandas.api.types import CategoricalDtype

from simplify.core.base import SimpleOptions
from simplify.core.base import SimpleState
from simplify.core.base import SimpleType
from simplify.core.conformers import ColumnsConformer
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class Ingredients(MutableMapping):
    """Container for storing Ingredient instances.

    Args:
        project (Optional['Project']): related Project or subclass instance.
            Defaults to None.
        ingredients (Optional[Dict[str, 'Ingredient']]): keys are names of the
            data objects (e.g. 'x', 'y_train', etc.) and values are Ingredient
            instances storing pandas data objects. Defaults to an empty
            dictionary.
        default (Optional[str]): name of data object in 'ingredients' to apply
            methods to by default, if a particular df is not specified. Defaults
            to 'df'.
        name (Optional[str]): this should be used to distinguish multiple sets
            of related data. Ordinarily, it is not needed. Defaults to
            'ingredients'.

    """
    project: Optional['Project'] = None
    ingredients: Optional[Dict[str, 'Ingredient']] = field(
        default_factory = dict)
    default: Optional[str] = 'df'
    name: Optional[str] = 'ingredient'

    def _post_init__(self) -> None:
        """Sets default attributes."""
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
        return self

    """ Required ABC Methods """

    def __getitem__(self, item: str) -> Union[pd.Series, pd.DataFrame]:
        """Returns item in the 'ingredients' dictionary.

        If there are no matches, the method searches for a matching wildcard in
        and tries to return the attribute matching that wildcard.

        Args:
            item (str): name of key in the 'ingredients' dictionary.

        Returns:
            Union[pd.Series, pd.DataFrame]: item stored as a 'ingredients'
                dictionary value.

        Raises:
            KeyError: if 'item' is not found in the 'ingredients' dictionary and
                is not a wildcard.

        """
        try:
            return self.ingredients[item]
        except KeyError:
            if item in self.wildcards:
                return getattr(self, item)
            else:
                raise KeyError(' '.join(
                    [item, 'is not in', self.__class__.__.name__]))

    def __setitem__(self,
            item: str,
            value: Union[pd.Series, pd.DataFrame]) -> None:
        """Sets 'item' in the 'ingredients' dictionary to 'value'.

        Args:
            item (str): name of key in the 'ingredients' dictionary.
            value (Union[pd.Series, pd.DataFrame]): value to be paired with
                'item' in the 'ingredients' dictionary.

        """
        self.ingredients[item] = value
        return self

    def __delitem__(self, item: str) -> None:
        """Deletes item in the 'ingredients' dictionary.

        Args:
            item (str): name of key in the 'ingredients' dictionary.

        """
        try:
            del self.ingredients[item]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of the 'ingredients' dictionary.

        Returns:
            Iterable of the 'ingredients' dictionary.

        """
        return iter(self.ingredients)

    def __len__(self) -> int:
        """Returns length of the 'ingredients' dictionary.

        Returns:
            Integer of length of 'ingredients' dictionary.

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
        # Looks for 'attribute' in the 'default' Ingredient instance.
        print('test', attribute)
        print('test', self.default)
        print('test', self.ingredients[self.default])
        try:
            return getattr(self.ingredients[self.default], attribute)
        except KeyError:
            # Looks for 'attribute' as an item in 'ingredients'.
            return self.ingredients[attribute]

    def __add__(self, other: 'Ingredient') -> None:
        """Adds 'other' to the 'ingredients' dictionary.

        Args:
            other ('Ingredient'): an 'Ingredient' instance.

        """
        self.add(name = other.name, ingredient = other)
        return self

    def __iadd__(self, other: 'Ingredient') -> None:
        """Adds 'other' to the 'ingredients' dictionary.

        Args:
            other ('Ingredient'): an 'Ingredient' instance.

        """
        self.add(name = other.name, ingredient = other)
        return self

    """ Public Methods """

    def add(self, name: str, ingredient: 'Ingredient') -> None:
        """Adds 'ingredient' to the 'ingredients' dictionary.

        Args:
            ingredient ('Ingredient'): an 'Ingredient' instance.

        """
        self.ingredients[name] = ingredient
        return self


@dataclass
class Ingredient(Collection):
    """A container of a pandas data object with related metadata.

    Args:
        df (Union[pd.DataFrame, pd.Series]): a stored pandas data object.
        columns (Optional[Dict[str, str]]): keys are column names and values
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

    df: Union[pd.DataFrame, pd.Series]
    columns: Optional[Dict[str, str]] = field(default_factory = dict)
    prefixes: Optional[Dict[str, str]] = field(default_factory = dict)
    ingredients: Optional['Ingredients'] = None
    name: Optional[str] = 'ingredient'

    def _post_init__(self) -> None:
        """Sets 'state' for data exporting."""
        self.state = SimpleState(states = ['raw', 'interim', 'processed'])
        return self

    """ Required ABC Methods """

    def __contains__(self, item: str) -> bool:
        """Returns if 'item' is in 'columns' of the 'df' attribute.

        Args:
            item (str): item to check for existence.

        Returns:
            boolean value if 'item' is in 'df.columns'.

        """
        return item in self.df.columns

    def __iter__(self) -> Iterable:
        """Returns 'iterrows' method for 'df' attribute."""
        return self.df.iterrows()

    def __len__(self) -> int:
        """Returns integer length of 'df' attribute."""
        return len(self.df)

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
        # Returns appropriate lists of columns with datatype 'attribute'.
        if attribute in self.datatypes:
            return self._get_columns_by_type(datatype = attribute[:-1])
        # Combines 'floats' and 'integers' into 'numerics'.
        elif attribute in ['numerics']:
            return self.floats + self.integers
        # Returns list of 'dropped_columns' by comparing to '_start_columns'.
        elif attribute in ['dropped_columns']:
            return self._start_columns - list(self.columns.keys())
        else:
            # Looks for 'attribute' in 'state'.
            try:
                return getattr(self.state, attribute)
            except AttributeError:
                # Tries to apply method to pandas object itself.
                try:
                    return getattr(self.df, attribute)
                except:
                    raise AttributeError(' '.join(
                        [attribute, 'is not in', self.__class__.__.name__]))

    def __add__(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with the 'ingredients' dictionary.

        Args:
            other (Union[pd.DataFrame, pd.Series, 'Ingredient']): a pandas data
                object or another 'Ingredient' instance.

        """
        self.add(ingredient = other)
        return self

    def __iadd__(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with the 'ingredients' dictionary.

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
            return list(self.columns.keys())

    def _crosscheck_columns(self) -> None:
        """Harmonizes 'columns' dictionary with 'df' columns attribute."""
        for column in list(self.df.columns.values()):
            if column not in self.columns:
                self.columns[column] = self._infer_type(column = column)
        for column in list(self.columns.keys()):
            if column not in self.df.columns:
                del self.columns[column]
        return self

    def _get_columns_by_type(self, datatype: str) -> List[str]:
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'all_columns'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [key for key, value in self.columns.items() if value == datatype]

    def _get_indices(self, columns: Union[List[str], str]) -> List[bool]:
        """Gets column indices for a list of column names.

        Args:
            columns (Union[List[str], str]): name(s) of columns for which
                indices are sought.

        Returns:
            bool mask for columns matching 'columns'.

        """
        return [self.df.columns.get_loc(column) for column in listify(columns)]

    def infer_type(self, column: pd.Series) -> str:
        """Infers column datatype of a column.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        Non-standard python datatypes cannot be inferred.

        Args:


        """
        try:
            for proxy, datatype in self.types.values():
                type_columns = df.select_dtypes(
                    include = [datatype]).columns.to_list()
                self.columns.update(
                    dict.fromkeys(type_columns, self.types[datatype]))
        except AttributeError:
            pass
        return self

    def _initialize_datatypes(self) -> None:
        """Initializes datatypes for stored pandas data object."""
        if not self.columns:
            self.infer_datatypes()
        else:
            self._crosscheck_columns()
        self._start_columns = list(self.df.columns.values)
        return self

    """ Public Methods """

    def add(self,
            other: Union[pd.DataFrame, pd.Series, 'Ingredient']) -> None:
        """Combines 'other' with the 'ingredients' dictionary.

        Args:
            ingredient (Union[pd.DataFrame, pd.Series, 'Ingredient']): a pandas
                data object or another 'Ingredient' instance.

        ToDo:
            Add functionality for all cases.

        """
        return self

    def add_unique_index(self,
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
            TypeError: if 'df' is not a DataFrame (usually because it is a
                pandas Series).

        """
        try:
            self.df[column] = range(1, len(self.df.index) + 1)
            self.columns.update({column, 'integer'})
            if assign_index:
                self.df.set_index(column, inplace = True)
        except (TypeError, AttributeError):
            raise TypeError('To add an index, df must be a pandas DataFrame')
        return self

    def apply(self, func: Callable, **kwargs) -> None:
        """Applies 'func' to 'df' atttribute with **kwargs).

        Args:
            func (Callable): to be applied to the DataFrame.
            **kwargs: any arguments to be passed to 'func'.

        """
        self.df = func(self.df, **kwargs)
        return self

    @ColumnConformer
    def change_datatype(self,
            columns: [Union[List[str], str]],
            datatype: str) -> None:
        """Changes column datatypes of columns passed.

        The datatype becomes the new datatype for the columns in both the
        'columns' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            columns (list or str): column name(s) for datatypes to be changed.
            datatype (str): contains name of the datatype to convert the
                columns.

        """
        for column in listify(columns):
            self.columns[column] = datatype
        self.convert_column_datatypes()
        return self

    def convert_column_datatypes(self,
            raise_errors: Optional[bool] = False) -> None:
        """Converts column data to the datatypes in 'columns' dictionary.

        Args:
            raise_errors (Optional[bool]): whether errors should be raised when
                converting datatypes or ignored. Selecting False (the default)
                risks type mismatches between the datatypes listed in the
                'columns' dict and 'df', but it prevents the program from being
                halted if an error is encountered.

        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.columns.items():
            if not datatype in ['string']:
                self.df[column].astype(
                    dtype = self.types[datatype],
                    copy = False,
                    errors = raise_errors)
        # Attempts to downcast datatypes to simpler forms if possible.
        self.downcast()
        return self

    @ColumnConformer
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
            KeyError: if column in 'columns' is not in 'df'.

        """
        for column in self._check_columns(columns):
            if self.columns[column] in ['boolean']:
                self.df[column] = self.df[column].astype(bool)
            elif self.columns[column] in ['integer', 'float']:
                try:
                    self.df[column] = pd.to_numeric(
                        self.df[column],
                        downcast = 'integer')
                    if min(self.df[column] >= 0) and allow_unsigned:
                        self.df[column] = pd.to_numeric(
                            self.df[column],
                            downcast = 'unsigned')
                except ValueError:
                    self.df[column] = pd.to_numeric(
                        self.df[column],
                        downcast = 'float')
            elif self.columns[column] in ['categorical']:
                self.df[column] = self.df[column].astype('category')
            elif self.columns[column] in ['list']:
                self.df[column].apply(
                    listify,
                    axis = 'columns',
                    inplace = True)
            elif self.columns[column] in ['datetime']:
                self.df[column] = pd.to_datetime(self.df[column])
            elif self.columns[column] in ['timedelta']:
                self.df[column] = pd.to_timedelta(self.df[column])
            else:
                raise KeyError(' '.join([column, ' is not in df']))
        return self

    @ColumnConformer
    def drop_columns(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Drops list of columns and columns with prefixes listed.

        Args:
            df (DataFrame or Series): pandas object for columns to be dropped
            columns(list): columns to drop.

        """
        try:
            self.df.drop(columns, axis = 'columns', inplace = True)
        except TypeError:
            self.df.drop(columns, inplace = True)
        return self

    @ColumnConformer
    def get_series(self,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Creates a Series (row) with the 'columns' dict.

        Default values are added to each item in the series so that pandas does
        not automatically infer the datatype when a value is passed.

        Args:
            columns (list or str): index names for pandas Series.
            return_series (bool): whether the Series should be returned (True)
                or assigned to attribute named in 'default_df' (False).

        Returns:
            Either nothing, if 'return_series' is False or a pandas Series with
                index names matching 'columns' keys and datatypes matching
                'columns values'.

        """
        row = pd.Series(index = self._check_columns(columns = columns))
        # Fills series with default_values based on datatype.
        for column, datatype in self.columns.items():
            row[column] = self.default_values[datatype]
        return row

    def infer_datatypes(self) -> None:
        """Infers column datatypes and adds those datatypes to types.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        This methods also allows the user to choose which datatypes to look for
        by changing the 'options' dict stored in 'types'.

        Non-standard python datatypes cannot be inferred.

        Args:
            df (DataFrame): pandas object for datatypes to be inferred.

        """
        try:
            for datatype in self.types.options.values():
                type_columns = df.select_dtypes(
                    include = [datatype]).columns.to_list()
                self.columns.update(
                    dict.fromkeys(type_columns, self.types[datatype]))
        except AttributeError:
            pass
        return self




@dataclass
class DataProxies(SimpleOptions):

    def _make_dataframe_proxies(self):
        """Creates proxy mapping for dataframe attribute names."""
        self._dataframes = {
            'full': [self.x, self.y],
            'train': [self.x_train, self.y_train],
            'test': [self.x_test, self.y_test],
            'validation': [self.x_val, self.y_val]}
        self.library= {
            'full': [],
            'train': [],
            'test': [],
            'validation': []}
        for group, dataframes in self._dataframes.items():
            for i, prefix in enumerate(self.data_prefixes):
                suffix = getattr(self, '_'.join([group, 'suffix']))
                if suffix:
                    name =  '_'.join([prefix, suffix])
                else:
                    name = prefix
                self.library[group].append(name)
                setattr(self, name, self._dataframes[group][i])
        return self