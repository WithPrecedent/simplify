"""
.. module:: ingredients
:synopsis: data container for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from functools import wraps
from inspect import signature, Signature
import os
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from numpy import datetime64
import pandas as pd
from pandas.api.types import CategoricalDtype

from simplify.core.options import SimpleData
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify



INGREDIENTS = {
    'default_df': 'df',
    'options': {
        'unsplit': {
            'train': 'full_suffix',
            'test': None},
        'train_test': {
            'train': 'train_suffix',
            'test': 'test_suffix'},
        'train_val': {
            'train': 'train_suffix',
            'test': 'test_suffix'},
        'full': {
            'train': 'full_suffix',
            'test': 'full_suffix'}},
    'data_prefixes': ['x', 'y'],
    'train_suffix': 'train',
    'test_suffix': 'test',
    'validation_suffix': 'val',
    'full_suffix': ''}

""" Ingredients Decorators """

def backup_df(return_df: Optional[bool] = False) -> Callable:
    """Substitutes the default DataFrame or Series if 'df' is not passed.

    Args:
        return_df (Optional[bool]): whether to return_df when 'df' was not
            passed. Defaults to False.

    Returns:
        If 'df' is passed, a pandads DataFrame or Series is returned.
        If 'df' is not passed, the local attribute in the class named the value
            in 'default_df' is changed and 'self' is returned.

    """
    def shell_backup_df(method: Callable, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            call_signature = signature(method)
            parameters = dict(call_signature.parameters)
            arguments = dict(call_signature.bind(*args, **kwargs).arguments)
            unpassed = list(parameters.keys() - arguments.keys())
            if 'df' in unpassed:
                arguments.update({'df': getattr(self, self.default_df)})
                method.__signature__ = Signature(arguments)
                if return_df:
                    df = method(self, **arguments)
                    return df
                else:
                    setattr(self,
                        getattr(self, self.default_df),
                        method(self, **arguments))
                    return self
            else:
                df = method(self, **arguments)
                return df
        return wrapper
    return shell_backup_df


""" Ingredients Class """

@dataclass
class Ingredients(SimpleData):
    """Stores pandas DataFrames and Series with related information about those
    data containers.

    Ingredients uses pandas DataFrames or Series for all data storage, but it
    utilizes faster numpy methods where possible to increase performance.
    DataFrames and Series stored in ingredients can be imported and exported
    using the 'load' and 'save' methods in a class instance.

    Ingredients adds easy-to-use methods for common feature engineering
    steps. In addition, any user function can be applied to a DataFrame
    or Series contained in Ingredients by using the 'apply' method (mirroring
    the functionality of the pandas method).

    Args:
        name (str): designates the name of the class which should match the
            section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes.
        df (DataFrame, Series, or str): either a pandas data object or a string
            containing the complete path where a supported file type with data
            is located. This argument should be passed if the user has a
            pre-existing dataset and is not creating a new dataset with Farmer.
        x, y, x_train, y_train, x_test, y_test, x_val, y_val
            (DataFrames, Series, or str): These need not be passed when the
            class is instanced. They are merely listed for users who already
            have divided datasets and still wish to use the siMpLify package.
        columns (dict): contains column names as keys and datatypes for values
            for columns in a DataFrames or Series. Ingredients assumes that all
            data containers within the instance are related and share a pool of
            column names and types.
        prefixes (dict): contains column prefixes as keys and datatypes for
            values for columns in a DataFrames or Series. Ingredients assumes
            that all data containers within the instance are related and share a
            pool of column names and types.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced.

    """
    idea: 'Idea'
    inventory: 'Inventory'
    name: Optional[str] = 'ingredients'
    df: Optional[pd.DataFrame] = None
    default_df: Optional[str] = None
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None
    x_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.DataFrame] = None
    x_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.DataFrame] = None
    x_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.DataFrame] = None
    columns: Optional[Union[Dict[str, str]]] = None
    prefixes: Optional[Union[Dict[str, str]]] = None
    auto_publish: Optional[bool] = True
    export_folder: str = 'data'

    def __post_init__(self) -> None:
        """Sets default values and calls appropriate creation methods."""
        self.draft()
        if self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __getattr__(self, attribute: str) -> List[str]:
        # Returns appropriate lists of columns.
        if attribute in ['floats', 'integers', 'strings', 'lists', 'booleans',
                         'categoricals', 'datetimes', 'timedeltas']:
            try:
                return self.__dict__[attribute]
            except KeyError:
                return self._get_columns_by_type(datatype = attribute[:-1])
        elif attribute in ['numerics']:
            return self.floats + self.integers
        elif attribute in ['dropped_columns']:
            return self._start_columns - self.x_train.columns.values
        else:
            raise AttributeError(' '.join(
                [self.name, 'does not contain', attribute]))

    def __setattr__(self, attribute: str, value: Any) -> None:
        # Sets appropriate DataFrame based upon proxy mapping.
        try:
            if attribute in self.library:
                self.library[attribute] = value
        except (KeyError, AttributeError):
            self.__dict__[attribute] = value
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
        return columns or list(self.columns.keys())


    @backup_df


    def _get_columns_by_type(self, datatype: str) -> List[str]:
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'all_columns'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [k for k, v in self.columns.items() if v == datatype]

    @backup_df


    @backup_df


    def _publish_data(self) -> None:
        """Completes an Ingredients instance.

        This method checks all attributes listed in 'dataframes' and converts
        them, when possible, to pandas data containers.

        If a 'dataframe' is a pandas data container or is None, no action is
            taken.
        If a 'dataframe' is a file path, the file is loaded into a DataFrame and
            assigned to 'df'.
        If a 'dataframe' is a file folder, a glob in 'inventory' is created.
        If a 'dataframe' is a numpy array, it is converted to a pandas
            DataFrame.

        Raises:
            TypeError: if 'dataframe' is neither a file path, file folder, None,
                DataFrame, Series, or numpy array.

        """
        for stage, dataframes in self.library.items():
            for dataframe in dataframes:
                if (getattr(self, dataframe) is None
                        or isinstance(getattr(self, dataframe), pd.Series)
                        or isinstance(getattr(self, dataframe), pd.DataFrame)):
                    pass
                elif isinstance(dataframe, np.ndarray):
                    setattr(
                        self,
                        getattr(self, dataframe),
                         pd.DataFrame(data = getattr(self, dataframe)))
                else:
                    try:
                        setattr(
                            self,
                            getattr(self, dataframe),
                            self.inventory.load(
                                folder = self.inventory.data,
                                file_name = getattr(self, dataframe)))
                    except FileNotFoundError:
                        try:
                            self.inventory.make_glob(folder = dataframe)
                        except TypeError:
                            raise TypeError(' '.join(
                                ['df must be a file path, file folder,',
                                 'DataFrame, Series, None, or numpy array']))
        return self

    """ Public Tool Methods """

    @backup_df


    @backup_df


    # @make_columns_parameter



    def save_dropped(self,
            folder: Optional[str] = 'experiment',
            file_name: Optional[str] = 'dropped_columns',
            file_format: Optional[str] = 'csv') -> None:
        """Saves 'dropped_columns' into a file.

        Args:
            folder (str): file folder for file to be exported.
            file_name (str): file name without extension of file to be exported.
            file_format (str): file format name.

        """
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            self.inventory.save(
                variable = self.dropped_columns,
                folder = folder,
                file_name = file_name,
                file_format = file_format)
        elif self.verbose:
            print('No features were dropped during preprocessing.')
        return

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets defaults for Ingredients when class is instanced."""
        # Creates object for all available datatypes.
        # self.all_datatypes = DataTypes()
        # Creates 'datatypes' and 'prefixes' dicts if they don't exist.
        self.columns = self.columns or {}
        self.prefixes = self.prefixes or {}
        # Creates attribute names for user proxies to DataFrames.
        self._make_dataframe_proxies()
        return self

    def publish(self) -> None:
        # Creates data state machine instance.
        self.state = DataState()
        # Converts dataframes to appropriate forms.
        self._publish_data()
        # If 'columns' passed, checks to see if columns are in 'df'. Otherwise,
        # datatypes are inferred.
        self._initialize_datatypes()
        return self
