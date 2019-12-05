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
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from simplify.core.defaults import Defaults
from simplify.core.typesetter import SimpleFile
from simplify.core.typesetter import SimpleOptions
from simplify.core.typesetter import SimpleState
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


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

# def make_columns_parameter(method: Callable, *args, **kwargs) -> Callable:
#     """Decorator which creates a complete column list from passed arguments.

#     If 'prefixes', 'suffixes', or 'mask' are passed to the wrapped method, they
#     are combined with any passed 'columns' to form a list of 'columns' that are
#     ultimately passed to the wrapped method.

#     Args:
#         method (method): wrapped method.

#     Returns:
#         Callable:  with 'columns' parameter that combines items from 'columns',
#             'prefixes', 'suffixes', and 'mask' parameters into a single list
#             of column names using the 'create_column_list' method.

#     """
#     call_signature = signature(method)
#     @wraps(method)
#     def wrapper(self, *args, **kwargs):
#         new_arguments = {}
#         parameters = dict(call_signature.parameters)
#         arguments = dict(call_signature.bind(*args, **kwargs).arguments)
#         unpassed = list(parameters.keys() - arguments.keys())
#         if 'columns' in unpassed:
#             columns = []
#         else:
#             columns = listify(arguments['columns'])
#         try:
#             columns.extend(
#                 self.create_column_list(prefixes = arguments['prefixes']))
#             del arguments['prefixes']
#         except KeyError:
#             pass
#         try:
#             columns.extend(
#                 self.create_column_list(suffixes = arguments['suffixes']))
#             del arguments['suffixes']
#         except KeyError:
#             pass
#         try:
#             columns.extend(
#                 self.create_column_list(mask = arguments['mask']))
#             del arguments['mask']
#         except KeyError:
#             pass
#         if not columns:
#             columns = list(self.columns.keys())
#         arguments['columns'] = deduplicate(columns)
#         # method.__signature__ = Signature(arguments)
#         return method(self, **arguments)
#     return wrapper


""" Ingredients Class """

@dataclass
class Ingredients(SimpleOptions):
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
    library: 'Library'
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
        self.defaults = Defaults()
        self = self.defaults.apply(instance = self)
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
            if attribute in self.options:
                self.options[attribute] = value
        except (KeyError, AttributeError):
            self.__dict__[attribute] = value
        return self

    """ Private Methods """

    def _check_columns(self, columns: Optional[List[str]] = None) -> bool:
        """Returns 'columns' keys if columns doesn't exist.

        Args:
            columns (Optional[List[str]]): column names.

        Returns:
            if columns is not None, returns columns, otherwise, the keys of
                the 'columns' attribute are returned.

        """
        return columns or list(self.columns.keys())

    def _create_dataframe_proxies(self):
        """Creates proxy mapping for dataframe attribute names."""
        self._dataframes = {
            'full': [self.x, self.y],
            'train': [self.x_train, self.y_train],
            'test': [self.x_test, self.y_test],
            'validation': [self.x_val, self.y_val]}
        self.options= {
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
                self.options[group].append(name)
                setattr(self, name, self._dataframes[group][i])
        return self

    @backup_df
    def _crosscheck_columns(self, df: Optional[pd.DataFrame] = None) -> None:
        """Removes any columns in columns dictionary, but not in df.

        Args:
            df (DataFrame or Series): pandas object with column names to
                crosscheck.

        """
        for column in list(self.columns.keys()):
            try:
                del self.columns[column]
            except KeyError:
                pass
        return self

    def _get_columns_by_type(self, datatype: str) -> List[str]:
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'all_columns'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [k for k, v in self.columns.items() if v == datatype]

    @backup_df
    def _get_indices(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None) -> List[bool]:
        """Gets column indices for a list of column names.

        Args:
            df (DataFrame or Series): pandas object with column names to get
                indices for.
            columns (list or str): name(s) of columns for which indices are
                sought.

        Returns:
            bool mask for columns matching 'columns'.DataState

        """
        return [df.columns.get_loc(column) for column in listify(columns)]

    @backup_df
    def _initialize_datatypes(self, df: Optional[pd.DataFrame] = None) -> None:
        """Initializes datatypes for columns of pandas DataFrame or Series if
        not already provided.

        Args:
            df (DataFrame or Series): for datatypes to be determined.

        """
        if not self.columns:
            self.infer_datatypes(df = df)
        else:
            self._crosscheck_columns(df = df)
        self._start_columns = list(df.columns.values)
        return self

    def _publish_data(self) -> None:
        """Completes an Ingredients instance.

        This method checks all attributes listed in 'dataframes' and converts
        them, when possible, to pandas data containers.

        If a 'dataframe' is a pandas data container or is None, no action is
            taken.
        If a 'dataframe' is a file path, the file is loaded into a DataFrame and
            assigned to 'df'.
        If a 'dataframe' is a file folder, a glob in 'library' is created.
        If a 'dataframe' is a numpy array, it is converted to a pandas
            DataFrame.

        Raises:
            TypeError: if 'dataframe' is neither a file path, file folder, None,
                DataFrame, Series, or numpy array.

        """
        for stage, dataframes in self.options.items():
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
                            self.library.load(
                                folder = self.library.data,
                                file_name = getattr(self, dataframe)))
                    except FileNotFoundError:
                        try:
                            self.library.create_glob(folder = dataframe)
                        except TypeError:
                            raise TypeError(' '.join(
                                ['df must be a file path, file folder,',
                                 'DataFrame, Series, None, or numpy array']))
        return self

    """ Public Tool Methods """

    @backup_df
    def add_unique_index(self,
            df: Optional[pd.DataFrame] = None,
            column: Optional[str] = 'index_universal',
            make_index: Optional[bool] = False) -> None:
        """Creates a unique integer index for each row.

        Args:
            df (DataFrame): pandas object for index column to be added.
            column (str): contains the column name for the index.
            make_index (bool): boolean value indicating whether the index column
                should be made the actual index of the DataFrame.

        Raises:
            TypeError: if 'df' is not a DataFrame (usually because a Series is
                passed).

        """
        try:
            df[column] = range(1, len(df.index) + 1)
            self.columns.update({column, int})
            if make_index:
                df.set_index(column, inplace = True)
        except TypeError:
            error = 'To add an index, df must be a pandas DataFrame.'
            TypeError(error)
        return self

    @backup_df
    def apply(self,
            df: Optional[pd.DataFrame] = None,
            func: Optional[object] = None,
            **kwargs) -> None:
        """Allows users to pass a function to Ingredients instance which will
        be applied to the passed DataFrame (or uses default_df if none is
        passed).

        Args:
            df (DataFrame or Series): pandas object for 'func' to be applied.
            func (function): to be applied to the DataFrame.
            **kwargs: any arguments to be passed to 'func'.
        """
        df = func(df, **kwargs)
        return self

    # @make_columns_parameter
    @backup_df
    def auto_categorize(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            threshold: int = 10) -> None:
        """Automatically assesses each column to determine if it has less than
        threshold unique values and is not boolean. If so, that column is
        converted to category type.

        Args:
            df (DataFrame): pandas object for columns to be evaluated for
                'categorical' type.
            columns (list or str): column names to be checked.
            threshold (int): number of unique values necessary to form a
                category. If there are less unique values than the threshold,
                the column is converted to a category type. Otherwise, it will
                remain its current datatype.

        Raises:
            KeyError: if a column in 'columns' is not in 'df'.

        """
        for column in self._check_columns(columns):
            try:
                if not column in self.booleans:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
                        self.columns[column] = 'categorical'
            except KeyError:
                error = ' '.join([column, 'is not in df'])
                raise KeyError(error)
        return self


    # @make_columns_parameter
    # @backup_df
    def change_datatype(self,
            columns: [Union[List[str], str]],
            datatype: str,
            df: Optional[pd.DataFrame] = None) -> None:
        """Changes column datatypes of columns passed or columns with the
        prefixes passed.

        The datatype becomes the new datatype for the columns in both the
        'columns' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            df (DataFrame): pandas object for datatypes to be changed.
            columns (list or str): column name(s) for datatypes to be changed.
            datatype (str): contains name of the datatype to convert the
                columns.

        """
        for column in listify(columns):
            self.columns[column] = datatype
        self.convert_column_datatypes(df = df)
        return self

    @backup_df
    def convert_column_datatypes(self,
            df: Optional[pd.DataFrame] = None,
            raise_errors: Optional[bool] = False) -> None:
        """Attempts to convert all column data to the match the datatypes in
        'columns' dictionary.

        Args:
            df (DataFrame): pandas object with data to be changed to a new type.
            raise_errors (bool): whether errors should be raised when converting
                datatypes or ignored. Selecting False (the default) risks type
                mismatches between the datatypes listed in the 'columns' dict
                and 'df', but it prevents the program from being halted if
                an error is encountered.

        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.columns.items():
            if not datatype in ['string']:
                df[column].astype(
                    dtype = self.all_datatypes[datatype],
                    copy = False,
                    errors = raise_errors)
        # Attempts to downcast datatypes to simpler forms if possible.
        self.downcast(df = df)
        return self

    # @make_columns_parameter
    @backup_df
    def convert_rare(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            threshold: Optional[float] = 0) -> None:
        """Converts categories rarely appearing within categorical columns
        to empty string if they appear below the passed threshold.

        The threshold is defined as the percentage of total rows.

        Args:
            df (DataFrame): pandas object with 'categorical' columns.
            columns (list): column names for datatypes to be checked. If it is
                not passed, all 'categorical' columns will be checked.
            threshold (float): indicates the percentage of values in rows
                below which a default value is substituted.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.

        """
        if not columns:
            columns = self.categoricals
        for column in columns:
            try:
                df['value_freq'] = df[column].value_counts() / len(df[column])
                df[column] = np.where(
                    df['value_freq'] <= threshold,
                    self.default_values['categorical'],
                    df[column])
            except KeyError:
                error = column + ' is not in df'
                raise KeyError(error)
        if 'value_freq' in df.columns:
            df.drop('value_freq', axis = 'columns', inplace = True)
        return self

    @backup_df
    def create_column_list(self,
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
            for prefix in listify(prefixes, use_null = True):
                temp_list = [col for col in df if col.startswith(prefix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            temp_list = []
            for prefix in listify(suffixes, use_null = True):
                temp_list = [col for col in df if col.endswith(suffix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            column_names.extend(listify(columns, use_null = True))
        except TypeError:
            pass
        return deduplicate(iterable = column_names)

    # @make_columns_parameter
    def create_series(self,
            columns: Optional[Union[List[str], str]] = None,
            return_series: Optional[bool] = True) -> None:
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
        if return_series:
            return row
        else:
            setattr(self, self.default_df, row)
            return self

    # @make_columns_parameter
    @backup_df
    def decorrelate(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            threshold: Optional[float] = 0.95) -> None:
        """Drops all but one column from highly correlated groups of columns.

        The threshold is based upon the .corr() method in pandas. 'columns' can
        include any datatype accepted by .corr(). If 'columns' is None, all
        columns in the DataFrame are tested.

        Args:
            df (DataFrame): pandas object to be have highly correlated features
                removed.
            threshold (float): the level of correlation using pandas corr method
                above which a column is dropped. The default threshold is 0.95,
                consistent with a common p-value threshold used in social
                science research.

        """
        try:
            corr_matrix = df[columns].corr().abs()
        except TypeError:
            corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
        self.drop_columns(columns = corrs)
        return self

    # @make_columns_parameter
    @backup_df
    def downcast(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            allow_unsigned: Optional[bool] = True) -> None:
        """Decreases memory usage by downcasting datatypes.

        For numerical datatypes, the method attempts to cast the data to
        unsigned integers if possible when 'allow_unsigned' is True. If more
        data might be added later which, in the same column, has values less
        than zero, 'allow_unsigned' should be set to False.

        Args:
            df (DataFrame): pandas object for columns to be downcasted.
            columns (list): columns to downcast.
            allow_unsigned (bool): whether to allow downcasting to unsigned int.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.

        """
        for column in self._check_columns(columns):
            if self.columns[column] in ['boolean']:
                df[column] = df[column].astype(bool)
            elif self.columns[column] in ['integer', 'float']:
                try:
                    df[column] = pd.to_numeric(
                        df[column],
                        downcast = 'integer')
                    if min(df[column] >= 0) and allow_unsigned:
                        df[column] = pd.to_numeric(
                            df[column],
                            downcast = 'unsigned')
                except ValueError:
                    df[column] = pd.to_numeric(
                        df[column],
                        downcast = 'float')
            elif self.columns[column] in ['categorical']:
                df[column] = df[column].astype('category')
            elif self.columns[column] in ['list']:
                df[column].apply(
                    listify,
                    axis = 'columns',
                    inplace = True)
            elif self.columns[column] in ['datetime']:
                df[column] = pd.to_datetime(df[column])
            elif self.columns[column] in ['timedelta']:
                df[column] = pd.to_timedelta(df[column])
            else:
                error = column + ' is not in df'
                raise KeyError(error)
        return self

    # @make_columns_parameter
    @backup_df
    def drop_columns(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Drops list of columns and columns with prefixes listed.

        Args:
            df (DataFrame or Series): pandas object for columns to be dropped
            columns(list): columns to drop.

        """
        try:
            df.drop(columns, axis = 'columns', inplace = True)
        except TypeError:
            df.drop(columns, inplace = True)
        return self

    # @make_columns_parameter
    @backup_df
    def drop_infrequent(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None,
            threshold: Optional[float] = 0) -> None:
        """Drops boolean columns that rarely are True.

        This differs from the sklearn VarianceThreshold class because it is only
        concerned with rare instances of True and not False. This enables
        users to set a different variance threshold for rarely appearing
        information. 'threshold' is defined as the percentage of total rows (and
        not the typical variance formulas used in sklearn).

        Args:
            df (DataFrame): pandas object for columns to checked for infrequent
                boolean True values.
            columns (list or str): columns to check.
            threshold (float): the percentage of True values in a boolean column
                that must exist for the column to be kept.
        """
        if columns is None:
            columns = self.booleans
        infrequents = []
        for column in listify(columns):
            try:
                if df[column].mean() < threshold:
                    infrequents.append(column)
            except KeyError:
                error = ' '.join([column, 'is not in df'])
                raise KeyError(error)
        self.drop_columns(columns = infrequents)
        return self

    @backup_df
    def infer_datatypes(self,
            df: Optional[pd.DataFrame] = None) -> None:
        """Infers column datatypes and adds those datatypes to types.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        This methods also allows the user to choose which datatypes to look for
        by changing the 'options' dict stored in 'all_datatypes'.

        Non-standard python datatypes cannot be inferred.

        Args:
            df (DataFrame): pandas object for datatypes to be inferred.

        """
        try:
            for datatype in self.all_datatypes.options.values():
                type_columns = df.select_dtypes(
                    include = [datatype]).columns.to_list()
                self.columns.update(
                    dict.fromkeys(type_columns, self.all_datatypes[datatype]))
        except AttributeError:
            pass
        return self

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
            self.library.save(
                variable = self.dropped_columns,
                folder = folder,
                file_name = file_name,
                file_format = file_format)
        elif self.verbose:
            print('No features were dropped during preprocessing.')
        return

    # @make_columns_parameter
    @backup_df
    def smart_fill(self,
            df: Optional[pd.DataFrame] = None,
            columns: Optional[Union[List[str], str]] = None) -> None:
        """Fills na values in a DataFrame with defaults based upon the datatype
        listed in 'all_datatypes'.

        Args:
            df (DataFrame): pandas object for values to be filled
            columns (list): list of columns to fill missing values in. If no
                columns are passed, all columns are filled.

        Raises:
            KeyError: if column in 'columns' is not in 'df'.

        """
        for column in self._check_columns(columns):
            try:
                default_value = self.all_datatypes.default_values[
                        self.columns[column]]
                df[column].fillna(default_value, inplace = True)
            except KeyError:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    def split_xy(self, label: Optional[str] = 'label') -> None:
        """Splits df into 'x' and 'y' based upon the label ('y' column) passed.

        Args:
            df (DataFrame): initial pandas object to be split.
            label (str or list): name of column(s) to be stored in 'y'.'

        """
        self.x = df[list(df.columns.values).remove(label)]
        self.y = df[label],
        self.label_datatype = self.columns[label]
        self._crosscheck_columns()
        self.state.change('train_test')
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets defaults for Ingredients when class is instanced."""
        # Creates object for all available datatypes.
        # self.all_datatypes = DataTypes()
        # Creates 'datatypes' and 'prefixes' dicts if they don't exist.
        self.columns = self.columns or {}
        self.prefixes = self.prefixes or {}
        # Creates attribute names for user proxies to DataFrames.
        self._create_dataframe_proxies()
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


@dataclass
class DataFile(SimpleFile):
    """Manages data importing and exporting for siMpLify."""
    folder_path: str
    file_name: str
    file_format: 'FileFormat'

    def __post_init__(self):
        return self


@dataclass
class DataState(SimpleState):
    """State machine for siMpLify project workflow.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        state (Optional[str]): initial state. Defaults to 'unsplit'.

    """
    name: Optional[str] = 'data_state_machine'
    state: Optional[str] = 'unsplit'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        # Sets possible states
        self._options = SimpleOptions(options = ['unsplit', 'train_test', 'train_val', 'full']
        return self
