"""
.. module:: ingredients
:synopsis: data container for siMpLify
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from simplify.core.base import SimpleClass
from simplify.core.decorators import choose_df, combine_lists
from simplify.core.types import DataTypes


@dataclass
class Ingredients(SimpleClass):
    """Stores pandas DataFrames and Series with related information about those
    data containers.

    Ingredients uses pandas DataFrames or Series for all data storage, but it
    utilizes faster numpy methods where possible to increase performance.
    DataFrames and Series stored in ingredients can be imported and exported
    using the 'load' and 'save' methods in a class instance.

    Ingredients adds easy-to-use methods for common feature engineering
    techniques. In addition, any user function can be applied to a DataFrame
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
        _x, _y, _x_train, _y_train, _x_test, _y_test, _x_val, _y_val
            (DataFrames, Series, or file paths): These need not be passed when
            the class is instanced. They are merely listed for users who already
            have divided datasets and still wish to use the siMpLify package.
        datatypes (dict): contains column names as keys and datatypes for values
            for columns in a DataFrames or Series. Ingredients assumes that all
            data containers within the instance are related and share a pool of
            column names and types.
        prefixes (dict): contains column prefixes as keys and datatypes for
            values for columns in a DataFrames or Series. Ingredients assumes
            that all data containers within the instance are related and share a
            pool of column names and types.

    Since this class is a subclass to SimpleClass, all of its documentation
    applies as well.

    """
    name: str = 'ingredients'
    df: object = None
    default_df: str = 'df'
    _x: object = None
    _y: object = None
    _x_train: object = None
    _y_train: object = None
    _x_test: object = None
    _y_test: object = None
    _x_val: object = None
    _y_val: object = None
    datatypes: object = None
    prefixes: object = None

    def __post_init__(self):
        super().__post_init__()
        self.publish()
        return self

    """ Dunder Methods """

    def __getattr__(self, attribute):
        # Returns appropriate DataFrame based upon 'stage' attribute.
        if attribute in ['x_train', 'y_train', 'x_test', 'y_test']:
            prefix, suffix = attribute.split('_')
            mapped_suffix = self.options[self.stage][suffix]
            try:
                return self.__dict__[''.join(['_', prefix, mapped_suffix])]
            except TypeError:
                return None
        elif attribute in ['x', 'y', 'x_val', 'y_val']:
            return self.__dict__[''.join(['_', attribute])]
        elif attribute in ['floats', 'integers', 'strings', 'lists', 'booleans',
                           'categoricals', 'datetimes', 'timedeltas']:
            return self._get_columns_by_type(datatype = attribute[:-1])
        elif attribute in ['numerics']:
            return self.floats + self.integers

    def __setattr__(self, attribute, value):
        # Sets appropriate DataFrame based upon 'stage' attribute.
        if attribute in ['x_train', 'y_train', 'x_test', 'y_test']:
            prefix, suffix = attribute.split('_')
            mapped_suffix = self.options[self.stage][suffix]
            try:
                self.__dict__[''.join(['_', prefix, mapped_suffix])] = value
            except TypeError:
                pass
        elif attribute in ['x', 'y', 'x_val', 'y_val']:
            self.__dict__[''.join(['_', attribute])] = value
        else:
            self.__dict__[attribute] = value
        return self


    """ Private Methods """

    def _check_columns(self, columns = None):
        """Returns self.datatypes if columns doesn't exist.

        Args:
            columns (list): column names.

        Returns:
            if columns is not None, returns columns, otherwise, the keys of
                the 'datatypes' attribute is returned.

        """
        return columns or list(self.datatypes.keys())

    @choose_df
    def _crosscheck_columns(self, df = None):
        """Removes any columns in datatypes dictionary, but not in df.

        Args:
            df (DataFrame or Series): pandas object with column names to
                crosscheck.

        """
        for column in list(self.datatypes.keys()):
            if column not in df.columns:
                del self.datatypes[column]
        return self

    def _get_columns_by_type(self, datatype):
        """Returns list of columns of the specified datatype.

        Args:
            datatype (str): string matching datatype in 'all_datatypes'.

        Returns:
            list of columns matching the passed 'datatype'.

        """
        return [k for k, v in self.datatypes.items() if v == datatype]

    @choose_df
    def _get_indices(self, df = None, columns = None):
        """Gets column indices for a list of column names.

        Args:
            df (DataFrame or Series): pandas object with column names to get
                indices for.
            columns (list): names of columns for which indices are sought.

        Returns:
            bool mask for columns matching 'columns'.DataState

        """
        return [df.columns.get_loc(column) for column in columns]

    @choose_df
    def _initialize_datatypes(self, df = None):
        """Initializes datatypes for columns of pandas DataFrame or Series if
        not already provided.

        Args:
            df (DataFrame or Series): for datatypes to be determined.

        """
        if not self.datatypes:
            self.infer_datatypes(df = df)
        else:
            self._crosscheck_columns(df = df)
        self._start_columns = list(df.columns.values)
        return self

    """ Public Tool Methods """

    @choose_df
    def add_unique_index(self, df = None, column = 'index_universal',
                         make_index = False):
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
            self.datatypes.update({column, int})
            if make_index:
                df.set_index(column, inplace = True)
        except TypeError:
            error = 'To add an index, df must be a pandas DataFrame.'
            TypeError(error)
        return self

    @choose_df
    def apply(self, df = None, func = None, **kwargs):
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

    @combine_lists
    @choose_df
    def auto_categorize(self, df = None, columns = None, threshold = 10):
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
        for column in self.listify(self._check_columns(columns)):
            try:
                if not column in self.booleans:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
                        self.datatypes[column] = 'categorical'
            except KeyError:
                error = ' '.join([column, 'is not in df'])
                raise KeyError(error)
        return self

    @combine_lists
    @choose_df
    def change_datatype(self, df = None, columns = None, datatype = None):
        """Changes column datatypes of columns passed or columns with the
        prefixes passed.

        The datatype becomes the new datatype for the columns in both the
        'datatypes' dict and in reality - a method is called to try to convert
        the column to the appropriate datatype.

        Args:
            df (DataFrame): pandas object for datatypes to be changed.
            columns (list): column names for datatypes to be changed.
            datatype (str): contains name of the datatype to convert the
                columns.

        """
        for column in columns:
            self.datatypes[column] = datatype
        self.convert_column_datatypes(df = df)
        return self

    @choose_df
    def convert_column_datatypes(self, df = None, raise_errors = False):
        """Attempts to convert all column data to the match the datatypes in
        'datatypes' dictionary.

        Args:
            df (DataFrame): pandas object with data to be changed to a new type.
            raise_errors (bool): whether errors should be raised when converting
                datatypes or ignored. Selecting False (the default) risks type
                mismatches between the datatypes listed in the 'datatypes' dict
                and 'df', but it prevents the program from being halted if
                an error is encountered.

        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.datatypes.items():
            if not datatype in ['string']:
                df[column].astype(dtype = self.all_datatypes[datatype],
                                  copy = False,
                                  errors = raise_errors)
        # Attempts to downcast datatypes to simpler forms if possible.
        self.downcast(df = df)
        return self

    @combine_lists
    @choose_df
    def convert_rare(self, df = None, columns = None, threshold = 0):
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
                df[column] = np.where(df['value_freq'] <= threshold,
                                      self.default_values['categorical'],
                                      df[column])
            except KeyError:
                error = column + ' is not in df'
                raise KeyError(error)
        if 'value_freq' in df.columns:
            df.drop('value_freq', axis = 'columns', inplace = True)
        return self

    @choose_df
    def create_column_list(self, df = None, columns = None, prefixes = None,
                           mask = None):
        """Dynamically creates a new column list from a list of columns, lists
        of prefixes, and/or boolean mask.

        Args:
            df (DataFrame): pandas object.
            columns (list or str): column names to be included.
            prefixes (list or str): list of prefixes for columns to be included.
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
            for prefix in self.listify(prefixes):
                temp_list = [col for col in df if col.startswith(prefix)]
                column_names.extend(temp_list)
        except TypeError:
            pass
        try:
            column_names.extend(self.listify(columns))
        except TypeError:
            pass
        return self.deduplicate(iterable = column_names)

    @combine_lists
    def create_series(self, columns = None, return_series = True):
        """Creates a Series (row) with the 'datatypes' dict.

        Args:
            columns (list or str): index names for pandas Series.
            return_series (bool): whether the Series should be returned (True)
                or assigned to attribute named in 'default_df' (False).

        Returns:
            Either nothing, if 'return_series' is False or a pandas Series with
                index names matching 'datatypes' keys and datatypes matching
                'datatypes values'.

        """
        row = pd.Series(index = self._check_columns(columns = columns))
        # Fills series with default_values based on datatype.
        for column, datatype in self.datatypes.items():
            row[column] = self.default_values[datatype]
        if return_series:
            return row
        else:
            setattr(self, self.default_df, row)
            return self

    @combine_lists
    @choose_df
    def decorrelate(self, df = None, columns = None, threshold = 0.95):
        """Drops all but one column from highly correlated groups of columns.

        The threshold is based upon the .corr() method in pandas. columns can
        include any datatype accepted by .corr(). If columns is set to None,
        all columns in the DataFrame are tested.

        Args:
            df (DataFrame): pandas object to be have highly correlated features
                removed.
            threshold (float): the level of correlation using pandas corr method
                above which a column is dropped. The default threshold is 0.95,
                consistent with a common p-value threshold used in research.

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

    @combine_lists
    @choose_df
    def downcast(self, df = None, columns = None, allow_unsigned = True):
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
            try:
                if self.datatypes[column] in ['boolean']:
                    df[column] = df[column].astype(bool)
                elif self.datatypes[column] in ['integer', 'float']:
                    try:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'integer')
                        if min(df[column] >= 0) and allow_unsigned:
                            df[column] = pd.to_numeric(df[column],
                                                       downcast = 'unsigned')
                    except ValueError:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'float')
                elif self.datatypes[column] in ['categorical']:
                    df[column] = df[column].astype('category')
                elif self.datatypes[column] in ['list']:
                    df[column].apply(self.listify,
                                     axis = 'columns',
                                     inplace = True)
                elif self.datatypes[column] in ['datetime']:
                    df[column] = pd.to_datetime(df[column])
                elif self.datatypes[column] in ['timedelta']:
                    df[column] = pd.to_timedelta(df[column])
            except KeyError:
                error = column + ' is not in df'
                raise KeyError(error)
        return self

    @combine_lists
    @choose_df
    def drop_columns(self, df = None, columns = None):
        """Drops list of columns and columns with prefixes listed.

        In addition to removing the columns, any dropped columns have their
        column names stored in the cumulative 'dropped_columns' list. If you
        wish to make use of the 'dropped_columns' attribute, you should use this
        'drop_columns' method instead of dropping the columns directly.

        Args:
            df(DataFrame or Series): pandas object for columns to be dropped
            columns(list): columns to drop.
        """
        if isinstance(df, pd.DataFrame):
            df.drop(columns, axis = 'columns', inplace = True)
        else:
            df.drop(columns, inplace = True)
        self.dropped_columns.extend(columns)
        return self

    @combine_lists
    @choose_df
    def drop_infrequent(self, df = None, columns = None, threshold = 0):
        """Drops boolean columns that rarely are True.

        This differs from the sklearn VarianceThreshold class because it is only
        concerned with rare instances of True and not False. This enables
        users to set a different variance threshold for rarely appearing
        information. threshold is defined as the percentage of total rows (and
        not the typical variance formulas used in sklearn).

        Args:
            df(DataFrame): pandas object for columns to checked for infrequent
                boolean True values.
            columns(list): columns to check.
            threshold(float): the percentage of True values in a boolean column
            that must exist for the column to be kept.
        """
        if not columns:
            columns = self.booleans
        infrequents = []
        for column in self.booleans:
            try:
                if df[column].mean() < threshold:
                    infrequents.append(column)
            except KeyError:
                error = ' '.join([column, 'is not in df'])
                raise KeyError(error)
        self.drop_columns(columns = infrequents)
        return self

    @choose_df
    def infer_datatypes(self, df = None):
        """Infers column datatypes and adds those datatypes to types.

        This method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.) instead of
        simple types.

        This methods also allows the user to choose which datatypes to look for
        by changing the 'default_values' dict stored in 'all_datatypes'.

        Non-standard python datatypes cannot be inferred.

        Args:
            df (DataFrame): pandas object for datatypes to be inferred.
        """
        for datatype in self.all_datatypes.options.values():
            type_columns = df.select_dtypes(
                include = [datatype]).columns.to_list()
            self.datatypes.update(
                dict.fromkeys(type_columns,
                              self.all_datatypes[datatype]))
        return self

    def save_dropped(self, folder = 'experiment', file_name = 'dropped_columns',
                     file_format = 'csv'):
        """Saves 'dropped_columns' into a file.

        Args:
            folder (str): file folder for file to be exported.
            file_name (str): file name without extension of file to be exported.
            file_format (str): file format name.

        """
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            self.depot.save(variable = self.dropped_columns,
                                folder = folder,
                                file_name = file_name,
                                file_format = file_format)
        elif self.verbose:
            print('No features were dropped during preprocessing.')
        return

    @combine_lists
    @choose_df
    def smart_fill(self, df = None, columns = None):
        """Fills na values in a DataFrame with defaults based upon the datatype
        listed in the 'datatypes' dictionary.

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
                        self.datatypes[column]]
                df[column].fillna(default_value, inplace = True)
            except KeyError:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    def split_xy(self, label = 'label'):
        """Splits df into 'x' and 'y' based upon the label ('y' column) passed.

        Args:
            df (DataFrame): initial pandas object to be split.
            label (str or list): name of column(s) to be stored in 'y'.'
        """
        self.x = Ingredient(
            df = df[list(df.columns.values).remove(label)],
            datatypes = self.datatypes,
            prefixes = self.prefixes)
        self.y = Ingredient(
            df = df[label],
            datatypes = {label: self.datatypes[label]})
        self.state.change('train_test')
        return self

    """ Core siMpLify Methods """

    def draft(self):
        """Sets defaults for Ingredients when class is instanced."""
        # Creates object for all available datatypes.
        self.all_datatypes = DataTypes()
        print(self.all_datatypes)
        # Creates 'datatypes' and 'prefixes' dicts if they don't exist.
        self.datatypes = self.datatypes or {}
        self.prefixes = self.prefixes or {}
        # Creates data state machine instance.
        self.state = DataState()
        self.options = {
            'unsplit': {'train': '', 'test': None},
            'train_test': {'train': '_train', 'test': '_test'},
            'train_val': {'train': '_train', 'test': '_val'},
            'full': {'train': '', 'test': ''}}
        print(self.all_datatypes)
        return self

    def publish(self):
        """Finalizes Ingredients class instance."""
        # If datatypes passed, checks to see if columns are in 'df'. Otherwise,
        # datatypes are inferred.
        self._initialize_datatypes()
        return self

    """ Properties """

    @property
    def dropped_columns(self):
        return self._start_columns - self.x_train.columns.values


@dataclass
class DataState(SimpleClass):

    state: str = 'train'

    def __post_init__(self):
        self.draft()
        return self

    def __repr__(self):
        """Returns string name of 'state'."""
        return self.__str__()

    def __str__(self):
        """Returns string name of 'state'."""
        return self.state

    def change(self, new_state):
        """Changes 'state' to 'new_state'.

        Args:
            new_state(str): name of new state matching a string in 'states'.

        Raises:
            TypeError: if new_state is not in 'states'.

        """
        if new_state in self.states:
            self.state = new_state
        else:
            error = new_state + ' is not a recognized data state'
            raise TypeError(error)

    def draft(self):
        # Sets possible states
        self.states = ['unsplit', 'train_test', 'train_val', 'full']
        return self

    def publish(self):
        return self.state