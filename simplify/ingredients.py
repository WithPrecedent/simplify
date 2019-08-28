
import csv
from datetime import timedelta
from dataclasses import dataclass
from more_itertools import unique_everseen
import os

import numpy as np
import pandas as pd
from numpy import datetime64
from pandas.api.types import CategoricalDtype

from .decorators import check_df
from .summary import Summary
from .tools import listify


@dataclass
class Ingredients(object):
    """Imports, stores, and exports pandas DataFrames and Series, as well as
    related information about those data containers.

    Ingredients uses pandas DataFrames or Series for all data storage, but it
    utilizes faster numpy methods where possible to increase performance.
    Ingredients stores the data itself as well as related variables containing
    information about the data.

    DataFrames stored in ingredients can be imported and exported using the
    load and save methods from the Inventory class.

    Ingredients adds easy-to-use methods for common feature engineering
    techniques. In addition, any user function can be applied to a DataFrame
    or Series contained in Ingredients by using the apply method.

    Parameters:

        menu: an instance of Menu.
        inventory: an instance of Inventory.
        df: a pandas DataFrame or Series. This argument should be passed if
            the user has a pre-existing dataset.
        auto_prepare: a boolean variable indicating whether prepare method
            should be called when the class is instanced.
        default_df: a string listing the current default DataFrame or Series
            attribute that will be used when a specific DataFrame is not passed
            to a method. The default value is initially set to 'df'. The
            imported decorator check_df will look to the default_df to pick
            the appropriate DataFrame in such situations.
        x, y, x_train, y_train, x_test, y_test, x_val, y_val: DataFrames or
            Series. These DataFrames and Series need not be passed when the
            class is instanced. They are merely listed for users who already
            have divided datasets and still wish to use the siMpLify package.
        datatypes: dictionary containing column names and datatypes for
            DataFrames or Series.
        prefixes: dictionary containing list of prefixes for columns and
            corresponding datatypes for default DataFrame.
    """
    menu : object
    inventory : object
    df : object = None
    auto_prepare : bool = True
    auto_load : bool = False
    default_df : str = 'df'
    x : object = None
    y : object = None
    x_train : object = None
    y_train : object = None
    x_test : object = None
    y_test : object = None
    x_val : object = None
    y_val : object = None
    datatypes : object = None
    prefixes : object = None

    def __post_init__(self):
        """Localizes menu settings, sets class instance defaults, and prepares
        data and datatype dict if auto_prepare is True."""
        self.menu.inject(instance = self, sections = ['general'])
        # Sets default options for Ingredients.
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    def __contains__(self, item):
        """Checks if item is in dataframes dict; returns boolean."""
        if item in self.dataframes:
            return True
        else:
            return False

    def __delitem__(self, item):
        """Deletes item if in dataframes dict or, if an instance attribute, it
        is assigned a value of None."""
        if item in self.dataframes:
            del self.dataframes[item]
        elif hasattr(self, item):
            setattr(self, item, None)
        else:
            error = item + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getattr__(self, attr):
        """Returns values from column datatypes, column datatype dictionary,
        and section prefixes dictionary.
        """
        if attr in ['booleans', 'floats', 'integers', 'strings',
                    'categoricals', 'lists', 'datetimes', 'timedeltas']:
            return self._get_columns_by_type(self._datatype_names[attr[:-1]])
        elif (attr in ['scalers', 'encoders', 'mixers']
              and attr not in self.__dict__):
            return getattr(self, '_get_default_' + attr)()
        elif attr in ['columns']:
            if not self.datatypes:
                self.datatypes = {}
            return self.datatypes
        elif attr in ['sections']:
            if not self.prefixes:
                self.prefixes = {}
            return self.prefixes
        elif attr in self.__dict__:
            return self.__dict__[attr]
        elif attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        else:
            return None

    def __getitem__(self, item):
        """Returns item if item is in self.dataframes or is an atttribute."""
        if item in self.dataframes:
            return self.dataframes[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            error = item + ' is not in ' + self.__class__.__name__
            raise KeyError(error)

    def __iter__(self):
        """Returns iterable from default dataframe rows."""
        return getattr(self, self.default_df).iterrows()

    def __repr__(self):
        """Returns the name of the Ingredients instance in lowercase."""
        return self.__str__()

    def __setattr__(self, attr, value):
        """Sets values in column datatypes, column datatype dictionary, and
        section prefixes dictionary.
        """
        if attr in ['booleans', 'floats', 'integers', 'strings',
                    'categoricals', 'lists', 'datetimes', 'timedeltas']:
            self.__dict__['datatypes'].update(
                    dict.fromkeys(listify(self._datatype_names[attr[:-1]]),
                                  value))
            return self
        elif attr in ['columns']:
            self.__dict__['datatypes'].update({attr: value})
            return self
        elif attr in ['sections']:
            if not self.prefixes:
                self.__dict__['prefixes'] = {attr : value}
            else:
                self.__dict__['prefixes'].update({attr : value})
            return self
        else:
            self.__dict__[attr] = value
            return self

    def __setitem__(self, item, value):
        """Adds item and value to options dictionary."""
        if isinstance(item, str):
            if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                self.dataframes.update({item : value})
            else:
                error = value + ' must be a pandas Series or DataFrame'
                raise TypeError(error)
        else:
            error = value + ' must be a string type'
            raise TypeError(error)
        return self

    def __str__(self):
        """Returns the default dataframe."""
        return getattr(self, self.default_df)

    @property
    def full(self):
        """Returns the full dataset divided into x and y twice."""
        return (self.dataframes['x'], self.dataframes['y'],
                self.dataframes['x'], self.dataframes['y'])

    @property
    def test(self):
        """Returns the test data."""
        return self.dataframes['x_test'], self.dataframes['y_test']


    @property
    def train(self):
        """Returns the training data."""
        return self.dataframes['x_train'], self.dataframes['y_train']

    @property
    def train_test(self):
        """Returns the training and testing data."""
        return (self.dataframes['x_train'], self.dataframes['y_train'],
                self.dataframes['x_test'], self.dataframes['y_test'])

    @property
    def train_test_val(self):
        """Returns the training, test, and validation data."""
        return (self.dataframes['x_train'], self.dataframes['y_train'],
                self.dataframes['x_test'], self.dataframes['y_test'],
                self.dataframes['x_val'], self.dataframes['y_val'])

    @property
    def train_val(self):
        """Returns the training and validation data."""
        return (self.dataframes['x_train'], self.dataframes['y_train'],
                self.dataframes['x_val'], self.dataframes['y_val'])

    @property
    def val(self):
        """Returns the validation data."""
        return self.dataframes['x_val'], self.dataframes['y_val']

    @property
    def xy(self):
        """Returns the full dataset divided into x and y."""
        return self.dataframes['x'], self.dataframes['y']

    def _check_columns(self, columns = None):
        """Returns self.datatypes if columns doesn't exist.

        Parameters:
            columns: list of column names."""
        return columns or list(self.datatypes.keys())

    @check_df
    def _crosscheck_columns(self, df = None):
        """Removes any columns in datatypes dictionary, but not in df."""
        for column, datatype in self.datatypes.items():
            if column not in df.columns:
                del self.datatypes[column]
        return self

    def _get_columns_by_type(self, datatype):
        """Returns list of columns of the specified datatype.

        Parameters:
            datatype: string matching datatype in self.extensions.
        """
        return (
            [key for key, value in self.datatypes.items() if value == datatype])

    def _get_default_encoders(self):
        return self.categoricals

    def _get_default_mixers(self):
        return []

    def _get_default_scalers(self):
        return self.integers + self.floats

    def _initialize_datatypes(self, df = None):
        """Initializes datatypes for columns of pandas DataFrame or Series if
        not already provided.
        """
        check_order = [df, getattr(self, self.default_df), self.x,
                       self.x_train]
        for _data in check_order:
            if isinstance(_data, pd.DataFrame) or isinstance(_data, pd.Series):
                if not self.datatypes:
                    self.infer_datatypes(df = _data)
                else:
                    self._crosscheck_columns(df = _data)
                break
        return self

    def _remap_dataframes(self, data_to_use = None):
        """Remaps DataFrames returned by various properties of Ingredients so
        that methods and classes of siMpLify can use the same labels for
        analyzing the Ingredients DataFrames.

        Parameters:
            data_to_use: a string corresponding to a class property indicating
                which set of data is to be returned when the corresponding
                property is called.
        """
        if not data_to_use:
            data_to_use = 'train_test'
        # Sets values in self.dataframes which contains the mapping for class
        # attributes and DataFrames as determined in __getattr__.
        if data_to_use == 'train_test':
            self.dataframes = self.default_dataframes.copy()
        elif data_to_use == 'train_val':
            self.dataframes['x_test'] = self.x_val
            self.dataframes['y_test'] = self.y_val
        elif data_to_use == 'full':
            self.dataframes['x_train'] = self.x
            self.dataframes['y_train'] = self.y
            self.dataframes['x_test'] = self.x
            self.dataframes['y_test'] = self.y
        elif data_to_use == 'train':
            self.dataframes['x'] = self.x_train
            self.dataframes['y'] = self.y_train
        elif data_to_use == 'test':
            self.dataframes['x'] = self.x_test
            self.dataframes['y'] = self.y_test
        elif data_to_use == 'val':
            self.dataframes['x'] = self.x_val
            self.dataframes['y'] = self.y_val
        return self

    def _set_defaults(self):
        """Sets defaults for Ingredients when class is instanced."""
        # Sets default values for missing data based upon datatype of column.
        self.default_values = {'boolean' : False,
                               'float' : 0.0,
                               'integer' : 0,
                               'string' : '',
                               'categorical' : '',
                               'list' : [],
                               'datetime' : 1/1/1900,
                               'timedelta' : 0}
        # Sets string names of various datatypes available.
        self.datatype_names = {'boolean' : bool,
                               'float' : float,
                               'integer' : int,
                               'string' : object,
                               'categorical' : CategoricalDtype,
                               'list' : list,
                               'datetime' : datetime64,
                               'timedelta' : timedelta}
        # Creates reversed dictionary of datatype_names.
        self.datatype_names_reversed = {
            value : key for key, value in self.datatype_names.items()}
        # Declares dictionary of DataFrames contained in Ingredients to allow
        # temporary remapping of attributes in __getattr__.
        self.default_dataframes = {'x' : self.x,
                                   'y' : self.y,
                                   'x_train' : self.x_train,
                                   'y_train' : self.y_train,
                                   'x_test' : self.x_test,
                                   'y_test' : self.y_test,
                                   'x_val' : self.x_val,
                                   'y_val' : self.y_val}
        # Sets lists of columns for specialized use by Cookbook.
        self.scalers = []
        self.encoders = []
        self.mixers = []
        # Maps class properties to appropriate DataFrames using the default
        # train_test setting.
        self._remap_dataframes(data_to_use = 'train_test')
        return self

    @check_df
    def add_unique_index(self, df = None, column = 'index_universal',
                         make_index = False):
        """Creates a unique integer index for each row.

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            column: string containing the column name for the index.
            make_index: boolean value indicating whether the index column
                should be made the index of the DataFrame."""
        if isinstance(df, pd.DataFrame):
            df[column] = range(1, len(df.index) + 1)
            self.datatypes.update({column, int})
            if make_index:
                df.set_index(column, inplace = True)
        else:
            error = 'To add an index, df must be a pandas DataFrame.'
            TypeError(error)
        return self

    @check_df
    def apply(self, df = None, func = None, **kwargs):
        """Allows users to pass a function to Ingredients instance which will
        be applied to the passed DataFrame (or uses default_df if none is
        passed).

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            func: function to be applied to the DataFrame.
            **kwargs: any arguments to be passed to func.
        """
        df = func(df, **kwargs)
        return self

    @check_df
    def auto_categorize(self, df = None, columns = None, threshold = 10):
        """Automatically assesses each column to determine if it has less than
        threshold unique values and is not boolean. If so, that column is
        converted to category type.

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            columns: a list of column names.
            threshold: integer of unique values necessary to form a category.
                If there are less unique values than the threshold, the column
                is converted to a category type. Otherwise, it will remain its
                current datatype.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if not column in self.booleans:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
            else:
                error = column + ' is not in ingredients DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def change_datatype(self, df = None, columns = None, prefixes = None,
                        datatype = str):
        """Changes column datatypes of columns passed or columns with the
        prefixes passed. datatype becomes the new datatype for the columns.

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            columns: a list of column names.
            prefixes: a list of prefix names.
            datatype: a string containing the datatype to convert the columns
                and columns with prefixes to.
        """
        if prefixes or columns:
            columns_list = self.create_column_list(df = df,
                                                   prefixes = prefixes,
                                                   columns = columns)
        else:
            columns_list = self.datatypes.keys()
        for column in columns_list:
            self.datatypes[column] = datatype
        self.convert_column_datatypes(df = df)
        return self

    @check_df
    def conform(self, df = None, step = None):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the current step.

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            step: string corresponding to the current state.
        """
        self.step = step
        for column, datatype in self.datatypes.items():
            if self.step in ['harvest', 'clean']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.datatypes[column] = str
            elif self.step in ['bundle', 'deliver']:
                if datatype in ['list', 'pattern']:
                    self.datatypes[column] = 'category'
        self.convert_column_datatypes(df = df)
        return self

    @check_df
    def convert_column_datatypes(self, df = None, raise_errors = False):
        """Attempts to convert all column data to the datatypes in
        self.datatypes.

        Parameters:
            df: pandas DataFrame. If none is provided, the default DataFrame
                is used.
            raise_errors: a boolean variable indicating whether errors should
                be raised when converting datatypes or ignored.
        """
        if raise_errors:
            raise_errors = 'raise'
        else:
            raise_errors = 'ignore'
        for column, datatype in self.datatypes.items():
            if not isinstance(datatype, str):
                df[column].astype(dtype = datatype,
                                  copy = False,
                                  errors = raise_errors)
        self.downcast(df = df)
        return self

    @check_df
    def convert_rare(self, df = None, columns = None, threshold = 0):
        """Converts categories rarely appearing within categorical columns
        to empty string if they appear below the passed threshold. threshold is
        defined as the percentage of total rows.

        Parameters:
            df: a pandas DataFrame.
            columns: a list of columns to check. If not passed, all columns
                in self.datatypes listed as 'categorical' type are used.
            threshold: a float indicating the percentage of values in rows
                below which a default_value is substituted.
        """
        if not columns:
            columns = self.categoricals
        for column in columns:
            if column in df.columns:
                df['value_freq'] = (df[column].value_counts() / len(df[column]))
                df[column] = np.where(df['value_freq'] <= threshold,
                                      self.default_values['categorical'],
                                      df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        if 'value_freq' in df.columns:
            df.drop('value_freq', axis = 'columns', inplace = True)
        return self

    @check_df
    def create_column_list(self, df = None, columns = None, prefixes = None):
        """Dynamically creates a new column list from a list of columns and/or
        lists of prefixes.

        Parameters:
            df: a pandas DataFrame.
            columns: a list of columns to be included in returned list.
            prefixes: a list of prefixes for columns to identify.
        """
        if prefixes:
            temp_list = []
            prefixes_list = []
            for prefix in listify(prefixes):
                temp_list = [col for col in df if col.startswith(prefix)]
                prefixes_list.extend(temp_list)
        if columns:
            if prefixes:
                columns = listify(columns) + prefixes_list
            else:
                columns = listify(columns)
        else:
            columns = prefixes_list
        return columns

    def create_series(self, columns = None, return_series = False):
        """Creates a Series (row) with the datatypes in columns.

        Parameters:
            columns: a list of index names for pandas series.
            return_series: boolean value indicating whether the Series should
                be returned (True) or assigned to attribute named in default_df
                (False):
        """
        # If columns is not passed, the keys of self.datatypes are used.
        if not columns and self.datatypes:
            columns = list(self.datatypes.keys())
        row = pd.Series(index = columns)
        # Fills series with default_values based on datatype.
        if self.datatypes:
            for column, datatype in self.datatypes.items():
                row[column] = self.default_values[datatype]
        if return_series:
            return row
        else:
            setattr(self, self.default_df, row)
            return self

    @check_df
    def decorrelate(self, df = None, columns = None, threshold = 0.95):
        """Drops all but one column from highly correlated groups of columns.
        threshold is based upon the .corr() method in pandas. columns can
        include any datatype accepted by .corr(). If columns is set to None,
        all columns in the DataFrame are tested.

        Parameters:
            df: a pandas DataFrame.
            threshold: a float indicating the level of correlation using
                pandas corr method above which a column is dropped.
        """
        if columns:
            corr_matrix = df[columns].corr().abs()
        else:
            corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k = 1).astype(np.bool))
        corrs = [col for col in upper.corrs if any(upper[col] > threshold)]
        self.drop_columns(columns = corrs)
        return self

    @check_df
    def downcast(self, df = None, columns = None):
        """Decreases memory usage by downcasting datatypes. For numerical
        datatypes, the method attempts to cast the data to unsigned integers if
        possible.

        Parameters:
            df: a pandas DataFrame.
            columns: a list of columns to downcast
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if self.datatypes[column] in ['boolean']:
                    df[column] = df[column].astype(bool)
                elif self.datatypes[column] in ['integer', 'float']:
                    try:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'integer')
                        if min(df[column] >= 0):
                            df[column] = pd.to_numeric(df[column],
                                                       downcast = 'unsigned')
                    except ValueError:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'float')
                elif self.datatypes[column] in ['categorical']:
                    df[column] = df[column].astype('category')
                elif self.datatypes[column] in ['list']:
                    df[column].apply(listify,
                                     axis = 'columns',
                                     inplace = True)
                elif self.datatypes[column] in ['datetime']:
                    df[column] = pd.to_datetime(df[column])
                elif self.datatypes[column] in ['timedelta']:
                    df[column] = pd.to_timedelta(df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def drop_columns(self, df = None, columns = None, prefixes = None):
        """Drops list of columns and columns with prefixes listed. In addition,
        any dropped columns are stored in the cumulative dropped_columns
        list.

        Parameters:
            df: a pandas DataFrame.
            columns: a list of columns to drop.
            prefixes: a list of prefixes for columns to drop.
        """
        columns = self.create_column_list(columns = columns,
                                          prefixes = prefixes)
        df.drop(columns, axis = 'columns', inplace = True)
        self.dropped_columns.extend(columns)
        return self

    @check_df
    def drop_infrequent(self, df = None, columns = None, threshold = 0):
        """Drops boolean columns that rarely are True. This differs
        from the sklearn VarianceThreshold class because it is only
        concerned with rare instances of True and not False. This enables
        users to set a different variance threshold for rarely appearing
        information. threshold is defined as the percentage of total rows (and
        not the typical variance formulas used in sklearn).

        Parameters:
            df: a pandas DataFrame.
            columns: a list of columns to check. If not passed, all boolean
                columns will be used.
            threshold: a float indicating the percentage of True values in a
                boolean column that must exist for the column to be kept.
        """
        if not columns:
            columns = self.booleans
        infrequents = []
        for column in self.booleans:
            if column in columns:
                if df[column].mean() < threshold:
                    infrequents.append(column)
        self.drop_columns(columns = infrequents)
        return self

    @check_df
    def infer_datatypes(self, df = None):
        """Infers column datatypes and adds those datatypes to types. This
        method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.). This also
        allows the user to choose which datatypes to look for by changing the
        default_values dictionary. Non-standard python datatypes cannot be
        inferred.

        Parameters:
            df: a pandas DataFrame.
        """
        # Creates list of all possible datatypes.
        self._all_datatypes = list(self.datatype_names.values())
        # Makes datatypes dictionary from inferred datatypes.
        if not self.datatypes:
            self.datatypes = {}
        for datatype in self._all_datatypes:
            type_columns = df.select_dtypes(
                include = [datatype]).columns.to_list()
            self.datatypes.update(
                dict.fromkeys(type_columns,
                              self.datatype_names_reversed[datatype]))
        return self

    def load(self, name = None, file_path = None, folder = None,
             file_name = None, file_format = None):
        """Loads DataFrame or Series into Ingredients instance from a file.

        Parameters:
            name: name of attribute for DataFrame or Series to be stored.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Inventory.extensions.
        """
        # If name is not provided, the attribute name in default_df is used.
        if not name:
            name = self.default_df
        setattr(self, name, self.inventory.load(file_path = file_path,
                                                folder = folder,
                                                file_name = file_name,
                                                file_format = file_format))
        return self

    def prepare(self):
        """Prepares Ingredients class instance."""
        if self.verbose:
            print('Preparing ingredients')
        # If self.df is a path_name, the file located there is imported.
        if (not(isinstance(self.df, pd.DataFrame) or
                isinstance(self.df, pd.Series))
                and os.path.isfile(self.df)):
            self.load(name = 'df', file_path = self.df)
        # Initializes a list of dropped column names so that users can track
        # which features are omitted from analysis.
        self.dropped_columns = []
        # If datatypes passed, checks to see if columns are in df. Otherwise,
        # datatypes are inferred.
        self._initialize_datatypes()
        # Sets class for summarizing DataFrames in Ingredients.
        self.summarizer = Summary()
        return self

    @check_df
    def save(self, df = None, file_path = None, folder = None, file_name = None,
             file_format = None):
        """Exports a DataFrame or Series attribute to disc.

        Parameters:
            df: a pandas DataFrame.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                Inventory.extensions.
        """
        if self.verbose:
            print('Saving ingredients')
        self.inventory.save(variable = df,
                            file_path = file_path,
                            folder = folder,
                            file_name = file_name,
                            file_format = file_format)
        return

    def save_dropped(self, file_name = 'dropped_columns', file_format = 'csv'):
        """Saves dropped_columns into a file

        Parameters:
            file_name: string containing name of file to be exported.
            file_format: string of file extension from Inventory.extensions.
        """
        # Deduplicates dropped_columns list
        self.dropped_columns = list(unique_everseen(self.dropped_columns))
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            self.inventory.save(variable = self.dropped_columns,
                                folder = self.inventory.experiment,
                                file_name = file_name,
                                file_format = file_format)
        elif self.verbose:
            print('No features were dropped during preprocessing.')
        return

    @check_df
    def smart_fill(self, df = None, columns = None):
        """Fills na values in DataFrame to defaults based upon the datatype
        listed in the columns dictionary.

        Parameters:
            df: a pandas DataFrame.
            columns: list of columns to fill missing values in.
        """
        for column in self._check_columns(columns):
            if column in df:
                default_value = self.default_values[self.datatypes[column]]
                df[column].fillna(default_value, inplace = True)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def split_xy(self, df = None, label = 'label'):
        """Splits df into x and y based upon the label passed.

        Parameters:
            df: a pandas DataFrame.
            label: name of column(s) to be stored in self.y
        """
        self.x = df.drop(label, axis = 'columns')
        self.y = df[label]
        # drops columns in self.y from datatypes dictionary.
        self._crosscheck_columns(df = self.x)
        return self

    @check_df
    def summarize(self, df = None, transpose = True, file_name = 'data_summary',
                  file_format = 'csv'):
        """Creates and exports a DataFrame of common summary data using the
        Summary class.

        Parameters:
            df: a pandas DataFrame.
            transpose: boolean value indicating whether the df columns should be
                listed horizontally (True) or vertically (False) in report.
            file_name: string containing name of file to be exported.
            file_format: string of file extension from Inventory.extensions.
        """
        self.summarizer.start(df = df, transpose = transpose)
        if self.verbose:
            print('Saving ingredients summary data')
        self.inventory.save(variable = self.summarizer.report,
                            folder = self.inventory.experiment,
                            file_name = file_name,
                            file_format = file_format,
                            header = self.summarizer.df_header,
                            index = self.summarizer.df_index)
        return self
