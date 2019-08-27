
import csv
from datetime import timedelta
from dataclasses import dataclass
from more_itertools import unique_everseen

import numpy as np
import pandas as pd
from numpy import datetime64
from pandas.api.types import CategoricalDtype

from .decorators import check_df
from .tools import listify


@dataclass
class Ingredients(object):
    """Imports, stores, and exports pandas DataFrames and Series, as well as
    related information about those data containers.

    Ingredients uses pandas DataFrames or Series for all data storage, but its
    subclasses utilize faster numpy methods where possible to increase
    performance. Ingredients stores the data itself as well as a set of related
    variables about the data.

    DataFrames stored in ingredients can be imported and exported using the
    load and save methods. Current file formats supported are csv, feather, and
    hdf5.

    A Menu object needs to be passed when a Ingredients instance is created.
    If the auto_prepare option is selected, an Inventory object must be passed
    as well.

    Ingredients adds easy-to-use methods for common feature engineering
    techniques. There are methods for creating column dictionaries for the
    different data types commonly appearing in machine learning scripts
    (column_types and create_column_list). Any function can be applied to a
    DataFrame contained in Ingredients by using the apply method.

    Ingredients also includes some methods which are designed to be accessible
    and user-friendly than the commonly-used methods. For example, data can
    easily be downcast to save memory with the downcast method and
    smart_fill fills na data with appropriate defaults based upon the column
    datatypes (either provided by the user via datatypes or through
    inference).

    Attributes:

        menu: an instance of Menu.
        inventory: an instance of Inventory.
        df: a pandas DataFrame or Series.
        auto_prepare: a boolean variable indicating whether prepare method
            should be called when class instanced.
        default_df: a string listing the current default DataFrame or Series
            attribute that will be used when a specific DataFrame is not passed
            to a method. The default value is initially set to 'df'.
        x, y, x_train, y_train, x_test, y_test, x_val, y_val: DataFrames or
            Series. These DataFrames and Series need not be passed when the
            class is instanced. They are merely listed for users who already
            have divided datasets and still wish to use the siMpLify package.
        datatypes: dictionary containing column names and datatypes for df
            or x (if data has been split) DataFrames or Series.
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
            self.dataframes.pop(item)
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
                    'categoricals', 'list', 'datetime', 'timedelta', 'mixer',
                    'scaler', 'encoder']:
            return self._get_columns_by_type(self._datatype_names[attr])
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
                    'categoricals', 'list', 'datetime', 'timedelta', 'mixer',
                    'scaler', 'encoder']:
            self.__dict__['datatypes'].update(
                    dict.fromkeys(listify(self._datatype_names[attr]), value))
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
        """Returns self.columns if columns doesn't exist."""
        return columns or self.columns

    @check_df
    def _crosscheck_columns(self, df = None):
        """Removes any columns in columns dictionary, but not in DataFrame."""
        for column, datatype in self.columns.items():
            if column not in df.columns:
                self.columns.pop(column)
        return self

    def _get_columns_by_type(self, datatype):
        """Returns list of columns of the specified datatype."""
        return (
            [key for key, value in self.columns.items() if value == datatype])

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

    def _remap_dataframes(self, data_to_use):
        """Remaps DataFrames returned by various properties of Ingredients so
        that methods and classes of siMpLify can use the same labels for
        analyzing the Ingredients DataFrames.
        """
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
        self.default_values = {bool : False,
                               float : 0.0,
                               int : 0,
                               object : '',
                               CategoricalDtype : '',
                               list : [],
                               datetime64 : 1/1/1900,
                               timedelta : 0,
                               'mixer' : '',
                               'scaler' : 0,
                               'encoder' : ''}
        # Sets string names of various datatypes available.
        self.datatype_names = {'booleans' : bool,
                               'floats' : float,
                               'integers' : int,
                               'strings' : object,
                               'categoricals' : CategoricalDtype,
                               'list' : list,
                               'datetime' : datetime64,
                               'timedelta' : timedelta,
                               'mixer' : 'mixer',
                               'scaler' : 'scaler',
                               'encoder' : 'encoder'}
        # Declares dictionary of DataFrames contained in Ingredients to allow
        # temporary remapping.
        self.default_dataframes = {'x' : self.x,
                                   'y' : self.y,
                                   'x_train' : self.x_train,
                                   'y_train' : self.y_train,
                                   'x_test' : self.x_test,
                                   'y_test' : self.y_test,
                                   'x_val' : self.x_val,
                                   'y_val' : self.y_val}
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
            columns_list = self.columns.keys()
        for column in columns_list:
            self.columns[column] = datatype
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
        for column, datatype in self.columns.items():
            if self.step in ['harvest', 'clean']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.columns[column] = str
            elif self.step in ['bundle', 'deliver']:
                if datatype in ['list', 'pattern']:
                    self.columns[column] = 'category'
        self.convert_column_datatypes(df = df)
        return self

    @check_df
    def convert_column_datatypes(self, df = None, raise_errors = False):
        """Attempts to convert all column data to the datatypes list in
        self.columns.

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
        for column, datatype in self.columns.items():
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
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                default_value = self.default_values[CategoricalDtype]
                df['value_freq'] = (df[column].value_counts()
                                    / len(df[column]))
                df[column] = np.where(df['value_freq'] <= threshold,
                                      default_value, df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        df.drop('value_freq', axis = 'columns', inplace = True)
        return self

    @check_df
    def create_column_list(self, df = None, columns = None, prefixes = None):
        """Dynamically creates a new column list from a list of columns and/or
        lists of prefixes.
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

    def create_series(self, df = None, columns = None):
        """Creates a Series (row) with the datatypes in columns."""
        if not columns:
            columns = list(self.columns.keys())
        row = pd.Series(index = columns)
        if self.columns:
            for column, datatype in self.columns.items():
                row[column] = self.default_values[datatype]
        if not df:
            setattr(self, self.default_df, row)
            return self
        else:
            return row

    @check_df
    def decorrelate(self, df = None, threshold = 0.95):
        """Drops all but one column from highly correlated groups of columns.
        threshold is based upon the .corr() method in pandas. columns can
        include any datatype accepted by .corr(). If columns is set to 'all',
        all columns in the DataFrame are tested.
        """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k = 1).astype(np.bool))
        columns = [col for col in upper.columns if any(upper[col] > threshold)]
        self.drop_columns(columns = columns)
        return self

    @check_df
    def downcast(self, df = None, columns = None):
        """Decreases memory usage by downcasting datatypes. For numerical
        datatypes, the method attempts to cast the data to unsigned integers if
        possible.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if self.columns[column] in [bool]:
                    df[column] = df[column].astype(bool)
                elif self.columns[column] in [int, float]:
                    try:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'integer')
                        if min(df[column] >= 0):
                            df[column] = pd.to_numeric(df[column],
                                                       downcast = 'unsigned')
                    except ValueError:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'float')
                elif self.columns[column] in [CategoricalDtype]:
                    df[column] = df[column].astype('category')
                elif self.columns[column] in [list]:
                    df[column].apply(listify,
                                     axis = 'columns',
                                     inplace = True)
                elif self.columns[column] in [np.datetime64]:
                    df[column] = pd.to_datetime(df[column])
                elif self.columns[column] in [timedelta]:
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
        not the typical variance formulas).
        """
        cols = []
        for column in self._check_columns(df):
            if column in df.columns:
                if df[column].mean() < threshold:
                    df.drop(column, axis = 'columns', inplace = True)
                    cols.append(column)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        self.drop_columns(columns = cols)
        return self

    @check_df
    def infer_datatypes(self, df = None):
        """Infers column datatypes and adds those datatypes to types. This
        method is an alternative to default pandas methods which can use
        complex datatypes (e.g., int8, int16, int32, int64, etc.). This also
        allows the user to choose which datatypes to look for by changing the
        default_values dictionary. Non-standard python datatypes cannot be
        inferred."""
        # Creates list of all possible datatypes.
        self._all_datatypes = list(self.default_values.keys())
        # Gets corresponding columns dictionary and initializes it if
        # necessary.
        for datatype in self._all_datatypes:
            if not isinstance(datatype, str):
                type_columns = df.select_dtypes(
                        include = [datatype]).columns.to_list()
                self.columns.update(dict.fromkeys(type_columns, datatype))
        return self

    def load(self, folder = None, file_name = None, file_path = None,
             file_format = None):
        """Loads DataFrame into Ingredients instance from a file."""
        setattr(self, self.default_df, self.inventory.load(
                folder = folder,
                file_name = file_name,
                file_path = file_path,
                file_format = file_format))
        return self

    def prepare(self):
        """Prepares Ingredients class instance."""
        if self.verbose:
            print('Preparing ingredients')
        if (not (isinstance(getattr(self, self.default_df), pd.DataFrame)
                 or isinstance(getattr(self, self.default_df), pd.Series))
                 and self.auto_load):
            self.load()
        # Initializes a list of dropped column names so that users can track
        # which features are omitted from analysis.
        self.dropped_columns = []
        # If datatypes passed, checks to see if columns are in df. Otherwise,
        # datatypes are inferred.
        self._initialize_datatypes()
        self.summarizer = Summary()
        return self

    @check_df
    def save(self, df = None, folder = None, file_name = None,
             file_path = None, file_format = None):
        """Exports a DataFrame or Series attribute to disc."""
        self.inventory.save(variable = df,
                            folder = folder,
                            file_name = file_name,
                            file_path = file_path,
                            file_format = file_format)
        return

    def save_drops(self, file_name = 'dropped_columns', export_path = ''):
        """Saves dropped_columns into a .csv file."""
        self.dropped_columns = list(unique_everseen(self.dropped_columns))
        if not export_path:
            export_path = self.inventory.create_path(
                    folder = self.inventory.experiment,
                    file_name = file_name,
                    file_format = 'csv')
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            with open(export_path, 'wb') as export_file:
                csv_writer = csv.writer(export_file)
                csv_writer.writerow(self.dropped_columns)
        return

    @check_df
    def smart_fill(self, df = None, columns = None):
        """Fills na values in DataFrame to defaults based upon the datatype
        listed in the columns dictionary.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                default_value = self.default_values[self.columns[column]]
                df[column].fillna(default_value, inplace = True)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def split_xy(self, df = None, label = 'label'):
        """Splits ingredients into x and y based upon the label passed."""
        self.x = df.drop(label, axis = 'columns')
        self.y = df[label]
        self._crosscheck_columns(df = self.x)
        return self

    @check_df
    def summarize(self, df = None, export_path = '', export_summary = True,
                  transpose = True):
        """Creates a DataFrame of common summary data.

        summarize is more inclusive than pandas.describe() and includes
        boolean and numerical columns by default. If an export_path is passed,
        the summary table is automatically saved to disc.
        """
#        summary_columns = ['variable', 'datatype', 'count', 'min', 'q1',
#                           'median', 'q3', 'max', 'mad', 'mean', 'stan_dev',
#                           'mode', 'sum']
#        self.summary = pd.DataFrame(columns = summary_columns)
#        for i, col in enumerate(df.columns):
#            new_row = pd.Series(index = summary_columns)
#            new_row['variable'] = col
#            new_row['datatype'] = df[col].dtype
#            new_row['count'] = len(df[col])
#            if df[col].dtype == bool:
#                df[col] = df[col].astype(int)
#            if df[col].dtype.kind in 'bifcu':
##                new_row['min'] = df[col].min()
#                new_row['min'] = getattr(df[col], 'min')()
#                new_row['q1'] = getattr(df[col], 'quantile')(0.25)
#                new_row['median'] = df[col].median()
#                new_row['q3'] = df[col].quantile(0.75)
#                new_row['max'] = df[col].max()
#                new_row['mad'] = df[col].mad()
#                new_row['mean'] = df[col].mean()
#                new_row['stan_dev'] = df[col].std()
#                new_row['mode'] = df[col].mode()[0]
#                new_row['sum'] = df[col].sum()
#            self.summary.loc[len(self.summary)] = new_row
#        self.summary.sort_values('variable', inplace = True)
#        if not transpose:
#            self.summary = self.summary.transpose()
#            df_header = False
#            df_index = True
#        else:
#            df_header = True
#            df_index = False
#        if export_summary:
#            if self.verbose:
#                print('Saving ingredients summary data')
#            self.inventory.save(variable = df,
#                                folder = self.inventory.experiment,
#                                file_format = 'csv',
#                                file_name = 'data_summary',
#                                header = df_header,
#                                index = df_index)
        self.summarizer.start(df = df, transpose = transpose)
        if self.verbose:
            print('Saving ingredients summary data')
        self.inventory.save(variable = self.summarizer.report,
                            folder = self.inventory.experiment,
                            file_format = 'csv',
                            file_name = 'data_summary',
                            header = self.summarizer.df_header,
                            index = self.summarizer.df_index)
        return self

@dataclass
class Summary(object):

    auto_prepare : bool = True

    def __post_init__(self):
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    def _set_defaults(self):
        self.options = {'datatype' : ['dtype'],
                        'count' : 'count',
                        'min' :'min',
                        'q1' : ['quantile', 0.25],
                        'median' : 'median',
                        'q3' : ['quantile', 0.75],
                        'max' : 'max',
                        'mad' : 'mad',
                        'mean' : 'mean',
                        'stan_dev' : 'std',
                        'mode' : ['mode', [0]],
                        'sum' : 'sum',
                        'kurtosis' : 'kurtosis',
                        'skew' : 'skew',
                        'variance' : 'var',
                        'stan_error' : 'sem',
                        'unique' : 'nunique'}
        return self

    def prepare(self):
        self.columns = ['variable']
        self.columns.extend(list(self.options.keys()))
        self.report = pd.DataFrame(columns = self.columns)
        return self

    def start(self, df = None, transpose = True):
        for i, column in enumerate(df.columns):
            row = pd.Series(index = self.columns)
            row['variable'] = column
            if df[column].dtype == bool:
                df[column] = df[column].astype(int)
            if df[column].dtype.kind in 'bifcu':
                for key, value in self.options.items():
                    if isinstance(value, str):
                        row[key] = getattr(df[column], value)()
                    elif isinstance(value, list):
                        if len(value) < 2:
                            row[key] = getattr(df[column], value[0])
                        elif isinstance(value[1], list):
                            row[key] = getattr(df[column],
                               value[0])()[value[1]]
                        else:
                            row[key] = getattr(df[column],
                               value[0])(value[1])
            self.report.loc[len(self.report)] = row
        if not transpose:
            self.report = self.report.transpose()
            self.df_header = False
            self.df_index = True
        else:
            self.df_header = True
            self.df_index = False
        return self