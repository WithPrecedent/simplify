
import csv
from datetime import timedelta
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
import os
import requests

from more_itertools import unique_everseen
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .almanac.blackacre import Blackacre
from .cookbook.countertop import Countertop


@dataclass
class Ingredients(Countertop, Blackacre):
    """Imports, stores, and exports pandas dataframes and series, as well as
    related information about those data containers.

    Ingredients uses pandas dataframes or series for all data storage, but its
    subclasses utilize faster numpy methods where possible to increase
    performance. Ingredients stores the data itself as well as a set of related
    variables about the data.

    Dataframes stored in ingredients can be imported and exported using the
    load and save methods. Current file formats supported are csv, feather, and
    hdf5.

    A Menu object needs to be passed when a Ingredients instance is created.
    If the quick_start option is selected, an Inventory object must be passed
    as well.

    Ingredients adds easy-to-use methods for common feature engineering
    techniques. Methods include converting rarely appearing categories to a
    default value (convert_rare), dropping boolean columns with infrequent True
    values (drop_infrequent), and reshaping dataframes (reshape_wide and
    reshape_long). There are methods for creating column dictionaries for the
    different data types commonly appearing in machine learning scripts
    (column_types and create_column_list). Any function can be applied to a
    dataframe contained in Engineer by using the apply method.

    Ingredients also includes some methods which are designed to be accessible
    and user-friendly than the commonly-used methods. For example, data can
    easily be downcast to save memory with the downcast method and
    smart_fill_na fills na data with appropriate defaults based upon the column
    datatypes (either provided by the user via column_dict or through
    inference).

    Attributes:
        df: a pandas dataframe or series.
        menu: an instance of Menu.
        inventory: an instance of Inventory.
        quick_start: a boolean variable indicating whether data should
            automatically be loaded into the df attribute.
        default_df: the current default dataframe or series attribute that will
            be used when a specific dataframe is not passed to a class method.
            The value is a string corresponding to the attribute dataframe
            name and is initially set to 'df'.
        x, y, x_train, y_train, x_test, y_test, x_val, y_val: dataframes or
            series. These dataframes (and corresponding columns dictionaries)
            need not be passed when the class is instanced. They are merely
            listed for users who already have divided datasets and still wish
            to use the siMpLify package.
        columns_dict: dictionary containing column names and datatypes for df
            or x (if data has been split) dataframes or series
    """
    df : object = None
    menu : object = None
    inventory : object = None
    quick_start : bool = False
    default_df : str = 'df'
    x : object = None
    y : object = None
    x_train : object = None
    y_train : object = None
    x_test : object = None
    y_test : object = None
    x_val : object = None
    y_val : object = None
    columns_dict : object = None


    def __post_init__(self):
        """Localizes menu, initializes quick_start if that option is
        selected and infers column datatypes if a pandas dataframe or series
        is passed.
        """
        if self.menu:
            self.menu.localize(instance = self, sections = ['general'])
        else:
            error = 'Ingredients requires a Menu object'
            raise AttributeError(error)
        if self.verbose:
            print('Building ingredients')
        # If quick_start is set to true and a menu dictionary is passed,
        # ingredients is automatically loaded according to user specifications
        # in the menu file.
        if self.quick_start:
            if self.inventory:
                self.load(import_path = self.inventory.import_path,
                          test_data = self.inventory.test_data,
                          test_rows = self.inventory.test_chunk,
                          encoding = self.inventory.encoding)
            else:
                error = 'Ingredients quick_start requires an Inventory object'
                raise AttributeError(error)
        # Sets default options for Ingredients.
        self._set_defaults()
        # If column_dict passed, checks to see if columns are in df. Otherwise,
        # datatypes are inferred.
        self._initialize_columns()
        return self
#
#    def __delitem__(self, name):
#        return delattr(self, name)
#
    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return len(getattr(self, self.default_df))

    def __repr__(self):
        return getattr(self, self.default_df)
#
#    def __setitem__(self, name, value):
#        return setattr(self, name, value)

    def __str__(self):
        return getattr(self, self.default_df)

    def check_df(func):
        """Decorator which automatically uses the default dataframe if one
        is not passed to the decorated method.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            unpassed_args = argspec.args[len(args):]
            if 'df' in argspec.args and 'df' in unpassed_args:
                kwargs.update({'df' : getattr(self, self.default_df)})
            return func(self, *args, **kwargs)
        return wrapper

    def _check_columns(self, columns = None):
        if columns:
            return columns
        else:
            return self.columns

    @check_df
    def _crosscheck_columns(self, df = None):
        """Removes any columns in columns dictionary, but not in dataframe."""
        for column, datatype in self.columns.items():
            if column not in df.columns:
                self.columns.pop(column)
        return self

    def _deduplicate(self, type_list):
        """Removes duplicates from a list"""
        return list(unique_everseen(type_list))

    def _get_columns_by_type(self, datatype):
        column_list = self._deduplicate(
            [key for key, value in self.columns.items() if value == datatype])
        return column_list

    def _initialize_columns(self, df = None):
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            if not self.columns:
                self.infer_datatypes(df = df)
            else:
                self._crosscheck_columns(df = df)
        elif (isinstance(getattr(self, self.default_df), pd.DataFrame)
                or isinstance(getattr(self, self.default_df), pd.Series)):
            if not self.columns:
                self.infer_datatypes()
            else:
                self._crosscheck_columns()
        elif isinstance(self.x, pd.DataFrame) or isinstance(self.x, pd.Series):
            if not self.columns:
                self.infer_datatypes(df = self.x)
            else:
                self._crosscheck_columns(df = self.x)
        elif (isinstance(self.x_train, pd.DataFrame)
                or isinstance(self.x_train, pd.Series)):
            if not self.columns:
                self.infer_datatypes(df = self.x_train)
            else:
                self._crosscheck_columns(df = self.x_train)
        return self

    def _listify(self, ingredients):
        """Checks to see if the methods are stored in a list. If not, the
        methods are converted to a list or a list of 'none' is created.
        """
        if not ingredients:
            return ['none']
        elif isinstance(ingredients, list):
            return ingredients
        else:
            return [ingredients]

    def _set_defaults(self):
        # Sets default values for missing data based upon datatype of column.
        self._default_values = {bool : False,
                                float : 0.0,
                                int : 0,
                                object : '',
                                CategoricalDtype : '',
                                list : [],
                                np.datetime64 : 1/1/1900,
                                timedelta : 0,
                                'mixer' : '',
                                'scaler' : 0,
                                'encoder' : ''}
        # Initializes a list of dropped column names so that users can track
        # which features are omitted from analysis.
        self._dropped_columns = []
        return self

    @check_df
    def _update_columns(self, df = None):

        return self

    @property
    def booleans(self):
        """Returns boolean columns."""
        return self._get_columns_by_type(bool)

    @property
    def categoricals(self):
        """Returns caterogical columns."""
        return self._get_columns_by_type(CategoricalDtype)

    @property
    def columns(self, datatype = dict):
        if not self.columns_dict:
            self.columns_dict = {}
        if datatype in [dict]:
            return self.columns_dict
        elif datatype in [list]:
            return list(self.columns_dict.keys())
        else:
            error = 'columns can only return dict and list datatypes'
            raise TypeError(error)
            return self

    @columns.setter
    def columns(self, column, datatype):
        self.column_dict.update({column : datatype})
        return self

    @property
    def datetimes(self):
        """Returns datetime columns."""
        return self._get_columns_by_type(np.datetime64)

    @property
    def default_values(self):
        """Returns current default values for datatypes."""
        return self._default_values

    @property
    def dropped(self):
        """Returns list of dropped columns."""
        return self.deduplicate(self._dropped_columns)

    @property
    def encoders(self):
        """Returns columns with 'encoder' datatype."""
        return self._get_columns_by_type('encoder')

    @property
    def floats(self):
        """Returns float columns."""
        return self._get_columns_by_type(float)

    @property
    def integers(self):
        """Returns int columns."""
        return self._get_columns_by_type(int)

    @property
    def lists(self):
        """Returns list columns."""
        return self._get_columns_by_type(list)

    @property
    def mixers(self):
        """Returns columns with 'mixer' datatype."""
        return self._get_columns_by_type('mixer')

    @property
    def numerics(self):
        """Returns float and int columns."""
        return self._deduplicate(self.floats + self.integers)

    @property
    def scalers(self):
        """Returns columns with 'scaler' datatype."""
        return self._get_columns_by_type('scaler')

    @property
    def strings(self):
        """Returns str (object type) columns."""
        return self._get_columns_by_type(object)

    @property
    def timedeltas(self):
        """Returns timedelata columns."""
        return self._get_columns_by_type(timedelta)

    @property
    def full(self):
        """Returns the full dataset divided into x and y."""
        return self.x, self.y

    @property
    def test(self):
        """Returns the test data."""
        return self.x_test, self.y_test

    @property
    def train_test(self):
        """Returns the training and testing data."""
        return self.x_train, self.y_train, self.x_test, self.y_test

    @property
    def train(self):
        """Returns the training data."""
        return self.x_train, self.y_train

    @property
    def train_test_val(self):
        """Returns the training, test, and validation data."""
        return (self.x_train, self.y_train, self.x_test, self.y_test,
                self.x_val, self.y_val)

    @property
    def train_val(self):
        """Returns the training and validation data."""
        return self.x_train, self.y_train, self.x_val, self.y_val

    @property
    def val(self):
        """Returns the validation data."""
        return self.x_val, self.y_val

    def add_datatype(self, name, default_value):
        self._default_values.update({name : default_value})
        return self

    @check_df
    def add_unique_index(self, df = None, column = 'index_universal',
                         make_index = False):
        """Creates a unique integer index for each row."""
        if isinstance(df, pd.DataFrame):
            df[column] = range(1, len(df.index) + 1)
            self.columns_dict.update({column, int})
            if make_index:
                df.set_index(column, inplace = True)
        else:
            error = 'To add an index, df must be a pandas dataframe.'
            TypeError(error)
        return self

    @check_df
    def apply(self, df = None, func = None, **kwargs):
        """Allows users to pass a function to Ingredients instance which will
        be applied to the passed dataframe (or uses default_df if none is
        passed).
        """
        df = func(df, **kwargs)
        return self

    @check_df
    def auto_categorize(self, df = None, columns = None, threshold = 10):
        """Automatically assesses each column to determine if it has less than
        threshold unique values and is not boolean. If so, that column is
        converted to category type.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                if not column in self.booleans:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
            else:
                error = column + ' is not in ingredients dataframe'
                raise KeyError(error)
        return self

    @check_df
    def change_type(self, df = None, columns = None, prefixes = None,
                    datatype = str):
        """Changes column datatypes of columns passed or columns with the
        prefixes passed. datatype becomes the new datatype for the columns.
        """
        columns_list = self.create_column_list(df = df,
                                               prefixes = prefixes,
                                               columns = columns)
        for column in columns_list:
            self.columns[column] = datatype
        return self

    @check_df
    def convert_rare(self, df = None, columns = None, threshold = 0):
        """Converts categories rarely appearing within categorical columns
        to empty string if they appear below the passed threshold. threshold is
        defined as the percentage of total rows.
        """
        for column in self._check_columns(columns):
            if column in df.columns:
                default_value = self._default_values[CategoricalDtype]
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
            for prefix in prefixes:
                temp_list = [col for col in df if col.startswith(prefix)]
                prefixes_list.extend(temp_list)
        if columns:
            if prefixes:
                column_list = columns + prefixes_list
            else:
                column_list = columns
        else:
            column_list = prefixes_list
        return column_list

    @check_df
    def decorrelate(self, df = None, threshold = 0.95):
        """Drops all but one column from highly correlated groups of columns.
        threshold is based upon the .corr() method in pandas. columns can
        include any datatype accepted by .corr(). If columns is set to 'all',
        all columns in the dataframe are tested.
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
                    df[column].apply(self._listify,
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

    def download(self, file_path, file_url):
        """Downloads file from a URL if the file is available."""
        file_response = requests.get(file_url)
        with open(file_path, 'wb') as a_file:
            a_file.write(file_response.content)
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

    def initialize_series(self, df = None):
        """Creates a series (row) with the datatypes in columns."""
        row = pd.Series(index = self.columns.keys())
        for column, datatype in self.columns.items():
            row[column] = self.default_values[datatype]
        if not df:
            setattr(self, self.default_df, row)
            return self
        else:
            return row

    def load(self, import_folder = '', file_name = 'ingredients',
             import_path = '', file_type = 'csv', usecolumns = None,
             index = False, encoding = 'windows-1252', test_data = False,
             test_rows = 500, return_df = False, message = 'Importing data'):
        """Imports pandas dataframes from different file formats."""
        if not import_path:
            if not import_folder:
                import_folder = self.inventory.import_folder
            import_path = self.inventory.make_path(folder = import_folder,
                                               file_name = file_name,
                                               file_type = file_type)
        if self.verbose:
            print(message)
        if test_data:
            nrows = test_rows
        else:
            nrows = None
        if file_type == 'csv':
            df = pd.read_csv(import_path,
                             index_col = index,
                             nrows = nrows,
                             usecols = usecolumns,
                             encoding = encoding,
                             low_memory = False)

        elif file_type == 'h5':
            df = pd.read_hdf(import_path,
                             chunksize = nrows)
        elif file_type == 'feather':
            df = pd.read_feather(import_path,
                                 nthreads = -1)
        if not return_df:
            setattr(self, self.default_df, df)
            return self
        else:
            return df

    @check_df
    def reshape_long(self, df = None, stubs = None, id_col = '', new_col = '',
                     sep = ''):
        """A simple wrapper method for pandas wide_to_long method using more
        intuitive parameter names than 'i' and 'j'.
        """
        df = (pd.wide_to_long(df,
                              stubnames = stubs,
                              i = id_col,
                              j = new_col,
                              sep = sep).reset_index())
        return self

    @check_df
    def reshape_wide(self, df = None, df_index = '', columns = None,
                     values = None):
        """A simple wrapper method for pandas pivot method named as
        corresponding method to reshape_long.
        """
        df = (df.pivot(index = df_index,
                       columns = columns,
                       values = values).reset_index())
        return self

    @check_df
    def save(self, df = None, file_name = None):
        """Exports pandas dataframe."""
        if self.verbose:
            print('Saving', file_name)
        export_path = os.path.join(self.inventory.data_out, file_name)
        self.inventory.save_df(df, export_path)
        return

    def save_drops(self, file_name = 'dropped_columns', export_path = ''):
        """Saves dropped_columns into a .csv file."""
        self._dropped_columns = list(unique_everseen(self._dropped_columns))
        if not export_path:
            export_path = self.inventory.create_path(
                    folder = self.inventory.experiment,
                    file_name = file_name,
                    file_type = 'csv')
        if self._dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            with open(export_path, 'wb') as export_file:
                csv_writer = csv.writer(export_file)
                csv_writer.writerow(self.dropped_columns)
        return

    def scrape(self, file_path, file_url):
        return self

    def set_default_value(self, datatype, default_value):
        self._default_values[datatype] = default_value
        return self

    @check_df
    def smart_fillna(self, df = None, columns = None):
        """Fills na values in dataframe to defaults based upon the datatype
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
        """Creates a dataframe of common summary data.

        summarize is more inclusive than pandas.describe() and includes
        boolean and numerical columns by default. If an export_path is passed,
        the summary table is automatically saved to disc.
        """
        summary_columns = ['variable', 'datatype', 'count', 'min', 'q1',
                           'median', 'q3', 'max', 'mad', 'mean', 'stan_dev',
                           'mode', 'sum']
        self.summary = pd.DataFrame(columns = summary_columns)
        for i, col in enumerate(df.columns):
            new_row = pd.Series(index = summary_columns)
            new_row['variable'] = col
            new_row['datatype'] = df[col].dtype
            new_row['count'] = len(df[col])
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            if df[col].dtype.kind in 'bifcu':
                new_row['min'] = df[col].min()
                new_row['q1'] = df[col].quantile(0.25)
                new_row['median'] = df[col].median()
                new_row['q3'] = df[col].quantile(0.75)
                new_row['max'] = df[col].max()
                new_row['mad'] = df[col].mad()
                new_row['mean'] = df[col].mean()
                new_row['stan_dev'] = df[col].std()
                new_row['mode'] = df[col].mode()[0]
                new_row['sum'] = df[col].sum()
            self.summary.loc[len(self.summary)] = new_row
        self.summary.sort_values('variable', inplace = True)
        if not transpose:
            self.summary = self.summary.transpose()
            df_header = False
            df_index = True
        else:
            df_header = True
            df_index = False
        if export_summary:
            if self.verbose:
                print('Saving ingredients summary data')
            summary_path = os.path.join(self.inventory.experiment,
                                       'data_summary.csv')
            self.inventory.save_df(df, summary_path, df_header = df_header,
                                   df_index = df_index)

        return self