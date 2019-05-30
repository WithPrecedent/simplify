"""
Class for organizing and loading data into machine learning algorithms. Data
uses pandas dataframes or series for all data, but utilizes faster numpy
methods where possible to increase performance. It is part of the siMpLify
package which is designed to make python machine learning more accessible and
easier to use.

Data stores the data itself as well as a set of settings and variables about
the data stored (primarily column data types).

Data incorporates easy-to-use methods for common feature engineering techniques
such as converting rarely appearing categories to a default value
(convert_rare), dropping boolean columns with infrequent True values
(drop_infrequent), and reshaping dataframes (reshape_wide and reshape_long).
There are methods for creating column dictionaries for the different
data types commonly appearing in machine learning scripts (column_types and
create_column_list). Any function can be applied to a dataframe contained
in Data by using the apply method.

Data also includes some methods which are designed to be more accessible and
user-friendly than the commonly-used methods. For example, data can easily be
downcast to save memory with the downcast method and smart_fill_na fills na
data with appropriate defaults based upon the column datatypes (either provided
by the user via column_dict or through inference).

Dataframes stored in data can be imported and exported using the load and save
methods. Current data formats supported are csv, feather, and hdf5.

A Settings object needs to be passed when a Data instance is created. If
the quick_start option is selected, a Filer object must be passed as well.
"""
import csv
from datetime import timedelta
from dataclasses import dataclass
from functools import wraps
from inspect import getfullargspec
from more_itertools import unique_everseen
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype


@dataclass
class Data(object):
    """
    Primary class for storing and manipulating data used in machine learning
    projects.
    """
    settings : object
    filer : object = None
    df : object = None
    quick_start : bool = False
    default_df = str = 'df'
    column_dict : object = None

    def __post_init__(self):
        self.settings.localize(instance = self, sections = ['general'])
        if self.verbose:
            print('Building data container')
        """
        If quick_start is set to true and a settings dictionary is passed,
        data is automatically loaded according to user specifications in the
        settings file.
        """
        if self.quick_start:
            if self.filer:
                self.load(import_path = self.filer.import_path,
                          test_data = self.filer.test_data,
                          test_rows = self.filer.test_chunk,
                          encoding = self.filer.encoding)
            else:
                error = 'Data quick_start requires a Filer object'
                raise AttributeError(error)
        self.column_type_dicts = {bool : [],
                                  float : [],
                                  int : [],
                                  object : [],
                                  str : [],
                                  CategoricalDtype : [],
                                  list : [],
                                  np.datetime64 : [],
                                  timedelta : [],
                                  'interactor' : [],
                                  'scaler' : [],
                                  'encoder' : []}
        self.default_values = {bool : False,
                               float : 0.0,
                               int : 0,
                               object : '',
                               str : '',
                               CategoricalDtype : '',
                               list : [],
                               np.datetime64 : 1/1/1990,
                               timedelta : 0,
                               'interactor' : '',
                               'scaler' : 0,
                               'encoder' : ''}
        self.custom_columns = []
        self.dropped_columns = []
        if self.quick_start:
            self.auto_column_types()
        return self

    @property
    def bool_columns(self):
        return self.column_type_dicts[bool]

    @property
    def float_columns(self):
        return self.column_type_dicts[float]

    @property
    def int_columns(self):
        return self.column_type_dicts[int]

    @property
    def str_columns(self):
        return self.column_type_dicts[object]

    @property
    def category_columns(self):
        return self.column_type_dicts[CategoricalDtype]

    @property
    def list_columns(self):
        return self.column_type_dicts[list]

    @property
    def datetime_columns(self):
        return self.column_type_dicts[np.datetime64]

    @property
    def timedelta_columns(self):
        return self.column_type_dicts[timedelta]

    @property
    def numeric_columns(self):
        return self.float_columns + self.int_columns

    @property
    def interactor_columns(self):
        if not 'encoder' in self.column_dict:
            return self.category_columns
        else:
            return self.column_type_dicts['encoder']

    @property
    def encoder_columns(self):
        if not 'interactor' in self.column_dict:
            return self.category_columns
        else:
            return self.column_type_dicts['interactor']

    @property
    def scaler_columns(self):
        if not 'scaler' in self.column_dict:
            return self.numeric_columns
        else:
            self.column_type_dicts['scaler']

    @property
    def full(self):
        return self.x, self.y

    @property
    def train_test(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    @property
    def train_val(self):
        return self.x_train, self.y_train, self.x_val, self.y_val

    @property
    def train_test_val(self):
        return (self.x_train, self.y_train, self.x_test, self.y_test,
                self.x_val, self.y_val)

    @property
    def train(self):
        return self.x_train, self.y_train

    @property
    def test(self):
        return self.x_test, self.y_test

    @property
    def val(self):
        return self.x_val, self.y_val

    def check_df(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            unpassed_args = argspec.args[len(args):]
            if 'df' in argspec.args and 'df' in unpassed_args:
                kwargs.update({'df' : getattr(self, self.default_df)})
            return func(self, *args, **kwargs)
        return wrapper

    def __str__(self):
        return getattr(self, self.default_df)

    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            error = name + ' does not exist in Data instance'
            raise KeyError(error)
            return self

    def __setitem__(self, name, value):
        setattr(self, name, value)
        return self

    def __delitem__(self, name):
        delattr(self, name)
        return self

    @check_df
    def _check_columns(self, df = None, columns = None):
        if not columns:
            return df.columns
        else:
            return columns

    @check_df
    def _crosscheck_columns(self, df = None):
        for data_type, column_list in self.column_type_dicts.items():
            if column_list:
                for column in column_list:
                    if not column in df.columns:
                        self.column_type_dicts[data_type].remove(column)
        dict_keys = list(self.column_dict.keys())
        for column in dict_keys:
            if not column in df.columns:
                self.column_dict.pop(column)
                if column != self.label:
                    self.dropped_columns.append(column)
        return self

    def _remove_from_column_list(self, column_list, new_columns):
        column_list = [col for col in column_list if col not in new_columns]
        return column_list

    def _listify(self, variable):
        """
        Checks to see if the variable is currently a list type. If the variable
        is None, it is converted to a list with the string 'none'. If it is a
        string, it is converted to a list with that string. If the variable
        is already a list, it is returned unchanged.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def add_df_group(self, group_name, df_list):
        return self.df_options.update(
                {group_name : self.settings._listify(df_list)})

    @check_df
    def add_unique_index(self, df = None, column = 'index_universal',
                         make_index = False):
        df[column] = range(1, len(df.index) + 1)
        if make_index:
            df.set_index(column, inplace = True)
        return self

    @check_df
    def apply(self, df = None, func = None, **kwargs):
        """
        Allows users to pass a function to data which will be applied to the
        passed dataframe (or uses df if none is passed).
        """

        df = func(df, **kwargs)
        return self

    @check_df
    def auto_column_types(self, df = None):
        self.column_dict = {}
        for data_type, column_list in self.column_type_dicts.items():
            if not data_type in ['interactor', 'scaler', 'encoder']:
                columns = df.select_dtypes(
                        include = [data_type]).columns.to_list()
                if columns:
                    self.column_type_dicts[data_type].extend(columns)
                self.column_dict.update(dict.fromkeys(columns, data_type))
        return self

    @check_df
    def auto_categorize(self, df = None, columns = None, threshold = 10):
        """
        Automatically assesses each column to determine if it has less than
        threshold unique values and is not boolean. If so, that column is
        converted to categorical type.
        """
        columns = self._check_columns(df = df, columns = columns)
        for column in columns:
            if column in df.columns:
                if not column in self.column_type_dicts[bool]:
                    if df[column].nunique() < threshold:
                        df[column] = df[column].astype('category')
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def create_column_list(self, df = None, columns = None, prefixes = None):
        """
        Dynamically creates a new column list from a list of columns and/or
        lists of prefixes.
        """
        if prefixes:
            temp_list = []
            prefixes_list = []
            for prefix in prefixes:
                temp_list = [x for x in df if x.startswith(prefix)]
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
    def change_column_type(self, df = None, columns = None, prefixes = None,
                           data_type = str):
        columns = self.create_column_list(prefixes = prefixes,
                                          columns = columns)
        self.column_dict.update(dict.fromkeys(columns, data_type))
        for d_type, column_list in self.column_type_dicts.items():
            if d_type == data_type:
                self.column_type_dicts[data_type].extend(columns)
            else:
                self.column_type_dicts[d_type] = (
                        self._remove_from_column_list(column_list, columns))
        return self

    @check_df
    def convert_rare(self, df = None, columns = None, threshold = 0):
        """
        The method converts categories rarely appearing within categorical
        data columns to empty string if they appear below the passed threshold.
        Threshold is defined as the percentage of total rows.
        """
        columns = self._check_columns(df = df, columns = columns)
        for column in columns:
            if column in df.columns:
                default_value = self.default_values(CategoricalDtype)
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
    def decorrelate(self, df = None, threshold = 0.95):
        """
        Drops all but one column from highly correlated groups of columns.
        Threshold is based upon the .corr() method in pandas. columns can
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
        """
        Method to decrease memory usage by downcasting datatypes. For
        numerical datatypes, the method attempts to cast the data to unsigned
        integers if possible.
        """
        columns = self._check_columns(df = df, columns = columns)
        for column in columns:
            if column in df.columns:
                if self.column_dict[column] in [bool]:
                    df[column] = df[column].astype(bool)
                elif self.column_dict[column] in [int, float]:
                    try:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'integer')
                        if min(df[column] >= 0):
                            df[column] = pd.to_numeric(df[column],
                                                       downcast = 'unsigned')
                    except ValueError:
                        df[column] = pd.to_numeric(df[column],
                                                   downcast = 'float')
                elif self.column_dict[column] in [CategoricalDtype]:
                    df[column] = df[column].astype('category')
                elif self.column_dict[column] in [list]:
                    df[column].apply(self._listify,
                                     axis = 'columns',
                                     inplace = True)
                elif self.column_dict[column] in [np.datetime64]:
                    df[column] = pd.to_datetime(df[column])
                elif self.column_dict[column] in [timedelta]:
                    df[column] = pd.to_timedelta(df[column])
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def drop_columns(self, df = None, columns = None, prefixes = None):
        """
        Drops list of columns and columns with prefixes listed. In addition,
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
        """
        This method drops boolean columns that rarely have True. This differs
        from the sklearn VarianceThreshold class because it is only
        concerned with rare instances of True and not False. This enables
        users to set a different variance threshold for rarely appearing
        information. Threshold is defined as the percentage of total rows (and
        not the typical variance formulas).
        """
        cols = []
        columns = self._check_columns(df = df, columns = columns)
        for column in columns:
            if column in df.columns:
                if df[column].mean() < threshold:
                    df.drop(column, axis = 'columns', inplace = True)
                    cols.append(column)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        self.drop_columns(columns = cols)
        return self

    def initialize_series(self, df = None):
        row = pd.Series(index = self.column_dict.keys())
        for column, data_type in self.column_dict.items():
            row[column] = self.default_values[data_type]
        if not df:
            setattr(self, self.default_df, row)
            return self
        else:
            return row

    def load(self, import_folder = '', file_name = 'data', import_path = '',
             file_type = 'csv', usecolumns = None, index = False,
             encoding = 'windows-1252', test_data = False, test_rows = 500,
             return_df = False, message = 'Importing data'):
        """
        Imports pandas dataframes from different file formats.
        """
        if not import_path:
            if not import_folder:
                import_folder = self.filer.import_folder
            import_path = self.filer.make_path(folder = import_folder,
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
        """
        A simple wrapper method for pandas wide_to_long method using more
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
        """
        A simple wrapper method for pandas pivot method named as corresponding
        method to reshape_long.
        """
        df = (df.pivot(index = df_index,
                       columns = columns,
                       values = values).reset_index())
        return self

    @check_df
    def smart_fillna(self, df = None, columns = None):
        """
        Fills na values in dataframe to defaults based upon the datatype listed
        in the columns dictionary. If the dictionary of datatypes does not
        exist, the method fills columns based upon the current datatype
        inferred by pandas. Because their is no good default category, the
        method uses an empty string ('').
        """
        columns = self._check_columns(df = df, columns = columns)
        for column in columns:
            if column in df.columns:
                default_value = self.default_values[self.column_dict[column]]
                df[column].fillna(default_value, inplace = True)
            else:
                error = column + ' is not in DataFrame'
                raise KeyError(error)
        return self

    @check_df
    def split_xy(self, df = None, label = 'label'):
        """
        Splits data into x and y based upon the label passed.
        """
        self.x = df.drop(label, axis = 'columns')
        self.y = df[label]
        self._crosscheck_columns(df = self.x)
        return self

    @check_df
    def summarize(self, df = None, export_path = '', export_summary = True,
                  transpose = False):
        """
        Creates a dataframe of common summary data. It is more inclusive than
        describe() and includes boolean and numerical columns by default.
        If an export_path is passed, the summary table is automatically saved
        to disc.
        """
        summary_columns = ['variable', 'data_type', 'count', 'min', 'q1',
                           'median', 'q3', 'max', 'mad', 'mean', 'stan_dev',
                           'mode', 'sum']
        self.summary = pd.DataFrame(columns = summary_columns)
        for i, col in enumerate(df.columns):
            new_row = pd.Series(index = summary_columns)
            new_row['variable'] = col
            new_row['data_type'] = df[col].dtype
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
        if transpose:
            self.summary = self.summary.transpose()
            df_header = False
            df_index = True
        else:
            df_header = True
            df_index = False
        if export_summary:
            if not export_path:
                export_path = os.path.join(self.filer.results,
                                           'data_summary.csv')
            self.save(df = self.summary,
                      index = df_index,
                      header = df_header,
                      export_path = export_path,
                      file_type = self.filer.results_format,
                      message = 'Exporting summary data')
        return self

    @check_df
    def save(self, df = None, export_folder = '', file_name = 'data',
             export_path = '', file_type = 'csv', index = False, header = True,
             encoding = 'windows-1252', float_format = '%.4f',
             boolean_out = True, message = 'Exporting data'):
        """
        Exports pandas dataframes to different file formats an encoding of
        boolean variables as True/False or 1/0.
        """
        if not export_path:
            if not export_folder:
                export_folder = self.filer.data
            export_path = self.filer.make_path(folder = export_folder,
                                               file_name = file_name,
                                               file_type = file_type)
        if not isinstance(df, pd.DataFrame):
            df = getattr(self, self.default_df)
        if self.verbose:
            print(message)
        if not boolean_out:
            df.replace({True : 1, False : 0}, inplace = True)
        if file_type == 'csv':
            df.to_csv(export_path,
                      encoding = encoding,
                      index = index,
                      header = header,
                      float_format = float_format)
        elif file_type == 'h5':
            df.to_hdf(export_path)
        elif file_type == 'feather':
            if isinstance(df, pd.DataFrame):
                df.reset_index(inplace = True)
                df.to_feather(export_path)
        return

    def save_drops(self, file_name = 'dropped_columns', export_path = ''):
        """
        Saves dropped_columns into a .csv file.
        """
        self.dropped_columns = list(unique_everseen(self.dropped_columns))
        if not export_path:
            export_folder = self.filer.results
            export_path = self.filer.make_path(folder = export_folder,
                                               file_name = file_name,
                                               file_type = 'csv')
        if self.dropped_columns:
            if self.verbose:
                print('Exporting dropped feature list')
            with open(export_path, 'wb') as export_file:
                csv_writer = csv.writer(export_file)
                csv_writer.writerow(self.dropped_columns)
        return