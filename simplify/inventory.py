
import csv
import datetime
from functools import wraps
from inspect import getfullargspec
import os

from dataclasses import dataclass
import pandas as pd
import pickle

from .cookbook.countertop import Countertop


@dataclass
class Inventory(Countertop):
    """Creates and stores dynamic and static file paths for the siMpLify
    package.
    """
    menu : object
    root_folder : str = '..'
    data_folder : str = 'data'
    results_folder : str = 'results'
    import_file : str = 'data'
    export_file : str = 'data'
    import_format : str = 'csv'
    export_format : str = 'csv'
    stage : str = 'cookbook'
    use_defaults : bool = True

    def __post_init__(self):
        """Localizes 'files' settings as attributes and sets paths and folders.
        """
        self.menu.localize(instance = self, sections = ['files', 'general'])
        self.stages_dict = {'cultivate' : ['_raw', '_interim'],
                            'reap' : ['_interim', '_interim'],
                            'thresh' : ['_interim', '_interim'],
                            'bale' : ['_interim', '_interim'],
                            'clean' : ['_interim', '_processed'],
                            'cookbook' : ['_processed', '_processed']}
        self._set_folders()
        self._set_io_paths()
        return self

    @property
    def data_in(self):
        return getattr(self, self.stages_dict[self.stage][0])

    @property
    def data_out(self):
        return getattr(self, self.stages_dict[self.stage][1])

    @property
    def experiment(self):
        return self._experiment

    @property
    def external(self):
        return self._external

    @property
    def recipe(self):
        return self._recipe

    @property
    def results(self):
        return self._results

    def check_kwargs(func):

        """Decorator which replaces None in kwargs with default values."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            argspec = getfullargspec(func)
            unpassed_args = argspec.args[len(args):]
            if 'df' in argspec.args and 'df' in unpassed_args:
                kwargs.update({'df' : getattr(self, self.default_df)})
            return func(self, *args, **kwargs)
        return wrapper

    def _check_boolean_out(self, boolean_out):
        if boolean_out or boolean_out == False:
            return boolean_out
        else:
            return self.boolean_out

    def _check_encoding(self, encoding):
        if encoding:
            return encoding
        else:
            return self.encoding

    def _check_file_type(self, file_type, io_status):
        if file_type:
            return file_type
        elif io_status == 'import':
            return self.import_format
        elif io_status == 'export':
            return self.export_format
        else:
            return 'csv'

    def _check_float(self, float_format):
        if float_format:
            return float_format
        else:
            return self.float_format

    def _check_test_data(self, test_data, test_rows):
        if test_data == False:
            return None
        elif not test_data and not self.test_data:
            return None
        else:
            if not test_rows:
                test_rows = self.test_rows
            return test_rows

    def _file_name(self, prefix = '', file_name = '', suffix = '',
                   file_type = ''):
        """Creates file name with prefix, suffix, and file extension."""
        extensions = {'csv' : '.csv',
                      'pickle' : '.pkl',
                      'feather' : '.ftr',
                      'h5' : '.hdf',
                      'excel' : '.xlsx',
                      'text' : '.txt',
                      'xml' : '.xml',
                      'png' : '.png'}
        if file_type in extensions:
            return prefix + file_name + suffix + extensions[file_type]
        else:
            return prefix + file_name + suffix + '.' + file_type

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist."""
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _set_folders(self):
        """Creates data and results folders based upon passed parameters."""
        self._data = os.path.join(self.root_folder, self.data_folder)
        self._raw = os.path.join(self._data, 'raw')
        self._interim = os.path.join(self._data, 'interim')
        self._processed = os.path.join(self._data, 'processed')
        self._external = os.path.join(self._data, 'external')
        self._results = os.path.join(self.root_folder, self.results_folder)
        subfolder = ('experiment_'
                     + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self._experiment = os.path.join(self.results, subfolder)
        self._make_folder(self._raw)
        self._make_folder(self._interim)
        self._make_folder(self._processed)
        self._make_folder(self._external)
        self._make_folder(self._results)
        self._make_folder(self._experiment)
        return self

    def _set_io_paths(self):
        """Creates a single import and export path from passed parameters."""
        self.import_path = self.create_path(folder = self.data_in,
                                             file_name = self.import_file,
                                             file_type = self.import_format)
        self.export_path = self.create_path(folder = self.data_out,
                                             file_name = self.export_file,
                                             file_type = self.export_format)
        return self

    def _set_recipe_path(self, recipe):
        """Creates file or folder path for recipe-specific exports."""
        if recipe.cleaver.technique != 'none':
            subfolder = ('recipe_'
                         + recipe.model.technique + '_'
                         + recipe.cleaver.technique
                         + str(recipe.number))
        else:
            subfolder = ('recipe_'
                         + recipe.model.technique
                         + str(recipe.number))
        self._recipe = self.create_path(folder = self.experiment,
                                         subfolder = subfolder)
        return self

    def create_path(self, folder = '', subfolder = '', prefix = '',
                    file_name = '', suffix = '', file_type = 'csv'):
        """Creates file and/or folder path."""
        if subfolder:
            folder = os.path.join(folder, subfolder)
        self._make_folder(folder)
        if file_name:
            file_name = self._file_name(prefix = prefix,
                                        file_name = file_name,
                                        suffix = suffix,
                                        file_type = file_type)
            return os.path.join(folder, file_name)
        else:
            return folder

    def initialize_series_writer(self, folder, file_name, file_path,
                                 encoding, column_list, dialect = 'excel'):
        """Initializes writer object for line-by-line saving to a .csv file.
        """
        file_path = self._check_path(folder, file_name, file_path, 'csv')
        with open(file_path, mode = 'w', newline = '',
                  encoding = encoding) as self.output_series:
                self.writer = csv.writer(self.output_series, dialect = dialect)
                self.writer.writerow(column_list)
        return self

    def load_csv(self, file_path, df_index = None, df_header = None,
                 encoding = None, usecolumns = None, nrows = None):
        variable = pd.read_csv(file_path,
                               encoding = encoding,
                               index_col = df_index,
                               header = df_header,
                               usecols = usecolumns,
                               nrows = nrows,
                               low_memory = False)
        return variable

    def load_feather(self, file_path):
        variable = pd.read_feather(file_path, nthreads = -1)
        return variable

    def load_hdf(self, file_path, encoding  = None, usecolumns = None,
                 nrows = None):
        variable = pd.read_hdf(file_path,
                               chunksize = nrows,
                               columns = usecolumns)
        return variable

    def load_json(self, file_path, encoding  = None, usecolumns  = None,
                  nrows  = None):
        variable = pd.read_json(file_path = file_path,
                                encoding = encoding,
                                chunksize = nrows,
                                columns = usecolumns)
        return variable

    def pickle_object(self, variable, file_path):
        pickle.dump(variable, open(file_path, 'wb'))
        return self

    def save_df(self, variable, file_path, file_type = None, df_index = False,
                df_header = True, encoding = None, float_format = None,
                boolean_out = None):
        boolean_out = self._check_boolean_out(boolean_out)
        encoding = self._check_encoding(encoding)
        float_format = self._check_float(float_format)
        file_type = self._check_file_type(file_type, 'import')
        if boolean_out:
            variable.replace({True : 1, False : 0}, inplace = True)
        if file_type == 'csv':
            variable.to_csv(file_path,
                            encoding = encoding,
                            index = df_index,
                            header = df_header,
                            float_format = float_format)
        elif file_type == 'h5':
            variable.to_hdf(file_path)
        elif file_type == 'feather':
            variable.reset_index(inplace = True)
            variable.to_feather(file_path)
        return self

    def save_series(self, variable, file_path, boolean_out = None):
        boolean_out = self._check_boolean_out(boolean_out)
        if boolean_out:
            variable.replace({True : 1, False : 0}, inplace = True)
        self.writer.writerow(variable)
        return self

    def unpickle_object(self, file_path, file_type = None, df_index = None,
                        df_header = None, encoding = None):
        variable = pickle.load(open(file_path, 'rb'))
        return variable