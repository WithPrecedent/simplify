
import csv
import datetime
import os
from dataclasses import dataclass
import pickle

import pandas as pd

from .implements.implement import Implement


@dataclass
class Inventory(Implement):
    """Creates and stores dynamic and static file paths for the siMpLify
    package.
    """
    menu : object
    root : str = '..'
    data : str = 'data'
    results : str = 'results'
    import_file : str = 'data'
    export_file : str = 'data'
    import_format : str = 'csv'
    export_format : str = 'csv'
    stage : str = 'processed'
    use_defaults : bool = True

    def __post_init__(self):
        """Localizes 'files' settings as attributes and sets paths and folders.
        """
        self.menu.localize(instance = self, sections = ['files', 'general'])
        self.load_datatypes = {'csv' : self._load_csv,
                               'feather' : self._load_feather,
                               'hdf' : self._load_hdf,
                               'json' : self._load_json,
                               'pickle' : self._unpickle_object,
                               'pkl' : self._unpickle_object}
        self.save_datatypes = {pd.DataFrame : self._save_df,
                               pd.Series : self._save_series,
                               object : self._pickle_object}
        self.next_stages = {'raw' : 'interim',
                            'interim' : 'processed',
                            'processed' : 'processed'}
        self._set_folders()
        self._set_io_paths()
        return self

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

    def _check_float(self, float_format):
        if float_format:
            return float_format
        else:
            return self.float_format

    def _check_path(self, folder, file_name, file_path, file_type):
        if not file_path:
            if not folder:
                folder = self.data
            new_path = self._create_path(folder = folder,
                                        file_name = file_name,
                                        file_type = file_type)
            return new_path
        else:
            return file_path

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

    def _load_csv(self, file_path, df_index, df_header, encoding, usecolumns,
                  nrows):
        variable = pd.read_csv(file_path,
                               encoding = encoding,
                               index_col = df_index,
                               header = df_header,
                               usecols = usecolumns,
                               nrows = nrows,
                               low_memory = False)
        return variable

    def _load_feather(self, file_path, df_index, df_header, encoding,
                      usecolumns, nrows):
        variable = pd.read_feather(file_path, nthreads = -1)
        return variable

    def _load_hdf(self, file_path, df_index, df_header, encoding, usecolumns,
                  nrows):
        variable = pd.read_hdf(file_path,
                               chunksize = nrows,
                               columns = usecolumns)
        return variable

    def _load_json(self, file_path, df_index, df_header, encoding, usecolumns,
                   nrows):
        variable = pd.read_json(file_path = file_path,
                                encoding = encoding,
                                chunksize = nrows,
                                columns = usecolumns)
        return variable

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist."""
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _pickle_object(self, variable, file_path, file_type, df_index,
                       df_header, encoding, float_format, boolean_out):
        pickle.dump(variable, open(file_path, 'wb'))
        return self

    def _recipe_path(self, model, recipe_number, cleave = '', file_name = '',
                     file_type = ''):
        """Creates file or folder path for recipe-specific exports."""
        if cleave:
            subfolder = ('recipe_'
                         + model.technique + '_'
                         + cleave.technique
                         + str(recipe_number))
        else:
            subfolder = ('recipe_'
                         + model.technique
                         + str(recipe_number))
        self._make_folder(folder = self._create_path(folder = self.recipes,
                                                    subfolder = subfolder))
        return self._create_path(folder = self.recipes,
                                subfolder = subfolder,
                                file_name = file_name,
                                file_type = file_type)

    def _save_df(self, variable, file_path, file_type, df_index, df_header,
                 encoding, float_format, boolean_out):
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

    def _save_series(self, variable, file_path, file_type, df_index, df_header,
                     encoding, float_format, boolean_out):
        if boolean_out:
            variable.replace({True : 1, False : 0}, inplace = True)
        self.writer.writerow(variable)
        return self

    def _set_folders(self):
        """Creates data and results folders based upon passed parameters."""
        self.data = os.path.join(self.root, self.data)
        self.data_raw = os.path.join(self.data, 'raw')
        self.data_interim = os.path.join(self.data, 'interim')
        self.data_processed = os.path.join(self.data, 'processed')
        self.data_external = os.path.join(self.data, 'external')
        self.results = os.path.join(self.root, self.results)
        subfolder = ('experiment_'
                     + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self.results = os.path.join(self.root, self.results, subfolder)
        self._make_folder(self.data_raw)
        self._make_folder(self.data_interim)
        self._make_folder(self.data_processed)
        self._make_folder(self.data_external)
        self._make_folder(self.results)
        return self

    def _set_io_paths(self):
        """Creates a single import and export path from passed parameters."""
        self._set_next_stage()
        self.import_path = self._create_path(folder = self.data,
                                            subfolder = self.stage,
                                            file_name = self.import_file,
                                            file_type = self.import_format)
        self.export_path = self._create_path(folder = self.data,
                                            subfolder = self.next_stage,
                                            file_name = self.export_file,
                                            file_type = self.export_format)
        return self

    def _set_next_stage(self):
        self.next_stage = self.next_stages[self.stage]
        return self

    def _unpickle_object(self, file_path, file_type, df_index, df_header,
                         encoding):
        variable = pickle.load(open(file_path, 'rb'))
        return variable

    def _create_path(self, folder = '', subfolder = '', prefix = '',
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

    def load(self, folder = None, file_name = 'unknown', file_path = None,
             file_type = 'csv', df_index = False, df_header = True,
             encoding = None, usecolumns = None, test_data = False,
             test_rows = 500, message = None):
        """Imports siMpLify objects from different file formats based upon
        user settings and arguments passed."""
        file_path = self._check_path(folder, file_name, file_path, file_type)
        encoding = self._check_encoding(encoding)
        nrows = self._check_test_data(test_data, test_rows)
        if self.verbose and message:
            print(message)
        method = self.load_datatypes[file_type]
        variable = method(file_path, file_type, df_index, df_header, encoding,
                          usecolumns, nrows)
        return variable

    def save(self, variable, folder = None, file_name = 'unknown',
             file_path = None, file_type = 'csv', df_index = False,
             df_header = True, encoding = None, float_format = None,
             boolean_out = None, message = None):
        """Exports siMpLify objects based upon datatype using user settings
        and arguments passed.
        """
        file_path = self._check_path(folder, file_name, file_path, file_type)
        boolean_out = self._check_boolean_out(boolean_out)
        encoding = self._check_encoding(encoding)
        float_format = self._check_float(float_format)
        if self.verbose and message:
            print(message)
        for datatype, method in self.save_datatypes.items():
            if isinstance(variable, datatype):
                method(variable, file_path, file_type, df_index, df_header,
                       encoding, float_format, boolean_out)
        return self