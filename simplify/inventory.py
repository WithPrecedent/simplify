
import csv
import datetime
import glob
import os

from dataclasses import dataclass
import pandas as pd
import pickle

from .implements import listify

@dataclass
class Inventory(object):
    """Creates and stores dynamic and static file paths, loads and saves
    various file types, and properly formats files for import and export.

    Attributes:
        menu: an instance of Menu
        root_folder: a string including the complete path from which the other
            paths and folders used by Inventory stem.
        data_folder: a string containing the data folder name.
        results_folder: a string containing the results folder name.
        import_format, export_format: strings (without leading periods) listing
            the default file types to use for loading and saving data files.
        stage: a string containing the name of the stage or step in the
            siMpLify package.
        datetime_naming: a boolean value setting whether the date and time
            should be used to create experiment subfolders (so that prior
            results are not overwritten).
        auto_prepare: boolean value as to whether prepare method should be
            called when the class is instanced.
    """
    menu : object
    root_folder : str = '..'
    data_folder : str = 'data'
    results_folder : str = 'results'
    import_format : str = 'csv'
    export_format : str = 'csv'
    stage : str = 'cook'
    datetime_naming : bool = True
    auto_prepare : bool = True

    def __post_init__(self):
        """Localizes 'files' settings as attributes and sets paths and folders.
        """
        self.menu.localize(instance = self, sections = ['general', 'files'])
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    @property
    def bundlers(self):
        """Returns external data folder where bundlers data files are located.
        """
        return self._bundlers

    @property
    def cleaners(self):
        """Returns folder containing cleaner .csv files."""
        return self._cleaners

    @property
    def data_in(self):
        """Returns data import folder."""
        if self._data_input_folder:
            return self._data_input_folder
        else:
            return getattr(self, self.stages_data[self.stage][0])

    @property
    def data_out(self):
        """Returns data export folder."""
        if self._data_output_folder:
            return self._data_output_folder
        else:
            return getattr(self, self.stages_data[self.stage][1])

    @property
    def experiment(self):
        """Returns active experiment folder."""
        return self._experiment

    @property
    def reapers(self):
        """Returns folder containing reaper .csv files."""
        return self._reapers

    @property
    def results(self):
        """Returns results folder."""
        return self._results

    def _check_boolean_out(self, boolean_out):
        """Checks value of local boolean_out variable. If not supplied, the
        default from the Menu instance is used.
        """
        if boolean_out or boolean_out == False:
            return boolean_out
        else:
            return self.boolean_out

    def _check_encoding(self, encoding):
        """Checks value of local encoding variable. If not supplied, the
        default from the Menu instance is used.
        """
        if encoding:
            return encoding
        else:
            return self.file_encoding

    def _check_file_type(self, file_type, io_status):
        """Checks value of local file_type variable. If not supplied, the
        default from the Menu instance is used based upon whether import or
        export methods are being used.
        """
        if file_type:
            return self.extensions[file_type]
        elif io_status == 'import':
            return self.extensions[self.import_format]
        elif io_status == 'export':
            return self.extensions[self.export_format]
        else:
            return '.csv'

    def _check_float(self, float_format):
        """Checks value of local float_format variable. If not supplied, the
        default from the Menu instance is used.
        """
        if float_format:
            return float_format
        else:
            return self.float_format

    def _check_test_data(self, test_data, test_rows):
        """Checks value of local test_data variable. If not supplied, the
        default from the Menu instance is used. If not selected, 'None' is
        returned because that prevents the parameter from being activated
        using pandas methods.
        """
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
        """Creates file name with prefix, suffix, and file extension. A
        period need not precede the file_type because the self.extensions
        dictionary will look for an appropriate extension name."""
        if file_type in self.extensions:
            return prefix + file_name + suffix + self.extensions[file_type]
        else:
            return prefix + file_name + suffix + '.' + file_type

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist."""
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _set_defaults(self):
        """Creates data, results, and experiment folders based upon passed
        parameters. The experiment folder name is based upon the date and time
        to avoid overwriting previous experiments unless datetime_naming is set
        to False. If False, a default folder named 'experiment' will be used.
        Also, creates a dictionary for file_type names and extensions.
        """
        self.stages_data = {'sow' : ['_raw', '_interim'],
                            'harvest' : ['_interim', '_interim'],
                            'clean' : ['_interim', '_interim'],
                            'bundle' : ['_interim', '_interim'],
                            'deliver' : ['_interim', '_processed'],
                            'cook' : ['_processed', '_processed']}
        self.extensions = {'csv' : '.csv',
                           'pickle' : '.pkl',
                           'feather' : '.ftr',
                           'h5' : '.hdf',
                           'hdf' : '.hdf',
                           'excel' : '.xlsx',
                           'text' : '.txt',
                           'xml' : '.xml',
                           'png' : '.png'}
        self._data = os.path.join(self.root_folder, self.data_folder)
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        self.harvest_subfolders = ['organizers', 'keywords', 'cleaners',
                                   'combiners']
        self._results = os.path.join(self.root_folder, self.results_folder)
        if self.datetime_naming:
            subfolder = ('experiment_'
                         + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        else:
            subfolder = 'experiment'
        self._experiment = os.path.join(self.results, subfolder)
        self._make_folder(self._data)
        self._make_folder(self._results)
        self._make_folder(self._experiment)
        self._data_input_folder = ''
        self._data_output_folder = ''
        return self

    def _set_plan_folder(self, plan, steps_to_use = None):
        """Creates file or folder path for plan-specific exports.

        Parameters:
            plan: an instance of Plan (or a subclass) for which files are to be
                saved.
            steps to use: a list of strings or single string containing names
                of steps from which the folder name should be created.
        """
        if steps_to_use:
            subfolder = plan.name + '_'
            for step in listify(steps_to_use):
                subfolder += getattr(plan, step).technique + '_'
            subfolder += str(plan.number)
        self.plan = os.path.join(self.experiment, subfolder)
        return self

    def add_data_subfolder(self, subfolder, io_status):
        """Adds subfolder to existing data input or output folder and that
        becomes the new data input or output folder.
        """
        if io_status == 'import':
            self._data_input_folder = os.path.join(self.data_in, subfolder)
        elif io_status == 'export':
            self._data_output_folder = os.path.join(self.data_out, subfolder)
        else:
            error = 'io_status must be "import" or "export"'
            raise KeyError(error)
        return self

    def add_file_extension(self, name, file_extension):
        """Adds or replaces a file extension option."""
        self.extensions.update({name : file_extension})
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

    def implement_glob(self, file_type = None, recursive = True):
        """Creates a list of paths in the self.data_in folder based upon
        file_type. If recursive is True, subfolders are searched as well.
        """
        file_type = self._check_file_type(file_type = file_type,
                                          io_status = 'import')
        self.globbed_paths = glob.glob(os.path.join(self.data_in, '**',
                                                    file_type),
                                       recursive = recursive)
        return self

    def initialize_series_writer(self, file_name, column_list, encoding = None,
                                 dialect = 'excel'):
        """Initializes writer object for line-by-line saving to a .csv file.
        """
        if not column_list:
            error = 'initialize_series_writer requires column_list as list'
            raise TypeError(error)
        encoding = self._check_encoding()
        file_path = os.path.join(self.data_out, file_name)
        with open(file_path, mode = 'w', newline = '',
                  encoding = encoding) as self.output_series:
                self.writer = csv.writer(self.output_series, dialect = dialect)
                self.writer.writerow(column_list)
        return self

    def load_csv(self, file_path, df_index = None, df_header = None,
                 encoding = None, usecolumns = None, nrows = None):
        """Loads csv file into pandas dataframe."""
        encoding = self._check_encoding()
        variable = pd.read_csv(file_path,
                               encoding = encoding,
                               index_col = df_index,
                               header = df_header,
                               usecols = usecolumns,
                               nrows = nrows,
                               low_memory = False)
        return variable

    def load_feather(self, file_path):
        """Loads feather file into pandas dataframe."""
        variable = pd.read_feather(file_path, nthreads = -1)
        return variable

    def load_hdf(self, file_path, usecolumns = None, nrows = None):
        """Loads hdf5 file into pandas dataframe."""
        variable = pd.read_hdf(file_path,
                               chunksize = nrows,
                               columns = usecolumns)
        return variable

    def load_json(self, file_path, encoding  = None, usecolumns  = None,
                  nrows  = None):
        """Loads json file into pandas dataframe."""
        encoding = self._check_encoding()
        variable = pd.read_json(file_path = file_path,
                                encoding = encoding,
                                chunksize = nrows,
                                columns = usecolumns)
        return variable

    def pickle_object(self, variable, file_path):
        """Pickles python object."""
        pickle.dump(variable, open(file_path, 'wb'))
        return self

    def prepare(self):
        """Sets data subfolders - defaults mirror cookiecutter names."""
        for folder in self.data_subfolders:
            setattr(self, '_' + folder, os.path.join(self._data, folder))
            self._make_folder(getattr(self, '_' + folder))
        for folder in self.harvest_subfolders:
            setattr(self, '_' + folder, os.path.join(self._external, folder))
            self._make_folder(getattr(self, '_' + folder))
        return self

    def save_df(self, variable, file_path, file_type = None, df_index = False,
                df_header = True, encoding = None, float_format = None,
                boolean_out = None):
        """Saves pandas dataframe to different file formats based upon
        file_type, or if not provided, default file_type from Menu instance.
        """
        boolean_out = self._check_boolean_out(boolean_out)
        encoding = self._check_encoding(encoding)
        float_format = self._check_float(float_format)
        file_type = self._check_file_type(file_type, 'export')
        if not boolean_out:
            variable.replace({True : 1, False : 0}, inplace = True)
        if file_type == '.csv':
            variable.to_csv(file_path,
                            encoding = encoding,
                            index = df_index,
                            header = df_header,
                            float_format = float_format)
        elif file_type == '.hdf':
            variable.to_hdf(file_path)
        elif file_type == '.ftr':
            variable.reset_index(inplace = True)
            variable.to_feather(file_path)
        return self

    def save_series(self, variable, boolean_out = None):
        """Saves a pandas series as a single row in a csv file.
        """
        boolean_out = self._check_boolean_out(boolean_out)
        if boolean_out:
            variable.replace({True : 1, False : 0}, inplace = True)
        self.writer.writerow(variable)
        return self

    def unpickle_object(self, file_path):
        """Returns an unpickled python object."""
        variable = pickle.load(open(file_path, 'rb'))
        return variable