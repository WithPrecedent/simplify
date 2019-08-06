
import csv
import glob
import os

from dataclasses import dataclass
import pandas as pd
import pickle

from .tools import listify


@dataclass
class Inventory(object):
    """Creates and stores dynamic and static file paths, loads and saves
    various file types, and properly formats files for import and export.

    Attributes:
        menu: an instance of Menu.
        root_folder: a string including the complete path from which the other
            paths and folders used by Inventory stem.
        data_folder: a string containing the data folder name.
        results_folder: a string containing the results folder name.
        import_format, export_format: strings (without leading periods) listing
            the default file types to use for loading and saving data files.
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
    def data_in(self):
        if not hasattr(self, 'step'):
            self.step = 'cook'
        return self.import_folders[self.step]

    @data_in.setter
    def data_in(self, step, data_folder):
        self.import_files.update({step, data_folder})
        return self

    @property
    def data_out(self):
        if not hasattr(self, 'step'):
            self.step = 'cook'
        return self.export_folders[self.step]

    @data_out.setter
    def data_out(self, step, data_folder):
        self.data_out = data_folder
        return self

    def _check_boolean_out(self, variable):
        """Either leaves boolean values as True/False or changes values to 1/0
        based on user settings.

        Parameters:
            variable: pandas DataFrame or Series with boolean output values.
        """
        # checks whether True/False should be exported in data files. If
        # self.boolean_out is set to false, 1/0 are used instead.
        if hasattr(self, 'boolean_out') and self.boolean_out == False:
            variable.replace({True : 1, False : 0}, inplace = True)
        return variable

    def _check_encoding(self, encoding = None):
        """Checks value of local encoding variable. If not supplied, the
        method checks for a local variable. If neither option exists,
        the default value of 'windows-1252' is returned.

        Parameters:
            encoding: str variable containing file encoding technique that will
                be used for file importing.
        """
        if encoding:
            return encoding
        elif hasattr(self, 'file_encoding'):
            return self.file_encoding
        else:
            return 'windows-1252'

    def _check_file_name(self, file_name, io_status):
        if file_name:
            return file_name
        elif io_status == 'import':
            return self.import_files[self.step]
        elif io_status == 'export':
            return self.export_files[self.step]

    def _check_file_type(self, file_type = None, io_status = None):
        """Checks value of local file_type variable. If not supplied, the
        default from the Menu instance is used based upon whether import or
        export methods are being used. If the Menu options don't exist,
        '.csv' is returned.

        Parameters:
            file_type: string matching one of the supported file types in
                self.extensions.
            io_status: either 'import' or 'export' based upon whether the user
                is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.
        """
        if file_type:
            return file_type
        elif io_status == 'import':
            return self.import_format
        elif io_status == 'export':
            return self.export_format
        else:
            return 'csv'

    def _check_float_format(self, float_format = None):
        """Checks value of local float_format variable. If not supplied, the
        method checks for a local variable. If neither option exists,
        the default value of '%.4f' is returned.

        Parameters:
            float_format: the desired format for exporting float numbers.
        """
        if float_format:
            return float_format
        elif hasattr(self, 'float_format'):
            return self.float_format
        else:
            return '%.4f'

    def _check_kwargs(self, variables_to_check, passed_kwargs):
        new_kwargs = passed_kwargs
        for variable in variables_to_check:
            if not variable in passed_kwargs:
                if variable in self.default_kwargs:
                    new_kwargs.update(
                            {variable : self.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable : getattr(self, variable)})
        return new_kwargs

    def _check_test_data(self, test_data = False, test_rows = None):
        """Checks value of local test_data variable. If not supplied, the
        default from the Menu instance is used. If not selected, 'None' is
        returned because that prevents the parameter from being activated
        using pandas methods.

        Parameters:
            test_data: boolean variable indicating whether a test sample should
                be used. If set to False, the full dataset is imported.
            test_rows: an integer containing the size of the test sample.
        """
        if test_data == False:
            return None
        elif not test_data and not self.test_data:
            return None
        else:
            if not test_rows:
                test_rows = self.test_rows
            return test_rows

    def _create_file_path(self, folder = None, file_name = None,
                        file_path = None, file_type = None,
                        io_status = 'import'):
        if file_path:
            return file_path
        else:
            file_name = self._check_file_name(file_name = file_name,
                                              io_status = io_status)
            if folder and hasattr(self, folder):
                return (os.path.join(getattr(self, folder), file_name)
                        + self.extensions[file_type])
            else:
                 return (os.path.join(folder, file_name)
                         + self.extensions[file_type])

    def _load_csv(self, file_path, **kwargs):
        """Loads csv file into pandas dataframe."""
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows' : self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _load_feather(self, file_path, **kwargs):
        """Loads feather file into pandas dataframe."""
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _load_hdf(self, file_path, **kwargs):
        """Loads hdf5 file into pandas dataframe."""
        additional_kwargs = ['columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop['usecols']
        return pd.read_hdf(file_path, **kwargs)

    def _load_json(self, file_path, **kwargs):
        """Loads json file into pandas dataframe."""
        additional_kwargs = ['encoding', 'columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop['usecols']
        return pd.read_json(file_path = file_path, **kwargs)

    def _load_pickle(self, file_path, **kwargs):
        """Returns an unpickled python object."""
        return pickle.load(open(file_path, 'rb'))

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist.

        Parameters:
            folder: a string containing the path of the folder.
        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _save_csv(self, variable, file_path, **kwargs):
        """Saves pandas dataframe to different file formats based upon
        file_type, or if not provided, default file_type from Menu instance.
        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_feather(self, variable, file_path, **kwargs):
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _save_hdf(self, variable, file_path, **kwargs):
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_json(self, variable, file_path, **kwargs):
        variable.to_json(file_path, **kwargs)
        return

    def _save_pickle(self, variable, file_path, **kwargs):
        """Pickles python object."""
        pickle.dump(variable, open(file_path, 'wb'))
        return

    def _set_defaults(self):
        """Creates data, results, and experiment folders based upon passed
        parameters. The experiment folder name is based upon the date and time
        to avoid overwriting previous experiments unless datetime_naming is set
        to False. If False, a default folder named 'experiment' will be used.
        Also, creates a dictionary for file_type names and extensions.
        """
        self.extensions = {'csv' : '.csv',
                           'pickle' : '.pkl',
                           'feather' : '.ftr',
                           'h5' : '.hdf',
                           'hdf' : '.hdf',
                           'excel' : '.xlsx',
                           'text' : '.txt',
                           'xml' : '.xml',
                           'png' : '.png'}
        self.folder_groups = {'data' : ['raw', 'interim', 'processed',
                                        'external'],
                              'external' : ['organizers', 'keywords',
                                            'parsers', 'combiners']}
        self.default_kwargs = {'index' : False,
                               'header' : None,
                               'low_memory' : False,
                               'dialect' : 'excel',
                               'usecols' : None,
                               'columns' : None,
                               'nrows' : None,
                               'index_col' : False}
        self.import_folders = {'sow' : 'raw',
                               'harvest' : 'raw',
                               'clean' : 'interim',
                               'bundle' : 'interim',
                               'deliver' : 'interim',
                               'cook' : 'processed'}
        self.export_folders = {'sow' : 'raw',
                               'harvest' : 'interim',
                               'clean' : 'interim',
                               'bundle' : 'interim',
                               'deliver' : 'processed',
                               'cook' : 'processed'}
        self.import_files = {'sow' : 'glob',
                             'harvest' : 'glob',
                             'clean' : 'harvested_data',
                             'bundle' : 'cleaned_data',
                             'deliver' : 'bundled_data',
                             'cook' : 'final_data'}
        self.export_files = {'sow' : 'glob',
                             'harvest' : 'harvested_data',
                             'clean' : 'cleaned_data',
                             'bundle' : 'bundled_data',
                             'deliver' : 'final_data',
                             'cook' : 'cooked_data'}
        return self

    def add_datatype(self, name, file_extension, load_method = None,
                     save_method = None):
        """Adds or replaces a file extension option."""
        self.extensions.update({name : file_extension})
        if load_method:
            setattr(self, '_load_' + name, load_method)
        if save_method:
            setattr(self, '_save_' + name, save_method)
        return self

    def add_folders(self, root_folder, subfolders):
        for folder in listify(subfolders):
            temp_folder = os.path.join(root_folder, folder)
            self._make_folder(folder = temp_folder)
            setattr(self, folder, temp_folder)
        return self

    def create_glob(self, file_type = None, recursive = True):
        """Creates a list of paths in the self.data_in folder based upon
        file_type. If recursive is True, subfolders are searched as well.
        """
        file_type = self._check_file_type(file_type = file_type,
                                          io_status = 'import')
        self.globbed_paths = glob.glob(os.path.join(self.data_in, '**',
                                                    file_type),
                                       recursive = recursive)
        return self

    def create_path(self, folder = '', subfolder = '', prefix = '',
                    file_name = '', suffix = '', file_type = 'csv'):
        """Creates file and/or folder path."""
        if subfolder:
            folder = os.path.join(folder, subfolder)
        self._make_folder(folder)
        if file_name:
            file_name = prefix + file_name + suffix
            file_path = self._create_file_path(folder = folder,
                                               file_name = file_name,
                                               file_type = file_type)
            return file_path
        else:
            return folder

    def initialize_writer(self, file_name, column_list, encoding = None,
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

    def load(self, folder = None, file_name = None, file_path = None,
             file_type = None, **kwargs):
        # Sets appropriate file_type using argument or default.
        file_type = self._check_file_type(file_type = file_type,
                                          io_status = 'import')
        # Sets file_path based on arguments or defaults.
        file_path = self._create_file_path(folder = folder,
                                           file_name = file_name,
                                           file_path = file_path,
                                           file_type = file_type,
                                           io_status = 'import')
        return getattr(self, '_load_' + file_type)(file_path = file_path,
                                                   **kwargs)

    def prepare(self):
        """Sets subfolders; defaults mirror cookiecutter names for data
        folders.
        """
        self.data = os.path.join(self.root_folder, self.data_folder)
        self.results = os.path.join(self.root_folder, self.results_folder)
        for root_folder, subfolders in self.folder_groups.items():
            self.add_folders(root_folder = root_folder,
                             subfolders = subfolders)
        return self

    def save(self, variable, folder = None, file_name = None, file_path = None,
             file_type = None, **kwargs):
        # Changes boolean values to 1/0 if self.boolean_out = False
        variable = self._check_boolean_out(variable = variable)
        # Sets appropriate file_type using argument or default.
        file_type = self._check_file_type(file_type = file_type,
                                          io_status = 'export')
        # Sets file_path based on arguments or defaults.
        file_path = self._create_file_path(folder = folder,
                                           file_name = file_name,
                                           file_path = file_path,
                                           file_type = file_type,
                                           io_status = 'export')
        # Calls appropriate method to save file
        getattr(self, '_save_' + file_type)(variable, file_path, **kwargs)
        return