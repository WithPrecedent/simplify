
import csv
from dataclasses import dataclass
import glob
import os
import pickle

import pandas as pd

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
        datetime_naming: a boolean value setting whether the date and time
            should be used to create experiment subfolders (so that prior
            results are not overwritten).
        auto_prepare: boolean value as to whether prepare method should be
            called when the class is instanced.
    """
    menu : object
    root_folder : str = ''
    data_folder : str = 'data'
    results_folder : str = 'results'
    datetime_naming : bool = True
    auto_prepare : bool = True

    def __post_init__(self):
        """injects select settings as attributes, sets default attributes,
        and calls prepare method if auto_prepare = True.
        """
        self.menu.inject(instance = self, sections = ['general', 'files'])
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    @property
    def file_in(self):
        if self.import_folder[self.step] in ['raw']:
            return 'glob'
        else:
            return list(self.import_folder.keys())[list(
                    self.import_folder.keys()).index(self.step) - 1] + '_data'

    @property
    def file_out(self):
        if self.export_folder[self.step] in ['raw']:
            return 'glob'
        else:
            return list(self.export_folder.keys())[list(
                    self.export_folder.keys()).index(self.step)] + '_data'

    @property
    def folder_in(self):
        return getattr(self, self.import_folder[self.step])

    @property
    def folder_out(self):
        return getattr(self, self.export_folder[self.step])

    @property
    def format_in(self):
        return self._get_file_format(io_status = 'import')

    @property
    def format_out(self):
        return self._get_file_format(io_status = 'export')

    @property
    def path_in(self):
        return self.create_path(io_status = 'import')

    @property
    def path_out(self):
        return self.create_path(io_status = 'export')

    def _add_branch(self, root_folder, subfolders):
        """Creates a branch of a folder tree and stores each folder name as
        a local variable containing the path to that folder.

        Parameters:
            root_folder: the folder from which the tree branch should be
                created.
            subfolders: a list of subfolder names forming the tree branch.
        """
        for subfolder in listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                              subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
            root_folder = temp_folder
        return self

    def _check_boolean_out(self, variable):
        """Either leaves boolean values as True/False or changes values to 1/0
        based on user settings.

        Parameters:
            variable: pandas DataFrame or Series with boolean output values.
        """
        # Checks whether True/False should be exported in data files. If
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

    def _check_file_name(self, file_name, io_status = None):
        """Checks passed file_name to see if it exists. If not, depending
        upon the io_status, a default file_name is returned.

        Parameters:
            file_name: string containing a file_name (without extension).
            io_status: either 'import' or 'export' based upon whether the user
                is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.
        """
        if file_name:
            return file_name
        elif io_status == 'import':
            return self.file_in
        elif io_status == 'export':
            return self.file_out

    def _check_file_format(self, file_format = None, io_status = None):
        """Checks value of local file_format variable. If not supplied, the
        default from the Menu instance is used based upon whether import or
        export methods are being used. If the Menu options don't exist,
        '.csv' is returned.

        Parameters:
            file_format: string matching one of the supported file types in
                self.extensions.
            io_status: either 'import' or 'export' based upon whether the user
                is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.
        """
        if file_format:
            return file_format
        elif io_status == 'import':
            return self.format_in
        elif io_status == 'export':
            return self.format_out
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

    def _check_folder(self, folder, io_status = None):
        """Checks if folder is a full path or string matching an attribute.
        If no folder name is provided, a default value is used.

        Parameters:
            folder: a string either containing a folder path or the name of an
                attribute containing a folder path.
            io_status: either 'import' or 'export' based upon whether the user
                is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.
        """
        if folder and os.path.isdir(folder):
            return folder
        elif folder and isinstance(folder, str):
            return getattr(self, folder)
        elif io_status == 'import':
            return self.folder_in
        elif io_status == 'export':
            return self.folder_out

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

    def _check_root_folder(self):
        if self.root_folder:
            if os.path.isdir(self.root_folder):
                self.root = self.root_folder
            else:
                self.root = os.path.abspath(self.root_folder)
        else:
            self.root = os.path.join('..', '..')
        return self

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
        if not test_data:
            return None
        elif not test_data and not self.test_data:
            return None
        else:
            if not test_rows:
                test_rows = self.test_rows
            return test_rows

    def _get_file_format(self, io_status):
        if getattr(self, io_status + '_folder')[self.step] in ['raw']:
            return self.source_format
        elif getattr(self, io_status + '_folder')[self.step] in ['interim']:
            return self.interim_format
        elif getattr(self, io_status + '_folder')[self.step] in ['processed']:
            return self.final_format
        return self

    def _load_csv(self, file_path, **kwargs):
        """Loads csv file into a pandas DataFrame."""
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows' : self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _load_excel(self, file_path, **kwargs):
        additional_kwargs = ['index_col', 'header', 'usecols']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows' : self.test_chunk})
        variable = pd.read_excel(file_path, **kwargs)
        return variable

    def _load_feather(self, file_path, **kwargs):
        """Loads feather file into pandas DataFrame."""
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _load_h5(self, file_path, **kwargs):
        return self._load_hdf(file_path, **kwargs)

    def _load_hdf(self, file_path, **kwargs):
        """Loads hdf5 file into pandas DataFrame."""
        additional_kwargs = ['columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_hdf(file_path, **kwargs)

    def _load_json(self, file_path, **kwargs):
        """Loads json file into pandas DataFrame."""
        additional_kwargs = ['encoding', 'columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize' : self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns' : kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_json(file_path = file_path, **kwargs)

    def _load_pickle(self, file_path, **kwargs):
        """Returns an unpickled python object."""
        return pickle.load(open(file_path, 'rb'))

    def _load_png(self, file_path, **kwargs):
        error = 'loading .png files is not supported'
        raise NotImplementedError(error)

    def _load_text(self, file_path, **kwargs):
        return self._load_txt(file_path = file_path, **kwargs)

    def _load_txt(self, file_path, **kwargs):
        with open(file_path, mode = 'r', errors = 'ignore',
                  encoding = self.file_encoding) as a_file:
            return a_file.read()

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist.

        Parameters:
            folder: a string containing the path of the folder.
        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _save_csv(self, variable, file_path, **kwargs):
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_excel(self, variable, file_path, **kwargs):
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.excel(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_feather(self, variable, file_path, **kwargs):
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _save_h5(self, variable, file_path, **kwargs):
        variable.to_hdf(file_path, **kwargs)
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

    def _save_png(self, variable, file_path, **kwargs):
        variable.savefig(file_path, bbox_inches = 'tight')
        variable.close()
        return

    def _set_defaults(self):
        """Creates data, results, and experiment folders based upon passed
        parameters. The experiment folder name is based upon the date and time
        to avoid overwriting previous experiments unless datetime_naming is set
        to False. If False, a default folder named 'experiment' will be used.
        Also, creates a dictionary for file_format names and extensions.
        """
        self.extensions = {'csv' : '.csv',
                           'excel' : '.xlsx',
                           'feather' : '.ftr',
                           'h5' : '.hdf',
                           'hdf' : '.hdf',
                           'pickle' : '.pkl',
                           'png' : '.png',
                           'text' : '.txt',
                           'txt' : '.txt'}
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        self.default_kwargs = {'index' : False,
                               'header' : None,
                               'low_memory' : False,
                               'dialect' : 'excel',
                               'usecols' : None,
                               'columns' : None,
                               'nrows' : None,
                               'index_col' : False}
        self.import_folder = {'sow' : 'raw',
                              'harvest' : 'raw',
                              'clean' : 'interim',
                              'bundle' : 'interim',
                              'deliver' : 'interim',
                              'cook' : 'processed'}
        self.export_folder = {'sow' : 'raw',
                              'harvest' : 'interim',
                              'clean' : 'interim',
                              'bundle' : 'interim',
                              'deliver' : 'processed',
                              'cook' : 'processed'}
        return self

    def add_file_format(self, file_format, extension, load_method,
                        save_method):
        """Adds or replaces a file extension option.

        Parameters:
            file_format: string name of the file_format.
            extension: file extension (without period) to be used.
            load_method: a method to be used when loading files of the passed
                file_format.
            save_method: a method to be used when saving files of the passed
                file_format.
        """
        self.extensions.update({file_format : extension})
        if isinstance(load_method, str):
            setattr(self, '_load_' + file_format, '_load_' + load_method)
        else:
            setattr(self, '_load_' + file_format, load_method)
        if isinstance(save_method, str):
            setattr(self, '_save_' + file_format, '_save_' + save_method)
        else:
            setattr(self, '_save_' + file_format, save_method)
        return self

    def add_folders(self, root_folder, subfolders):
        """Adds a list of subfolders to an existing root_folder.

        Parameters:
            root_folder: path of folder where subfolders should be created.
            subfolders: list of subfolder names to be created.
        """
        for subfolder in listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                              subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
        return self

    def add_tree(self, folder_tree):
        """Adds a folder tree to disc with corresponding attributes to the
        Inventory instance.

        Parameters:
            folder_tree: a dictionary containing a folder tree to be created
                with corresponding attributes to the Inventory instance.
        """
        for folder, subfolders in folder_tree.items():
            self._add_branch(root_folder = folder,
                             subfolders = subfolders)
        return self

    def conform(self, step):
        """Sets self.step to current step in siMpLify."""
        self.step = step
        return self

    def create_batch(self, folder = None, file_format = None,
                    include_subfolders = True):
        """Creates a list of paths in the self.data_in folder based upon
        file_format. If recursive is True, subfolders are searched as well for
        matching file_format files.
        """
        folder = self._check_folder(folder = folder)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        extension = self.extensions[file_format]
        return glob.glob(os.path.join(folder, '**', '*' + extension),
                         recursive = include_subfolders)

    def create_folder(self, folder, subfolder = None):
        """Creates folder path."""
        if subfolder:
            if folder and os.path.isdir(folder):
                folder = os.path.join(folder, subfolder)
            else:
                folder = os.path.join(getattr(self, folder), subfolder)
        self._make_folder(folder = folder)
        return folder

    def create_path(self, folder = None, file_name = None, file_format = None,
                    io_status = None):
        """Creates file path."""
        folder = self._check_folder(folder = folder,
                                    io_status = io_status)
        file_name = self._check_file_name(file_name = file_name,
                                          io_status = io_status)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = io_status)
        extension = self.extensions[file_format]
        if file_name == 'glob':
            file_path = self.create_batch(folder = folder,
                                          file_format = file_format)
        else:
            file_path = os.path.join(folder, file_name) + extension
        return file_path

    def initialize_writer(self, file_path, columns, encoding = None,
                          dialect = 'excel'):
        """Initializes writer object for line-by-line exporting to a .csv file.

        Parameters:
            file_path: a complete path to the file being created and written
                to.
            columns: a list of column names to be added to the first row of the
                file as column headers.
            encoding: a python encoding type. If none is provided, the default
                option is used.
            dialect: the specific type of csv file created.
        """
        if not columns:
            error = 'initialize_writer requires columns as a list type'
            raise TypeError(error)
        encoding = self._check_encoding()
        with open(file_path, mode = 'w', newline = '',
                  encoding = encoding) as self.output_series:
            self.writer = csv.writer(self.output_series, dialect = dialect)
            self.writer.writerow(columns)
        return self

    def inject(self, instance, sections, override = True):
        """Stores the default paths in the passed instance.

        Parameters:

            instance: either a class instance or class to which attributes
                should be added.
            sections: list of paths to be added to passed class. Data import
                and export paths are automatically added.
            override: if True, even existing attributes in instance will be
                replaced by items from this class.
        """
        instance.data_in = self.path_in
        instance.data_out = self.path_out
        for section in listify(sections):
            if hasattr(self, section + '_in') and override:
                setattr(instance, section + '_in',
                        getattr(self, section + '_in'))
                setattr(instance, section + '_out',
                        getattr(self, section + '_out'))
            elif override:
                setattr(instance, section, getattr(self, section))
        return

    def iterate(self, plans, ingredients = None, return_ingredients = True):
        """Iterates through a list of files contained in self.batch and
        applies the plans created by a Planner method (or subclass).

        Parameters:
            plans: an attribute of a Planner method (or subclass) containing
                methods used to modify an Ingredients instance.
            ingredients: an instance of Ingredients (or subclass).
            return_ingredients: a boolean value indicating whether ingredients
                should be returned by this method.
        """
        if ingredients:
            for file_path in self.batch:
                ingredients.source = self.load(file_path = file_path)
                for plan in plans:
                    ingredients = plan.start(ingredients = ingredients)
            if return_ingredients:
                return ingredients
            else:
                return self
        else:
            for file_path in self.batch:
                for plan in plans:
                    plan.start()
            return self

    def load(self, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Imports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Parameters:
            file_path: a complete file path for the file to be loaded.
            folder: a path to the folder from where the file should be loaded
                (not used if file_path is passed).
            file_name: a string containing the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                self.extensions.
            kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.
        """
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        if not file_path:
            file_path = self.create_path(folder = folder,
                                         file_name = file_name,
                                         file_format = file_format,
                                         io_status = 'import')
        if isinstance(file_path, str):
            return getattr(self, '_load_' + file_format)(file_path = file_path,
                                                         **kwargs)
        elif isinstance(file_path, list):
            error = 'file_path is a glob list - use iterate instead'
            raise TypeError(error)
        else:
            return file_path

    def prepare(self):
        """Creates data and results folders as well as other default subfolders
        (mirroring the cookie_cutter folder tree by default).
        """
        self._check_root_folder()
        self.add_folders(root_folder = self.root,
                         subfolders = [self.data_folder, self.results_folder])
        self.add_folders(root_folder = self.data,
                         subfolders = self.data_subfolders)
        return self

    def save(self, variable, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Exports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Parameters:
            variable: the variable being exported.
            file_path: a complete file path for the file to be saved.
            folder: a path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name: a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format: a string matching one the file formats in
                self.extensions.
            kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.
        """
        # Changes boolean values to 1/0 if self.boolean_out = False
        variable = self._check_boolean_out(variable = variable)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'export')
        if not file_path:
            file_path = self.create_path(folder = folder,
                                         file_name = file_name,
                                         file_format = file_format,
                                         io_status = 'export')
        getattr(self, '_save_' + file_format)(variable, file_path, **kwargs)
        return