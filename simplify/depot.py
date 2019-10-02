"""
.. module:: depot
:synopsis: contains class for file management of siMpLify package.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import csv
from dataclasses import dataclass
import glob
import os
import pickle

import pandas as pd

from simplify.core.base import SimpleClass
from simplify.core.types import FileTypes


@dataclass
class Depot(SimpleClass):
    """Manages files and folders for the siMpLify package.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and allows loading and saving of siMpLify, pandas,
    and numpy objects in set folders.

    Args:
        root_folder(str): the complete path from which the other paths and
            folders used by Depot should be created.
        data_folder(str): the data subfolder name or a complete path if the
            'data_folder' is not off of 'root_folder'.
        results_folder(str): the results subfolder name or a complete path if
            the 'results_folder' is not off of 'root_folder'.
        datetime_naming(bool): whether the date and time should be used to
            create experiment subfolders (so that prior results are not
            overwritten).
        auto_finalize(bool): whether to automatically call the 'finalize' method
            when the class is instanced. Unless making major changes to the
            file structure (beyond the 'root_folder', 'data_folder', and
            'results_folder' parameters), this should be set to True.
        state_dependent(bool): whether this class is depending upon the current
            state in the siMpLify package. Unless the user is radically changing
            the way siMpLify works, this should be set to True.
        lazy_import(bool): whether to import 'options' classes lazily. This
            should be set to False because the default 'options' do not include
            any classes in its values.
    """
    root_folder: str = ''
    data_folder: str = 'data'
    results_folder: str = 'results'
    datetime_naming: bool = True
    auto_finalize: bool = True
    state_dependent: bool = True
    lazy_import: bool = False

    def __post_init__(self):
        # Adds additional section of idea to be injected as local attributes.
        self.idea_sections = ['files']
        super().__post_init__()
        return self

    """ Private Methods """

    def _add_branch(self, root_folder, subfolders):
        """Creates a branch of a folder tree and stores each folder name as
        a local variable containing the path to that folder.

        Args:
            root_folder(str): the folder from which the tree branch should be
                created.
            subfolders(str or list): subfolder names to form the tree branch.
        """
        for subfolder in self.listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                             subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
            root_folder = temp_folder
        return self

    def _check_boolean_out(self, variable):
        """Either leaves boolean values as True/False or changes values to 1/0
        based on user settings.

        Args:
            variable(DataFrame or Series): pandas DataFrame or Series with some
                boolean values.

        Returns:
            variable(DataFrame or Series): either the original pandas data or
                the dataset with True/False converted to 1/0.
        """
        # Checks whether True/False should be exported in data files. If
        # 'boolean_out' is set to False, 1/0 are used instead.
        if hasattr(self, 'boolean_out') and self.boolean_out == False:
            variable.replace({True: 1, False: 0}, inplace = True)
        return variable

    def _check_file_name(self, file_name, io_status = None):
        """Checks passed file_name to see if it exists. If not, depending
        upon the io_status, a default file_name is returned.

        Args:
            file_name(str): file name (without extension).
            io_status(str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            string containing file name.
        """
        if file_name:
            return file_name
        elif io_status == 'import':
            return self.state_machine.file_in
        elif io_status == 'export':
            return self.state_machine.file_out

    def _check_file_format(self, file_format = None, io_status = None):
        """Checks value of local file_format variable. If not supplied, the
        default from the Idea instance is used based upon whether import or
        export methods are being used. If the Idea options don't exist,
        '.csv' is returned.

        Args:
            file_format(str): one of the supported file types in 'extensions'.
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.
        """
        if file_format:
            return file_format
        elif io_status == 'import':
            return self._get_file_format(io_status = 'import')
        elif io_status == 'export':
            return self._get_file_format(io_status = 'export')
        else:
            return 'csv'

    def _check_folder(self, folder, io_status = None):
        """Checks if folder is a full path or string matching an attribute.
        If no folder name is provided, a default value is used.

        Args:
            folder: a string either containing a folder path or the name of an
                attribute containing a folder path.
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file folder path.
        """
        if folder and os.path.isdir(folder):
            return folder
        elif folder and isinstance(folder, str):
            return getattr(self, folder)
        elif io_status == 'import':
            return self.state_machine.folder_in
        elif io_status == 'export':
            return self.state_machine.folder_out

    def _check_kwargs(self, variables_to_check, passed_kwargs):
        """Checks kwargs to see which ones are required for the particular
        method and/or substitutes default values if needed.

        Args:
            variables_to_check(list): variables to check for values.
            passed_kwargs(dict): kwargs passed to method.

        Returns:
            new_kwargs(dict): kwargs with only relevant parameters.

        """
        new_kwargs = passed_kwargs
        for variable in variables_to_check:
            if not variable in passed_kwargs:
                if variable in self.default_kwargs:
                    new_kwargs.update(
                            {variable: self.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable: getattr(self, variable)})
        return new_kwargs

    def _check_root_folder(self):
        """Checks if 'root_folder' exists on disc. If not, it is created."""
        if self.root_folder:
            if os.path.isdir(self.root_folder):
                self.root = self.root_folder
            else:
                self.root = os.path.abspath(self.root_folder)
        else:
            self.root = os.path.join('..', '..')
        return self

    def _get_file_format(self, io_status):
        """Returns appropriate file format based on 'step' and 'io_status'.

        Args:
            io_status(str): either 'import' or 'export' based upon whether the
            user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.
        """
        if io_status == 'import':
            return self.state_machine.format_in
        else:
            return self.state_machine.format_out

    def _inject_base(self):
        """Injects parent class with this Depot instance so that the instance is
        available to other files in the siMpLify package.
        """
        SimpleClass.depot = self
        return self

    def _load_csv(self, file_path, **kwargs):
        """Loads csv file into a pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _load_excel(self, file_path, **kwargs):
        """Loads Excel file into a pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['index_col', 'header', 'usecols']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_excel(file_path, **kwargs)
        return variable

    def _load_feather(self, file_path, **kwargs):
        """Loads feather file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _load_h5(self, file_path, **kwargs):
        """Loads hdf5 with '.h5' extension into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        return self._load_hdf(file_path, **kwargs)

    def _load_hdf(self, file_path, **kwargs):
        """Loads hdf5 file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize': self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns': kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_hdf(file_path, **kwargs)

    def _load_json(self, file_path, **kwargs):
        """Loads json file into pandas DataFrame.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(DataFrame): data loaded from disc.
        """
        additional_kwargs = ['encoding', 'columns']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize': self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns': kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_json(file_path = file_path, **kwargs)

    def _load_pickle(self, file_path, **kwargs):
        """Returns an unpickled python object.

        Args:
            file_path: complete file path of file.

        Returns:
            variable(object): pickled object loaded from disc.
        """
        return pickle.load(open(file_path, 'rb'))

    def _load_png(self, file_path, **kwargs):
        """Although png files are saved by siMpLify, they cannot be loaded.

        Raises:
            NotImplementedError: if called.
        """
        error = 'loading .png files is not supported'
        raise NotImplementedError(error)

    def _load_text(self, file_path, **kwargs):
        """Loads text file with python reader.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(str): string loaded from disc.
        """
        return self._load_txt(file_path = file_path, **kwargs)

    def _load_txt(self, file_path, **kwargs):
        """Loads text file with python reader.

        Args:
            file_path(str): complete file path of file.

        Returns:
            variable(str): string loaded from disc.
        """
        with open(file_path, mode = 'r', errors = 'ignore',
                  encoding = self.file_encoding) as a_file:
            return a_file.read()

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist.

        Args:
            folder(str): the path of the folder.
        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _save_csv(self, variable, file_path, **kwargs):
        """Saves csv file to disc.

        Args:
            variable(Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_excel(self, variable, file_path, **kwargs):
        """Saves Excel file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.excel(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_feather(self, variable, file_path, **kwargs):
        """Saves feather file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _save_h5(self, variable, file_path, **kwargs):
        """Saves hdf file with .h5 extension to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_hdf(self, variable, file_path, **kwargs):
        """Saves hdf file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_json(self, variable, file_path, **kwargs):
        """Saves json file to disc.

        Args:
            variable(DataFrame or Series): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.to_json(file_path, **kwargs)
        return

    def _save_pickle(self, variable, file_path, **kwargs):
        """Pickles file and saves it to disc.

        Args:
            variable(object): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        pickle.dump(variable, open(file_path, 'wb'))
        return

    def _save_png(self, variable, file_path, **kwargs):
        """Saves png file to disc.

        Args:
            variable(matplotlib object): variable to be saved to disc.
            file_path(str): complete file path of file.
        """
        variable.savefig(file_path, bbox_inches = 'tight')
        variable.close()
        return

    """ Public Tool Methods """

    def add_tree(self, folder_tree):
        """Adds a folder tree to disc with corresponding attributes to the
        Depot instance.

        Args:
            folder_tree(dict): a folder tree to be created with corresponding
            attributes to the Depot instance.
        """
        for folder, subfolders in folder_tree.items():
            self._add_branch(root_folder = folder, subfolders = subfolders)
        return self

    def create_batch(self, folder = None, file_format = None,
                    include_subfolders = True):
        """Creates a list of paths in 'folder_in' based upon 'file_format'.

        If 'include_subfolders' is True, subfolders are searched as well for
        matching 'file_format' files.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute with path.
            file_format(str): file format name.
            include_subfolders(bool):  whether to include files in subfolders
                when creating a batch.
        """
        folder = self._check_folder(folder = folder)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        extension = self.extensions[file_format]
        return glob.glob(os.path.join(folder, '**', '*' + extension),
                         recursive = include_subfolders)

    def create_folder(self, folder, subfolder = None):
        """Creates folder path from component parts.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute containing folder path.
            subfolder(str): subfolder name to be created off of folder.
        """
        if subfolder:
            if folder and os.path.isdir(folder):
                folder = os.path.join(folder, subfolder)
            else:
                folder = os.path.join(getattr(self, folder), subfolder)
        self._make_folder(folder = folder)
        return folder

    def create_path(self, folder = None, file_name = None, file_format = None,
                    io_status = None):
        """Creates file path from component parts.

        Args:
            folder(str): path of folder or string corresponding to class
                attribute containing folder path.
            file_name(str): file name without extension.
            file_format(str): file format name from 'extensions' dict.
            io_status: 'import' or 'export' indicating which direction the path
                is used for storing files (only needed to use defaults when
                other parameters are not provided).
            """
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

        Args:
            file_path(str): a complete path to the file being written to.
            columns(list): column names to be added to the first row of the
                file as column headers.
            encoding(str): a python encoding type.
            dialect(str): the specific type of csv file created.
        """
        if not columns:
            error = 'initialize_writer requires columns as a list type'
            raise TypeError(error)
        with open(file_path, mode = 'w', newline = '',
                  encoding = self.file_encoding) as self.output_series:
            self.writer = csv.writer(self.output_series, dialect = dialect)
            self.writer.writerow(columns)
        return self

    def iterate(self, plans, ingredients = None, return_ingredients = True):
        """Iterates through a list of files contained in self.batch and
        applies the plans created by a Planner method (or subclass).

        Args:
            plans(list): list of plan types (Recipe, Harvest, etc.)
            ingredients(Ingredients): an instance of Ingredients or subclass.
            return_ingredients(bool): whether ingredients should be returned by
            this method.

        Returns:
            If 'return_ingredients' is True: an in instance of Ingredients.
            If 'return_ingredients' is False, no value is returned.
        """
        if ingredients:
            for file_path in self.batch:
                ingredients.source = self.load(file_path = file_path)
                for plan in plans:
                    ingredients = plan.produce(ingredients = ingredients)
            if return_ingredients:
                return ingredients
            else:
                return self
        else:
            for file_path in self.batch:
                for plan in plans:
                    plan.produce()
            return self

    """ Core siMpLify Public Methods """

    def draft(self):
        """Creates default folder and file dicts."""
        # Initializes dict with file format names and corresponding file
        # extensions.
        self.extensions = FileTypes()
        # Creates list of default subfolders from 'data_folder' to create.
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        # Creates default parameters when they are not passed as kwargs to
        # methods in the class.
        self.default_kwargs = {'index': False,
                               'header': None,
                               'low_memory': False,
                               'dialect': 'excel',
                               'usecols': None,
                               'columns': None,
                               'nrows': None,
                               'index_col': False}
        # Creates options dict with keys as names of stages in siMpLify and the
        # values as 2-item lists with the first item being the default import
        # folder and the second being the default export folder.
        self.options = {'sow': ['raw', 'raw'],
                        'reap': ['raw', 'interim'],
                        'clean': ['interim', 'interim'],
                        'bale': ['interim', 'interim'],
                        'deliver': ['interim', 'processed'],
                        'cook': ['processed', 'processed'],
                        'critic': ['processed', 'processed']}
        # Sets index for import and export folders in 'options' dict.
        self.options_index = {'import': 0,
                              'export': 1}
        # Sets default folders to use for imports and exports that are not
        # data containers. The keys are based upon 'name' attributes of classes.
        self.class_options = {'ingredients': 'default',
                              'datatypes': 'recipe',
                              'cookbook': 'experiment',
                              'recipe': 'recipe',
                              'almanac': 'harvesters',
                              'harvest': 'harvesters',
                              'review': 'experiment',
                              'visualizer': 'recipe',
                              'evaluator': 'recipe',
                              'summarizer': 'recipe',
                              'comparer': 'experiment'}
        return self

    def edit_file_formats(self, file_format, extension, load_method,
                          save_method):
        """Adds or replaces a file extension option.

        Args:
            file_format(str): string name of the file_format.
            extension(str): file extension (without period) to be used.
            load_method(method): a method to be used when loading files of the
            passed file_format.
            save_method(method): a method to be used when saving files of the
            passed file_format.
        """
        self.extensions.update({file_format: extension})
        if isinstance(load_method, str):
            setattr(self, '_load_' + file_format, '_load_' + load_method)
        else:
            setattr(self, '_load_' + file_format, load_method)
        if isinstance(save_method, str):
            setattr(self, '_save_' + file_format, '_save_' + save_method)
        else:
            setattr(self, '_save_' + file_format, save_method)
        return self

    def edit_folders(self, root_folder, subfolders):
        """Adds a list of subfolders to an existing root_folder.

        Args:
            root_folder(str): path of folder where subfolders should be created.
            subfolders(str or list): subfolder names to be created.
        """
        for subfolder in self.listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                             subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
        return self

    def finalize(self):
        """Creates data and results folders as well as other default subfolders.
        """
        self._check_root_folder()
        self.edit_folders(root_folder = self.root,
                         subfolders = [self.data_folder, self.results_folder])
        self.edit_folders(root_folder = self.data,
                         subfolders = self.data_subfolders)
        return self

    def inject(self, instance, sections, override = True):
        """Stores the default paths in the passed instance.

        Args:
            instance(object): either a class instance or class to which
                attributes should be added.
            sections(list): attributes to be added to passed class. Data import
                and export paths are automatically added.
            override(bool): if True, existing attributes in instance will be
                replaced by items from this class.

        Returns:
            No value is returned, but passed instance is now injected with
            selected attributes.
        """
        instance.data_in = self.create_path(io_status = 'import')
        instance.data_out = self.create_path(io_status = 'export')
        for section in self.listify(sections):
            if hasattr(self, section + '_in') and override:
                setattr(instance, section + '_in',
                        getattr(self, section + '_in'))
                setattr(instance, section + '_out',
                        getattr(self, section + '_out'))
            elif override:
                setattr(instance, section, getattr(self, section))
        return

    def load(self, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Imports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path(str): a complete file path for the file to be loaded.
            folder(str): a path to the folder from where file is located
                (not used if file_path is passed).
            file_name(str): contains the name of the file to be loaded
                without the file extension (not used if file_path is passed).
            file_format(str): a string matching one the file formats in
                'extensions'.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        Raises:
            TypeError: if file_path is not a string (likely a glob list)
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
            return None

    def produce(self):
        """Injects Depot instance into base SimpleClass."""
        self._inject_base()
        return self

    def save(self, variable, file_path = None, folder = None, file_name = None,
             file_format = None, **kwargs):
        """Exports file by calling appropriate method based on file_format. If
        the various arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable(any): the variable being exported.
            file_path(str): a complete file path for the file to be saved.
            folder(Str): path to the folder where the file should be saved (not
                used if file_path is passed).
            file_name(str): a string containing the name of the file to be saved
                without the file extension (not used if file_path is passed).
            file_format(str): a string matching one the file formats in
                'extensions'.
            **kwargs: can be passed if additional options are desired specific
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

