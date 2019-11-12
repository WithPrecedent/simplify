"""
.. module:: depot
:synopsis: file management for siMpLify.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import csv
from dataclasses import dataclass
import datetime
import glob
import os
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.distributor import SimpleDistributor
from simplify.core.ingredients import Ingredients
from simplify.core.plan import SimplePlan
from simplify.core.types import FileTypes


@dataclass
class Depot(SimpleDistributor):
    """Manages files and folders for the siMpLify package.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and allows loading and saving of siMpLify, pandas,
    and numpy objects in set folders.

    Args:
        name (Optional[str]): designates the name of the class which should
            match the section of settings in the Idea instance and other methods
            throughout the siMpLify package. If subclassing siMpLify classes,
            it is often a good idea to maintain to the same 'name' attribute
            as the base class for effective coordination between siMpLify
            classes. Defaults to 'depot'.
        root_folder (Optional[str]): the complete path from which the other
            paths and folders used by Depot should be created. Defaults to ''
        data_folder (Optional[str]): the data subfolder name or a complete path
            if the 'data_folder' is not off of 'root_folder'. Defaults to
            'data'.
        results_folder (Optional[str]): the results subfolder name or a complete
            path if the 'results_folder' is not off of 'root_folder'. Defaults
            to 'results'.
        datetime_naming (Optional[bool]): whether the date and time should be
            used to create experiment subfolders (so that prior results are not
            overwritten). Defaults to True.

    """
    name: Optional[str] = 'depot'
    root_folder: Optional[str] = ''
    data_folder: Optional[str] = 'data'
    results_folder: Optional[str] = 'results'
    datetime_naming: Optional[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.root_folder, Depot):
            self = self.root_folder
        else:
            self.idea_sections = ['files']
            self.root_folder = self.root_folder or ''
            super().__post_init__()
        return self

    """ Private Methods """

    def _add_branch(self,
            root_folder: str,
            subfolders: Union[List[str], str]) -> None:
        """Creates a branch of a folder tree.

        Each created folder name is also stored as a local attribute with the
        same name as the created folder.

        Args:
            root_folder (str): the folder from which the tree branch should be
                created.
            subfolders (Union[List[str], str]): subfolder names to form the tree
                branch.

        """
        for subfolder in listify(subfolders):
            temp_folder = self.create_folder(folder = root_folder,
                                             subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
            root_folder = temp_folder
        return self

    def _check_boolean_out(self,
            data: Union[pd.Series, pd.DataFrame]) -> (
                Union[pd.Series, pd.DataFrame]):
        """Converts bool to 1/0 if 'boolean_out' is False.

        Args:
            data (Union[DataFrame, Series]): pandas object with some boolean
                values.

        Returns:
            data (Union[DataFrame, Series]): either the original pandas data or
                the dataset with True/False converted to 1/0.

        """
        # Checks whether True/False should be exported in data files. If
        # 'boolean_out' is set to False, 1/0 are used instead.
        if hasattr(self, 'boolean_out') and self.boolean_out == False:
            data.replace({True: 1, False: 0}, inplace = True)
        return data

    def _check_file_name(self, file_name: str, io_status: str) -> str:
        """Selects 'file_name' or default values.

        Args:
            file_name (str): file name (without extension).
            io_status (str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file name.

        """
        if file_name:
            return file_name
        else:
            return self.data_file_names[self.step][
                self.settings_index[io_status]]

    def _check_file_format(self, file_format: str, io_status: str) -> str:
        """Selects 'file_format' or default value.

        Args:
            file_format (str): one of the supported file types in 'extensions'.
            io_status (str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.

        """
        if file_format:
            return file_format
        else:
            return getattr(self, self.data_file_formats[self.step][
                self.settings_index[io_status]])

    def _check_folder(self, folder: str, io_status: str) -> str:
        """Selects 'folder' or default value.

        Args:
            folder: a string either containing a folder path or the name of an
                attribute containing a folder path.
            io_status (str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file folder path.

        """
        if folder and os.path.isdir(folder):
            return folder
        elif folder and isinstance(folder, str):
            return getattr(self, folder)
        else:
            return self.data_folders[self.step][self.settings_index[io_status]]

    def _check_kwargs(self,
            variables_to_check: List[str],
            passed_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Selects kwargs for particular methods.

        If a needed argument was not passed, default values are used.

        Args:
            variables_to_check (List[str]): variables to check for values.
            passed_kwargs (Dict[str, Any]): kwargs passed to method.

        Returns:
            new_kwargs (Dict[str, Any]): kwargs with only relevant parameters.

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

    def _check_root_folder(self) -> None:
        """Checks if 'root_folder' exists on disc. If not, it is created."""
        if self.root_folder:
            if os.path.isdir(self.root_folder):
                self.root = self.root_folder
            else:
                self.root = os.path.abspath(self.root_folder)
        else:
            self.root = os.path.join('..', '..')
        return self

    def _get_file_format(self, io_status) -> str:
        """Returns appropriate file format based on 'step' and 'io_status'.

        Args:
            io_status (str): either 'import' or 'export' based upon whether the
                user is seeking the appropriate file type based upon whether the
                file in question is being imported or exported.

        Returns:
            str containing file format.

        """
        if io_status == 'import':
            return self.state_machine.format_in
        else:
            return self.state_machine.format_out

    def _load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads csv file into a pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads Excel file into a pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        additional_kwargs = ['index_col', 'header', 'usecols']
        kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                    passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_excel(file_path, **kwargs)
        return variable

    def _load_feather(self, file_path: str, **kwargs):
        """Loads feather file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _load_h5(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads hdf5 with '.h5' extension into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        return self._load_hdf(file_path, **kwargs)

    def _load_hdf(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads hdf5 file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

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

    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads json file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

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

    def _load_pickle(self, file_path: str, **kwargs) -> object:
        """Returns an unpickled python object.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        return pickle.load(open(file_path, 'rb'))

    def _load_png(self, file_path: str, **kwargs) -> NotImplementedError:
        """Although png files are saved by siMpLify, they cannot be loaded.

        Raises:
            NotImplementedError: if called.

        """
        error = 'loading .png files is not supported'
        raise NotImplementedError(error)

    def _load_text(self, file_path: str, **kwargs) -> str:
        """Loads text file with python reader.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        return self._load_txt(file_path = file_path, **kwargs)

    def _load_txt(self, file_path: str, **kwargs) -> str:
        """Loads text file with python reader.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disc.

        """
        with open(file_path, mode = 'r', errors = 'ignore',
                  encoding = self.file_encoding) as a_file:
            return a_file.read()

    def _make_folder(self, folder: str) -> None:
        """Creates folder if it doesn't already exist.

        Args:
            folder (str): the path of the folder.

        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _save_csv(self, variable: pd.Series, file_path: str, **kwargs) -> None:
        """Saves pandas Series to disc as .csv file.

        Args:
            variable (Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_excel(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disc as an Excel file.

        Args:
            variable (DataFrame or Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(variables_to_check = additional_kwargs,
                                        passed_kwargs = kwargs)
            variable.excel(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _save_feather(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disc as a feather file.

        Args:
            variable (DataFrame or Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _save_h5(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disc as a hdf file with .h5 extension.

        Args:
            variable (DataFrame or Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_hdf(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disc as a hdf file.

        Args:
            variable (DataFrame or Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _save_json(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disc as an json file.

        Args:
            variable (DataFrame or Series): variable to be saved to disc.
            file_path (str): complete file path of file.

        """
        variable.to_json(file_path, **kwargs)
        return

    def _save_pickle(self, variable: object, file_path: str, **kwargs):
        """Pickles file and saves it to disc.
        Args:
            variable (object): variable to be saved to disc.
            file_path (str): complete file path of file.
        """
        pickle.dump(variable, open(file_path, 'wb'))
        return

    def _save_png(self, variable: object, file_path: str, **kwargs) -> None:
        """Saves png file to disc.
        Args:
            variable (matplotlib object): variable to be saved to disc.
            file_path (str): complete file path of file.
        """
        variable.savefig(file_path, bbox_inches = 'tight')
        variable.close()
        return

    def _set_experiment_folder(self) -> None:
        """Sets the experiment folder and corresponding attribute."""
        if self.datetime_naming:
            subfolder = '_'.join(['experiment_',
                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
        else:
            subfolder = 'experiment'
        self.experiment = self.create_folder(folder = self.results,
                                             subfolder = subfolder)
        return self

    def _set_plan_folder(self,
            plan: SimplePlan,
            name: Optional[str] = None) -> None:
        """Creates folder path for iterable-specific exports.

        Args:
            plan (SimplePlan): an instance of SimplePackage.
            name (string): name of attribute for the folder path to be stored
                and the prefix of the folder to be created on disc.

        """
        if name:
            subfolder = name + '_'
        else:
            subfolder = iterable.name + '_'
        if self._exists('naming_classes'):
            for step in listify(self.naming_classes):
                subfolder += getattr(iterable, step).technique + '_'
        subfolder += str(iterable.number)
        setattr(self, name, self.create_folder(folder = self.experiment,
                subfolder = subfolder))
        return self

    """ Public Tool Methods """

    def add_tree(self, folder_tree: Dict[str, str]) -> None:
        """Adds folder tree to disc and adds corresponding attributes.

        Args:
            folder_tree (Dict[str, str]): a folder tree to be created with
                corresponding attributes to the Depot instance.

        """
        for folder, subfolders in folder_tree.items():
            self._add_branch(root_folder = folder, subfolders = subfolders)
        return self

    def create_batch(self,
            folder: Optional[str] = None,
            file_format: Optional[str] = None,
            include_subfolders: Optional[bool] = True) -> Iterable[str]:
        """Creates a list of paths in 'folder_in' based upon 'file_format'.

        If 'include_subfolders' is True, subfolders are searched as well for
        matching 'file_format' files.

        Args:
            folder (Optional[str]): path of folder or string corresponding to
                class attribute with path.
            file_format (Optional[str]): file format name.
            include_subfolders (Optional[bool]):  whether to include files in
                subfolders when creating a batch.

        """
        folder = self._check_folder(folder = folder)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'import')
        extension = self.extensions[file_format]
        return glob.glob(os.path.join(folder, '**', '*' + extension),
                recursive = include_subfolders)

    def create_folder(self,
            folder: str,
            subfolder: Optional[str] = None) -> None:
        """Creates folder path from component parts.

        Args:
            folder (str): path of folder or string corresponding to class
                attribute containing folder path.
            subfolder (Optional[str]): subfolder name to be created off of
                'folder'.

        """
        if subfolder:
            if folder and os.path.isdir(folder):
                folder = os.path.join(folder, subfolder)
            else:
                folder = os.path.join(getattr(self, folder), subfolder)
        self._make_folder(folder = folder)
        return folder

    def create_path(self,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None,
            io_status: Optional[str] = None):
        """Creates file path from component parts.

        Args:
            folder (Optional[str]): path of folder or string corresponding to
                class attribute containing folder path.
            file_name (Optional[str]): file name without extension.
            file_format (Optional[str]): file format name from 'extensions'
                dict.
            io_status (Optional[str]): 'import' or 'export' indicating which
                direction the path is used for storing files (only needed to
                use defaults when other parameters are not provided).

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

    def initialize_writer(self,
            file_path: str,
            columns: List[str],
            encoding: Optional[str] = None,
            dialect: Optional[str] = 'excel'):
        """Initializes writer object for line-by-line exporting to a .csv file.

        Args:
            file_path (str): a complete path to the file being written to.
            columns (List[str]): column names to be added to the first row of
                the file as column headers.
            encoding (str): a python encoding type.
            dialect (str): the specific type of csv file created. Defaults to
                'excel'.

        """
        if not columns:
            error = 'initialize_writer requires columns as a list type'
            raise TypeError(error)
        with open(file_path, mode = 'w', newline = '',
                  encoding = self.file_encoding) as self.output_series:
            self.writer = csv.writer(self.output_series, dialect = dialect)
            self.writer.writerow(columns)
        return self

    def iterate(self,
            plans: List[str],
            ingredients: Ingredients = None,
            return_ingredients: Optional[bool] = True):
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

    """ Public Import/Export Methods """

    def load(self,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None,
            **kwargs):
        """Imports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path (Optional[str]): a complete file path for the file to be
                loaded.
            folder (Optional[str]: a path to the folder from where file is
                located (not used if file_path is passed).
            file_name (Optional[str]): contains the name of the file to be
                loaded without the file extension (not used if file_path is
                passed).
            file_format (Optional[str]): a string matching one the file formats
                in 'extensions'.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        Raises:
            TypeError: if file_path is not a string (likely a glob list)

        """
        file_format = self._check_file_format(
            file_format = file_format,
            io_status = 'import')
        if not file_path:
            file_path = self.create_path(
                folder = folder,
                file_name = file_name,
                file_format = file_format,
                io_status = 'import')
        if isinstance(file_path, str):
            return getattr(self, '_load_' + file_format)(
                file_path = file_path,
                **kwargs)
        elif isinstance(file_path, list):
            error = 'file_path is a glob list - use iterate instead'
            raise TypeError(error)
        else:
            return None

    def save(self,
            variable: Any,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None,
            **kwargs) -> None:
        """Exports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable (Any): the variable being exported.
            file_path (Optional[str]): a complete file path for the file to be
                saved.
            folder (Optional[str]): path to the folder where the file should be
                saved (not used if file_path is passed).
            file_name (Optional[str]): a string containing the name of the file
                to be saved without the file extension (not used if file_path is
                passed).
            file_format (Optional[str]): a string matching one the file formats
                in 'extensions'.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        """
        # Changes boolean values to 1/0 if self.boolean_out = False
        variable = self._check_boolean_out(variable = variable)
        file_format = self._check_file_format(file_format = file_format,
                                              io_status = 'export')
        if not file_path:
            file_path = self.create_path(
                folder = folder,
                file_name = file_name,
                file_format = file_format,
                io_status = 'export')
        getattr(self, '_'.join(['_save_', file_format]))(
            variable, file_path, **kwargs)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default folder and file settings."""
        # Calls SimpleComposite draft for initial baseline settings.
        super().draft()
        self._check_root_folder()
        # Creates dict with file format names and file extensions.
        self.extensions = FileTypes()
        # Creates list of default subfolders from 'data_folder' to create.
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        # Creates default parameters when they are not passed as kwargs to
        # methods in the class.
        self.default_kwargs = {
            'index': False,
            'header': None,
            'low_memory': False,
            'dialect': 'excel',
            'usecols': None,
            'columns': None,
            'nrows': None,
            'index_col': False}
        # Creates default data folders, file names, and file formats linked to
        # the various stages of the siMpLify process. Each values includes a
        # 2-item list with the first item being the default import option and
        # the second being the default export option.
        self.data_folders = {
            'sow': ['raw', 'raw'],
            'reap': ['raw', 'interim'],
            'clean': ['interim', 'interim'],
            'bale': ['interim', 'interim'],
            'deliver': ['interim', 'processed'],
            'chef': ['processed', 'processed'],
            'critic': ['processed', 'recipe']}
        self.data_file_names = {
            'sow': [None, None],
            'harvest': [None, 'harvested_data'],
            'clean': ['harvested_data', 'cleaned_data'],
            'bale': ['cleaned_data', 'baled_data'],
            'deliver': ['baled_data', 'final_data'],
            'chef': ['final_data', 'final_data'],
            'critic': ['final_data', 'predicted_data']}
        self.data_file_formats = {
            'sow': ['source_format', 'source_format'],
            'harvest': ['source_format', 'interim_format'],
            'clean': ['interim_format', 'interim_format'],
            'bale': ['interim_format', 'interim_format'],
            'deliver': ['interim_format', 'final_format'],
            'chef': ['final_format', 'final_format'],
            'critic': ['final_format', 'final_format']}
        # Sets dict to translate 'import'/'export' strings to index of lists in
        # the data settings above.
        self.settings_index = {'import': 0, 'export': 1}
        # Sets default folders for results to be exported based upon type of
        # information being exported.
        self.results_folders = {
            'isolated': 'recipe',
            'comparative': 'experiment'}
        self._check_root_folder()
        self.edit_folders(
            root_folder = self.root,
            subfolders = [self.data_folder, self.results_folder])
        self.edit_folders(
            root_folder = self.data,
            subfolders = self.data_subfolders)
        return self

    def edit_default_kwargs(self,
            kwargs_keys: Union[List[str], str],
            settings: Union[List[str], str]) -> None:
        """Adds or replaces default keys and values for kwargs.

        Args:
            kwargs_keys (Union[List[str], str]): key(s) to change in
                'default_kwargs'.
            settings (Union[List[str], str]): values(s) to change in
                'default_kwargs'.

        """
        self.default_kwargs(dict(zip(kwargs, settings)))
        return self

    def edit_file_formats(self,
            file_format: str,
            extension: str,
            load_method: Callable,
            save_method: Callable) -> None:
        """Adds or replaces a file extension option.

        Args:
            file_format (str): string name of the file_format.
            extension (str): file extension (without period) to be used.
            load_method (Callable): a method to be used when loading files of
                the passed file_format.
            save_method (Callable): a method to be used when saving files of the
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

    def edit_file_names(self,
            techniques: Union[List[str], str],
            file_names: Union[List[str], str]) -> None:
        """Adds data file names for specific techniques.

        Args:
            techniques(Union[List[str], str]): step or step names.
            file_names(Union[List[str], str]): file name or file names (without
                extension(s)).

        """
        self.file_names.update(dict(zip(techniques, file_names)))
        return self

    def edit_folders(self,
            subfolders: Union[List[str], str],
            root_folder: Optional[str] = None) -> None:
        """Adds a list of subfolders to an existing root_folder.

        Args:
            subfolders(Union[List[str], str]): subfolder names to be created.
            root_folder (Optional[str]): path of folder where subfolders should
                be created.

        """
        if root_folder is None:
            root_folder = self.root_folder
        for subfolder in listify(subfolders):
            temp_folder = self.create_folder(
                folder = root_folder,
                subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
        return self

    def publish(self, instance: 'SimpleComposite') -> 'SimpleComposite':
        """Injects Depot instance into passed instance.

        Args:
            instance (SimpleComposite): a class instance to which attributes should
                be added.

        Returns:
            SimpleComposite: instance with attribute(s) added.

        """
        instance.depot = self
        return instance