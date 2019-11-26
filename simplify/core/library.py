"""
.. module:: library
:synopsis: file management for siMpLify.
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import datetime
import glob
import os
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from simplify.core.typesetter import FileTypes
from simplify.core.utilities import listify


@dataclass
class Distributor(ABC):
    """Base class for siMpLify file management."""

    def __post_init__(self) -> None:
        # Creates dict with file format names and file extensions.
        if not hasattr('extensions'):
            Distributor.extensions = FileTypes()
            self.extensions = Distributor.extensions
        self.draft()
        return self

    """ Private Methods """

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
                        {variable: self.library.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable: getattr(self, variable)})
        return new_kwargs

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, instance: object) -> None:
        """Injects Distributer subclass instance into passed 'instance'.

        The attribute name will be the same as the Distributor subclass
        instance's 'name' attribute value.

        Args:
            instance (object): class (instance) for Distributer subclass to be
                injected.

        Returns:
            instance (object): class (instance) with Distributer subclass
                injected.

        """
        setattr(instance, self.name, self)
        return instance


@dataclass
class Library(Distributor):
    """Manages files and folders for siMpLify.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and provides methods for loading and saving siMpLify,
    pandas, and numpy objects.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        root_folder (Optional[str]): the complete path from which the other
            paths and folders used by library should be created. Defaults to
            '' (the current working directory).
        data_folder (Optional[str]): the data subfolder name or a complete path
            if the 'data_folder' is not off of 'root_folder'. Defaults to
            'data'.
        results_folder (Optional[str]): the results subfolder name or a complete
            path if the 'results_folder' is not off of 'root_folder'. Defaults
            to 'results'.
        datetime_naming (Optional[bool]): whether the date and time should be
            used to create Book subfolders (so that prior results are not
            overwritten). Defaults to True.

    """
    idea: 'Idea'
    name: Optional[str] = 'library'
    root_folder: Optional[str] = ''
    data_folder: Optional[str] = 'data'
    results_folder: Optional[str] = 'results'
    datetime_naming: Optional[bool] = True


    def __post_init__(self) -> None:
        """Processes key passed arguments to prepare class instance."""
        if isinstance(self.root_folder, Library):
            self = self.root_folder
        else:
            self.idea_sections = ['files']
            self = self.idea.apply(instance = self)
            self.root = self.root_folder or ''
            self.draft()
        return self

    """ Private Methods """

    def _check_root_folder(self) -> None:
        """Checks if 'root_folder' exists on disk. If not, it is created."""
        try:
            if os.path.isdir(self.root_folder):
                self.root = self.root_folder
            else:
                self.root = os.path.abspath(self.root_folder)
        except TypeError:
            self.root = os.path.join('..', '..')
        return self

    def _set_book_folder(self) -> None:
        """Sets the book folder and corresponding attribute."""
        if self.datetime_naming:
            subfolder = '_'.join(['book_',
                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
        else:
            subfolder = 'book'
        self.inventory.book = self.inventory.create_folder(
            folder = self.results,
            subfolder = subfolder)
        return self

    def _set_chapter_folder(self,
            chapter: 'Chapter',
            name: Optional[str] = None) -> None:
        """Creates folder path for iterable-specific exports.

        Args:
            chapter (Chapter): an instance of SimplePackage.
            name (string): name of attribute for the folder path to be stored
                and the prefix of the folder to be created on disk.

        """
        if name:
            subfolder = name + '_'
        else:
            subfolder = iterable.name + '_'
        if self._exists('naming_classes'):
            for step in listify(self.naming_classes):
                subfolder += getattr(iterable, step).step + '_'
        subfolder += str(iterable.number)
        setattr(sel.inventory, name, self.inventory.create_folder(
            folder = self.book,
            subfolder = subfolder))
        return self

    """ Composite Management Methods """

    def add_default_kwargs(self,
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

    def add_file_formats(self,
            file_format: str,
            extension: str,
            load_method: Union[Callable, str],
            save_method: Union[Callable, str]) -> None:
        """Adds or replaces a file extension option.

        Args:
            file_format (str): name of the file_format.
            extension (str): file extension (without period) to be used.
            load_method (Union[Callable, str]): a method to be used when loading
                files of the passed file_format or the string suffix (following
                '_transfer_') of an existing method.
            save_method (Union[Callable, str]): a method to be used when saving
                files of the passed file_format or the string suffix (following
                '_transfer_') of an existing method.

        """
        self.extensions.update({file_format: extension})
        try:
            setattr(
                self.importer,
                '_'.join(['_transfer_', file_format]),
                '_'.join(['_transfer_', load_method]))
        except TypeError:
            setattr(
                self.importer,
                '_'.join(['_transfer_', file_format]),
                load_method)
        try:
            setattr(
                self.exporter,
                '_'.join(['_transfer_', file_format]),
                '_'.join(['_transfer_', save_method]))
        except TypeError:
            setattr(
                self.exporter,
                '_'.join(['_transfer_', file_format]),
                save_method)
        return self

    def add_file_names(self,
            steps: Union[List[str], str],
            file_names: Union[List[str], str]) -> None:
        """Adds data file names for specific steps.

        Args:
            steps(Union[List[str], str]): step or step names.
            file_names(Union[List[str], str]): file name or file names (without
                extension(s)).

        """
        self.file_names.update(dict(zip(steps, file_names)))
        return self

    def add_folders(self,
            subfolders: Union[List[str], str],
            root_folder: Optional[str] = None) -> None:
        """Adds a list of subfolders to an existing root_folder.

        For every subfolder created, an attribute with the same name will
        also be created with its value corresponding to the full path of that
        new subfolder.

        Args:
            subfolders(Union[List[str], str]): subfolder name(s) to be created.
            root_folder (Optional[str]): path of folder where subfolders should
                be created. If None is passed, the value of the 'root'
                attribute is used.

        """
        if root_folder is None:
            root_folder = self.root
        for subfolder in listify(subfolders):
            temp_folder = self.inventory.add_folder(
                folder = root_folder,
                subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default folder and file settings."""
        self._check_root_folder()
        self.inventory = Inventory(library = self, idea = self.idea)
        self.importer = Importer(library = self, idea = self.idea)
        self.exporter = Exporter(library = self, idea = self.idea)
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
        self.add_folders(
            root_folder = self.root,
            subfolders = [self.data_folder, self.results_folder])
        # Creates list of default subfolders from 'data_folder' to create.
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        self.add_folders(
            root_folder = self.data,
            subfolders = self.data_subfolders)
        return self


@dataclass
class FilePath(Distributor):
    """Builds and contains a completed file_path.

    Args:
        distributor ('Distributor'): Importer or Exporter.
        stage (str): name of stage for determining default values.
        file_path (Optional[str]): full file path. Defaults to None.
        folder (Optional[str]): name of target folder (not used if 'file_path'
            passed). Defaults to None.
        file_name (Optional[str]): name of file (not used if 'file_path'
            passed). Defaults to None.
        file_format (Optional[str]): name of file format (not used if '
            file_path' passed). Defaults to None.

    """
    distributor: 'Distributor'
    stage: str
    file_path: Optional[str] = None
    folder: Optional[str] = None
    file_name: Optional[str] = None
    file_format: Optional[str] = None

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    def __str__(self) -> str:
        """Returns string name of 'state'."""
        return self.publish()

    """ Private Methods """

    def _check_file_name(self) -> str:
        """Selects 'file_name' or default values."""
        if not self.file_name:
            self.file_name = self.distributor.data_file_names[self.stage]
        return self

    def _check_file_format(self) -> str:
        """Selects 'file_format' or default value."""
        if not self.file_format:
            self.file_format = self.distributor.data_file_formats[self.stage]
        return self

    def _check_folder(self) -> str:
        """Selects 'folder' or default value."""
        if not self.folder:
            self.folder = self.distributor.data_folders[self.stage]
        else:
            try:
                self.folder = getattr(self, self.folder)
            except AttributeError:
                pass
        return self

    def _make_path(self):
        """Completes 'file_path' from component parts."""
        self.file_path = os.path.join(
            self.folder,
            '.'.join([self.file_name, self.extensions[self.file_format]]))
        return self

    """ Core siMpLify Methods """

    def draft(self):
        if self.file_path is None:
            for check in ('folder', 'file_name', 'file_format'):
                setattr(
                    self,
                    check,
                    getattr(self, '_'.join(['_check', check]))())
            self._make_path()
        return self

    def publish(self):
        return self.file_path


@dataclass
class Inventory(Distributor):
    """Manages folder structures for siMpLify.

    Args:
        library ('Library): parent Library class instance.
        idea ('Idea'): shared Idea instance.

    """
    library: 'Library'
    idea: 'Idea'

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Private Methods """

    def _make_folder(self, folder: str) -> None:
        """Creates folder if it doesn't already exist.

        Args:
            folder (str): the path of the folder.

        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    """ Public Tool Methods """

    def add_branch(self,
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

    def add_folder(self,
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

    def add_path(self,
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

    def add_tree(self, folder_tree: Dict[str, str]) -> None:
        """Adds folder tree to disk and adds corresponding attributes.

        Args:
            folder_tree (Dict[str, str]): a folder tree to be created with
                corresponding attributes to the library instance.

        """
        for folder, subfolders in folder_tree.items():
            self.add_branch(root_folder = folder, subfolders = subfolders)
        return self


@dataclass
class Importer(Distributor):
    """Manages file importing for siMpLify.

    Args:
        library ('Library): parent Library class instance.
        idea ('Idea'): shared Idea instance.

    """
    library: 'Library'
    idea: 'Idea'

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Private Methods """

    def _check_boolean_out(self,
            ingredients: Union[pd.Series, pd.DataFrame]) -> (
                Union[pd.Series, pd.DataFrame]):
        """Converts True/False to 1/0 if 'boolean_out' is False.

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

    def _transfer_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads csv file into a pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        additional_kwargs = ['encoding', 'index_col', 'header', 'usecols',
                             'low_memory']
        kwargs = self._check_kwargs(
            variables_to_check = additional_kwargs,
            passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_csv(file_path, **kwargs)
        return variable

    def _transfer_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads Excel file into a pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        additional_kwargs = ['index_col', 'header', 'usecols']
        kwargs = self._check_kwargs(
            variables_to_check = additional_kwargs,
            assed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'nrows': self.test_chunk})
        variable = pd.read_excel(file_path, **kwargs)
        return variable

    def _transfer_feather(self, file_path: str, **kwargs):
        """Loads feather file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        return pd.read_feather(file_path, nthreads = -1, **kwargs)

    def _transfer_h5(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads hdf5 with '.h5' extension into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        return self._transfer_hdf(file_path, **kwargs)

    def _transfer_hdf(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads hdf5 file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        additional_kwargs = ['columns']
        kwargs = self._check_kwargs(
            variables_to_check = additional_kwargs,
            passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize': self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns': kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_hdf(file_path, **kwargs)

    def _transfer_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Loads json file into pandas DataFrame.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        additional_kwargs = ['encoding', 'columns']
        kwargs = self._check_kwargs(
            variables_to_check = additional_kwargs,
            passed_kwargs = kwargs)
        if self.test_data and not 'chunksize' in kwargs:
            kwargs.update({'chunksize': self.test_rows})
        if 'usecols' in kwargs:
            kwargs.update({'columns': kwargs['usecols']})
            kwargs.pop('usecols')
        return pd.read_json(file_path = file_path, **kwargs)

    def _transfer_pickle(self, file_path: str, **kwargs) -> object:
        """Returns an unpickled python object.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        return pickle.load(open(file_path, 'rb'))

    def _transfer_png(self, file_path: str, **kwargs) -> NotImplementedError:
        """Although png files are saved by siMpLify, they cannot be loaded.

        Raises:
            NotImplementedError: if called.

        """
        error = 'loading .png files is not supported'
        raise NotImplementedError(error)

    def _transfer_text(self, file_path: str, **kwargs) -> str:
        """Loads text file with python reader.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        return self._transfer_txt(file_path = file_path, **kwargs)

    def _transfer_txt(self, file_path: str, **kwargs) -> str:
        """Loads text file with python reader.

        Args:
            file_path (str): complete file path of file.

        Returns:
            variable (str): string loaded from disk.

        """
        with open(file_path, mode = 'r', errors = 'ignore',
                  encoding = self.file_encoding) as a_file:
            return a_file.read()

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

    def iterate(self,
            chapters: List[str],
            ingredients: 'Ingredients' = None,
            return_ingredients: Optional[bool] = True):
        """Iterates through a list of files contained in self.batch and
        applies the chapters created by a book method (or subclass).
        Args:
            chapters(list): list of chapter types (Recipe, Harvest, etc.)
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
                for chapter in chapters:
                    data = chapter.produce(data = ingredients)
            if return_ingredients:
                return ingredients
            else:
                return self
        else:
            for file_path in self.batch:
                for chapter in chapters:
                    chapter.produce()
            return self

    """ Public Import/Export Methods """

    def transfer(self,
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
            return getattr(self, '_transfer_' + file_format)(
                file_path = file_path,
                **kwargs)
        elif isinstance(file_path, list):
            error = 'file_path is a glob list - use iterate instead'
            raise TypeError(error)
        else:
            return None

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default data folders, file names, and file formats linked to
        the various stages of the siMpLify process.

        """
        self.data_folders = {
            'sow': 'raw',
            'reap': 'raw',
            'clean': 'interim',
            'bale': 'interim',
            'deliver': 'interim',
            'chef': 'processed',
            'critic': 'processed'}
        self.data_file_names = {
            'sow': None,
            'harvest': None,
            'clean': 'harvested_data',
            'bale': 'cleaned_data',
            'deliver': 'baled_data',
            'chef': 'final_data',
            'critic': 'final_data'}
        self.data_file_formats = {
            'sow': 'source_format',
            'harvest': 'source_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'interim_format',
            'chef': 'final_format',
            'critic': 'final_format'}
        return self


@dataclass
class Exporter(Distributor):
    """Manages file exporting for siMpLify.

    Args:
        library ('Library): parent Library class instance.
        idea ('Idea'): shared Idea instance.

    """

    library: 'Library'
    idea: 'Idea'

    def __post_init__(self) -> None:
        self.draft()
        return self

    """ Private Methods """

    def _transfer_csv(self,
            variable: pd.Series,
            file_path: str,
            **kwargs) -> None:
        """Saves pandas Series to disk as .csv file.

        Args:
            variable (Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(
                variables_to_check = additional_kwargs,
                passed_kwargs = kwargs)
            variable.to_csv(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _transfer_excel(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disk as an Excel file.

        Args:
            variable (DataFrame or Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        if isinstance(variable, pd.DataFrame):
            additional_kwargs = ['index', 'header', 'encoding', 'float_format']
            kwargs = self._check_kwargs(
                variables_to_check = additional_kwargs,
                passed_kwargs = kwargs)
            variable.excel(file_path, **kwargs)
        elif isinstance(variable, pd.Series):
            self.writer.writerow(variable)
        return

    def _transfer_feather(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disk as a feather file.

        Args:
            variable (DataFrame or Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        variable.reset_index(inplace = True)
        variable.to_feather(file_path, **kwargs)
        return

    def _transfer_h5(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disk as a hdf file with .h5 extension.

        Args:
            variable (DataFrame or Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _transfer_hdf(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disk as a hdf file.

        Args:
            variable (DataFrame or Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        variable.to_hdf(file_path, **kwargs)
        return

    def _transfer_json(self,
            variable: Union[pd.DataFrame, pd.Series],
            file_path: str,
            **kwargs) -> None:
        """Saves pandas data object to disk as an json file.

        Args:
            variable (DataFrame or Series): variable to be saved to disk.
            file_path (str): complete file path of file.

        """
        variable.to_json(file_path, **kwargs)
        return

    def _transfer_pickle(self, variable: object, file_path: str, **kwargs):
        """Pickles file and saves it to disk.
        Args:
            variable (object): variable to be saved to disk.
            file_path (str): complete file path of file.
        """
        pickle.dump(variable, open(file_path, 'wb'))
        return

    def _transfer_png(self, variable: object, file_path: str, **kwargs) -> None:
        """Saves png file to disk.
        Args:
            variable (matplotlib object): variable to be saved to disk.
            file_path (str): complete file path of file.
        """
        variable.savefig(file_path, bbox_inches = 'tight')
        variable.close()
        return

    """ Public Import/Export Methods """

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

    def transfer(self,
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
        getattr(self, '_'.join(['_transfer_', file_format]))(
            variable, file_path, **kwargs)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default data folders, file names, and file formats linked to
        the various stages of the siMpLify process.

        """
        self.data_folders = {
            'sow': 'raw',
            'reap': 'interim',
            'clean': 'interim',
            'bale': 'interim',
            'deliver': 'processed',
            'chef': 'processed',
            'critic': 'recipe'}
        self.data_file_names = {
            'sow': None,
            'harvest': 'harvested_data',
            'clean': 'cleaned_data',
            'bale': 'baled_data',
            'deliver': 'final_data',
            'chef': 'final_data',
            'critic': 'predicted_data'}
        self.data_file_formats = {
            'sow': 'source_format',
            'harvest': 'interim_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'final_format',
            'chef': 'final_format',
            'critic': 'final_format'}
        return self
