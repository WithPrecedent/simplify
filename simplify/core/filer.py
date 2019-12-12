"""
.. module:: inventory
:synopsis: file management made simple
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from simplify.core.options import ManuscriptOptions
from simplify.core.outline import Outline
from simplify.core.defaults import Defaults
from simplify.core.utilities import listify


@dataclass
class Inventory(object):
    """Manages files and folders for siMpLify.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and provides methods for loading and saving siMpLify,
    pandas, and numpy objects.

    Args:
        root_folder (Optional[str]): the complete path from which the other
            paths and folders used by Inventory should be created. Defaults to
            the parent folder or the parent folder of the current working
            directory.
        data_folder (Optional[str]): the data subfolder name or a complete path
            if the 'data_folder' is not off of 'root_folder'. Defaults to
            'data'.
        results_folder (Optional[str]): the results subfolder name or a complete
            path if the 'results_folder' is not off of 'root_folder'. Defaults
            to 'results'.
        datetime_naming (Optional[bool]): whether the date and time should be
            used to create Book subfolders (so that prior results are not
            overwritten). Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: 'Idea'
    root_folder: Optional[str] = None
    data_folder: Optional[str] = None
    results_folder: Optional[str] = None
    datetime_naming: Optional[bool] = True
    auto_publish: Optional[bool] = True
    name: Optional[str] = 'files'

    def __post_init__(self) -> None:
        """Processes passed arguments to prepare class instance."""
        self = self.idea.apply(instance = self)
        self.defaults = Defaults()
        self = self.defaults.apply(instance = self)
        self.draft()
        if self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Passes specific 'attribute' to appropriate composite classes.

        Args:
            attribute (str): method or attribute sought.

        Raises:
            AttributeError: if 'attribute' is not found.

        Returns:
            Any: method or attribute from composite class.

        """
        if attribute in [
                'add_branch',
                'add_tree',
                'add_folder',
                'make_batch',
                'set_book_folder',
                'set_chapter_folder']:
            return getattr(self.foldifier, attribute)
        elif attribute in ['add_format']:
            return getattr(self.formatifier, attribute)
        elif attribute in ['load']:
            return getattr(self.importer, attribute)
        elif attribute in ['save', 'initialize_writer']:
            return getattr(self.exporter, attribute)
        elif attribute in ['add_default_kwargs']:
            return getattr(self.kwargifier, attribute)
        else:
            return super().__getattr__(attribute = attribute)

    """ Private Methods """

    def _check_root_folder(self) -> None:
        """Checks if 'root_folder' exists on disk. If not, it is created."""
        try:
            if os.path.isdir(self.root_folder):
                pass
            else:
                self.root_folder = os.path.abspath(self.root_folder)
        except TypeError:
            try:
                self.root_folder = os.path.join(self.root_folder)
            except TypeError:
                self.root_folder = os.path.join('..', '..')
        return self

    def _draft_options(self):
        self._options = ManuscriptOptions(options = {
            'data': 'data',
            'book': 'book',
            'chapter': 'chapter'})
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default folder and file settings."""
        self._check_root_folder()
        self._draft_options()
        self.folderifier = Folderifier(inventory = self)
        self.nameifier = Nameifier(inventory = self)
        self.formatifier = Formatifier(inventory = self)
        self.pathifier = Pathifier(inventory = self)
        self.kwargifier = Kwargifier(inventory = self)
        return self

    def publish(self) -> None:
        """Creates core folder tree."""
        self.folderifier.add_folders(
            subfolders = [self.data_folder, self.results_folder])
        # Creates list of default subfolders from 'data_folder' to create.
        self.folderifier.add_folders(
            subfolders = self.data_subfolders,
            root_folder = self.data)
        self.importer = Importer(inventory = self)
        self.exporter = Exporter(inventory = self)
        return self

    def apply(self, instance: object) -> None:
        """Injects Inventory instance into passed 'instance'.

        The attribute name will be the same as Inventory's 'name' attribute.

        Args:
            instance (object): instance for Inventory instance to be injected.

        Returns:
            instance (object): instance with Inventory instance injected.

        """
        setattr(instance, self.name, self)
        return instance


@dataclass
class Folderifier(object):
    """Builds folders and, if necessary, writes them to disk.

    Args:
        inventory ('Inventory'): Inventory instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'folderifier'

    def __post_init__(self) -> None:
        return self

    """ Private Methods """

    def _write_folder(self, folder: str) -> None:
        """Creates folder if it doesn't already exist.

        Args:
            folder (str): the path of the folder.

        """
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    """ Public Methods """

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
            temp_folder = self.make_folder(
                folder = root_folder,
                subfolder = subfolder)
            setattr(self, subfolder, temp_folder)
            root_folder = temp_folder
        return self


    def add_folders(self,
            subfolders: Union[List[str], str],
            root_folder: Optional[str] = None) -> None:
        """Adds a list of subfolders to an existing root_folder.

        For every subfolder created, an attribute with the same name will
        also be created with its value corresponding to the full path of that
        new subfolder.

        Args:
            subfolders (Union[List[str], str]): subfolder name(s) to be created.
            root_folder (Optional[str]): path of folder where subfolders should
                be created or name of attribute containing path of a folder. If
                not passed, the value of the 'root_folder' attribute is used.

        """
        if root_folder is None:
            root_folder = self.options['root']
        else:
            try:
                root_folder = getattr(self, root_folder)
            except (TypeError, AttributeError):
                pass
        for subfolder in listify(subfolders):
            new_folder = os.path.join(root_folder, subfolder)
            setattr(self, subfolder, os.path.join(root_folder, subfolder))
        return self

    def add_tree(self, folder_tree: Dict[str, str]) -> None:
        """Adds folder tree to disk and adds corresponding attributes.

        Args:
            folder_tree (Dict[str, str]): a folder tree to be created with
                corresponding attributes to the inventory instance.

        """
        for folder, subfolders in folder_tree.items():
            self.add_branch(root_folder = folder, subfolders = subfolders)
        return self

    def set_book_folder(self, prefix: Optional[str] = 'book') -> None:
        """Sets the book folder and corresponding attribute.

        Args:
            prefix (Optional[str]): either prefix to datetime naming or the
                book folder name. Defaults to 'book'.

        """
        if self.datetime_naming:
            subfolder = '_'.join(
                [prefix, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
        else:
            subfolder = prefix
        self.book = self.make_folder(
            folder = self.results_folder,
            subfolder = subfolder)
        return self

    def set_chapter_folder(self,
            chapter: 'Chapter',
            name: Optional[str] = None) -> None:
        """Creates folder path for iterable-specific exports.

        Args:
            chapter (Chapter): an instance of SimplePackage.
            name (string): name of attribute for the folder path to be stored
                and the prefix of the folder to be created on disk.

        """
        if not name:
            name = chapter.name
        subfolder = ''.join([name, '_'])
        try:
            for step in listify(self.naming_classes):
                subfolder += '_'.join([subfolder, chapter.pages[step].name])
        except AttributeError:
            pass
        subfolder = ''.join([subfolder, str(chapter.metadata['number'])])
        self.chapter = self.make_folder(
            folder = self.book,
            subfolder = subfolder)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets core default folders"""
        self._options = ManuscriptOptions(options = {
            'root': self.inventory.root_folder,
            'data': self.inventory.data_folder,
            'results': self.inventory.results_folder})
        return self

    def apply(self, folder):
        try:
            return self.options[folder]
        except KeyError:
            return folder

@dataclass
class Nameifier(object):
    """Builds file_names from passed arguments or default options.

    Args:
        inventory ('Inventory'): Inventory instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'nameifier'

    def __post_init__(self) -> None:
        return self


@dataclass
class Formatifier(object):
    """Sets appropriate file_format and file extension.

    Args:
        inventory ('Inventory'): Inventory instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'formatifier'

    """ Core siMpLify Methods """
    def draft(self):
        self._options = ManuscriptOptions(options = {
            'csv': FileFormat(
                name = 'csv',
                extension = '.csv',
                import_method = 'read_csv',
                export_method = 'to_csv',
                additional_kwargs = [
                    'encoding',
                    'index_col',
                    'header',
                    'usecols',
                    'low_memory'],
                test_size_parameter = 'nrows'),
            'excel': FileFormat(
                name = 'excel',
                extension = '.xlsx',
                import_method = 'read_excel',
                export_method = 'to_excel',
                additional_kwargs = ['index_col', 'header', 'usecols'],
                test_size_parameter = 'nrows'),
            'feather': FileFormat(
                name = 'feather',
                extension = '.feather',
                import_method = 'read_feather',
                export_method = 'to_feather',
                required = {'nthreads': -1}),
            'hdf': FileFormat(
                name = 'hdf',
                extension = '.hdf',
                import_method = 'read_hdf',
                export_method = 'to_hdf',
                additional_kwargs = ['columns'],
                test_size_parameter = 'chunksize'),
            'json': FileFormat(
                name = 'json',
                extension = '.json',
                import_method = 'read_json',
                export_method = 'to_json',
                additional_kwargs = ['encoding', 'columns'],
                test_size_parameter = 'nrows'),
            'pickle': FileFormat(
                name = 'pickle',
                extension = '.pickle',
                import_method = '_pickle_object',
                export_method = '_unpickle_object'),
            'png': FileFormat(
                name = 'png',
                extension = '.png',
                export_method = 'save_fig',
                required = {'bbox_inches': 'tight', 'format': 'png'}),
            'text': FileFormat(
                name = 'text',
                extension = '.txt',
                import_method = '_import_text',
                export_method = '_export_text'),}
        return self


@dataclass
class Pathifier(object):
    """Builds completed file_paths.

    Args:
        inventory ('Inventory'): Inventory instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'formatifier'

    def __post_init__(self) -> None:
        return self

    """ Private Methods """

    def _set_folder(self, folder: Optional[str] = None) -> str:
        """Selects 'folder' or default value.

        Args:
            folder (Optional[str]): name of target folder. Defaults to None.

        Returns:
            str: completed folder.

        """
        if not folder:
            folder = self.inventory.data_folders[self.inventory.stage]
        else:
            try:
                folder = getattr(self.inventory, folder)
            except AttributeError:
                pass
        return folder

    def _set_file_name(self, file_name: Optional[str] = None) -> str:
        """Selects 'file_name' or default values.

        Args:
            file_name (Optional[str]): name of file. Defaults to None.

        Returns:
            str: completed file_name.

        """
        if not file_name:
            file_name = self.inventory.data_file_names[self.inventory.stage]
        return file_name

    def _set_file_format(self, file_format: Optional[str] = None) -> str:
        """Selects 'file_format' or default value.

        Args:
            file_format (Optional[str]): name of file format. Defaults to None.

        Returns:
            str: completed file_format.

        """
        if not file_format:
            file_format = self.inventory.data_file_formats[self.inventory.stage]
        return file_format

    def _make_path(self,
            folder: str,
            file_name: str,
            file_format: str) -> str:
        """Creates completed file_path from passed arguments.

        Args:
            folder (str): name of target folder.
            file_name (str): name of file.
            file_format (str): name of file format.

        Returns:
            str: completed file path.

        """
        return os.path.join(folder,'.'.join(
            [file_name, self.inventory.extensions[file_format]]))

    """ Core siMpLify Methods """

    def apply(self,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> str:
        """Creates file path from passed arguments.

        Args:
            file_path (Optional[str]): full file path. Defaults to None.
            folder (Optional[str]): name of target folder (not used if
                'file_path' passed). Defaults to None.
            file_name (Optional[str]): name of file (not used if 'file_path'
                passed). Defaults to None.
            file_format (Optional[str]): name of file format (not used if '
                file_path' passed). Defaults to None.

        Returns:
            str of completed file path.

        """
        if file_path:
            return file_path
        else:
            return self._make_path(
                folder = self._set_folder(folder = folder),
                file_name = self._set_file_name(file_name = file_name),
                file_format = self._set_file_format(file_format = file_format))

@dataclass
class Kwargifier(object):
    """Builds completed file_paths.

    Args:
        inventory ('Inventory'): Inventory instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'kwargtifier'

    def __post_init__(self) -> None:
        return self

    """ Public Methods """

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
        self.options(dict(zip(kwargs, settings)))
        return self

@dataclass
class FileName(object):
    """File name container."""



@dataclass
class DataFileNames(object):

    def draft(self):


@dataclass
class Distributor(ABC):
    """Base class for siMpLify Importer and Exporter."""

    def __post_init__(self):
        self.draft()
        if self.auto_publish:
            self.publish()
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
                        {variable: self.inventory.default_kwargs[variable]})
                elif hasattr(self, variable):
                    new_kwargs.update({variable: getattr(self, variable)})
        return new_kwargs

    """ Composite Management Methods """

    """ Core siMpLify Methods """

    def draft(self) -> None:
        self._options = ManuscriptOptions(options = {
            'csv': 'csv',
            'matplotlib': 'mp',
            'pandas': 'pd',
            'pickle': 'pickle'}
        return self


@dataclass
class Importer(Distributor):
    """Manages file importing for siMpLify.

    Args:
        inventory ('Inventory): parent Inventory class instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'import'
    auto_publish: Optional[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Public Methods """

    def make_batch(self,
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

    def iterate_batch(self,
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

    """ Core siMpLify Methods """

    def apply(self, file_path: str, file_format: 'FileFormat', **kwargs) -> Any:
        """Imports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path (str): a complete file path for the file to be loaded.
            file_format ('FileFormat'): object with information about how the
                file should be loaded.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Any: depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        """
        try:
            return getattr(
                globals()[self.options[file_format.associated_package]],
                file_format.import_method)(
                    file_path, **kwargs)
        except AttributeError:
            raise AttributeError(' '.join(
                [file_format.import_method,
                 'is not a recognized import method']))


@dataclass
class Exporter(Distributor):
    """Manages file exporting for siMpLify.

    Args:
        inventory ('Inventory): parent Inventory class instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.

    """
    inventory: 'Inventory'
    name: Optional[str] = 'export'
    auto_publish: Optional[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _check_boolean_out(self,
            data: Union[pd.Series, pd.DataFrame]) -> (
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

    """ Public Methods """

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

    def iterate_writer(self):
        return self

    """ Core siMpLify Methods """

    def apply(self,
            variable: Any,
            file_path: str,
            file_format: 'FileFormat',
            **kwargs) -> None:
        """Exports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable (Any): the variable being exported.
            file_path (str): a complete file path for the file to be saved.
            file_format ('FileFormat'): object with information about how the
                file should be saved.
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        """
        # Changes boolean values to 1/0 if self.boolean_out = False
        if file_format.associated_package in ['pandas']:
            variable = self._check_boolean_out(variable = variable)
        try:
            getattr(variable, file_format.export_method)(file_path, **kwargs)
        except AttributeError:
            raise AttributeError(' '.join(
                [file_format.export_method,
                 'is not a recognized export method']))
        return self


@dataclass
class SimpleInventory(ABC):
    """Base class for storing and creating file paths."""

    """ Public Methods """

    def load(self,
            name: Optional[str] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Loads object from file into the subclass attribute 'name'.

        For any arguments not passed, default values stored in the shared
        Inventory instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            name (Optional[str]): name of attribute for the file contents to be
                stored. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                loaded. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be loaded from (not used if file_path is passed). Defaults to
                None.
            file_name (Optional[str]): contains the name of the file to be
                loaded without the file extension (not used if file_path is
                passed). Defaults to None.
            file_format (Optional[str]): name of file format in
                inventory.extensions. Defaults to None.

        """
        setattr(self, name, self.inventory.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format))
        return self

    def save(self,
            variable: Optional[Union['Manuscript', str]] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Exports a variable or attribute to disk.

        If 'variable' is not passed, 'self' will be used.

        For other arguments not passed, default values stored in the shared
        inventory instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            variable (Optional[Union['Manuscript'], str]): a python object
                or a string corresponding to a subclass attribute which should
                be saved to disk. Defaults to None.
            file_path (Optional[str]): a complete file path for the file to be
                saved. Defaults to None.
            folder (Optional[str]): a path to the folder where the file should
                be saved (not used if file_path is passed). Defaults to None.
            file_name (Optional[str]): contains the name of the file to be saved
                without the file extension (not used if file_path is passed).
                Defaults to None.
            file_format (Optional[str]): name of file format in
                inventory.extensions. Defaults to None.

        """
        # If variable is not passed, the subclass instance is saved.
        if variable is None:
            variable = self
        # If a string, 'variable' is converted to a local attribute with the
        # string as its name.
        else:
            try:
                variable = getattr(self, variable)
            except TypeError:
                pass
        self.inventory.save(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        return self


@dataclass
class FileFormat(object):
    """File format container."""

    name: Optional[str] = 'file_format'
    extension: Optional[str] = None
    import_method: Optional[str] = None
    export_method: Optional[str] = None
    addtional_kwargs: Optional[List[str]] = None
    required: Optional[Dict[str, Any]] = None
    test_size_parameter: Optional[str] = None