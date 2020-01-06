"""
.. module:: inventory
:synopsis: file management made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import csv
from dataclasses import dataclass
from dataclasses import field
import datetime
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd

from simplify.core.base import SimpleDistributor
from simplify.core.base import SimpleContents
from simplify.core.base import SimplePath
from simplify.core.utilities import listify


@dataclass
class Inventory(SimpleContents):
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
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    idea: 'Idea' = None
    root_folder: Optional[Union[str, List[str]]] = field(
        default_factory = ['', ''])
    data_folder: Optional[str] = 'data'
    results_folder: Optional[str] = 'results'
    datetime_naming: Optional[bool] = True
    name: Optional[str] = 'files'

    def __post_init__(self) -> None:
        """Processes passed arguments to prepare class instance."""
        # Injects attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
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
        if attribute in ['add_tree', 'add_folders', 'make_batch']:
            return getattr(self.folders, attribute)
        elif attribute in ['add_format']:
            return getattr(self.pathifier.file_formats.add)
        elif attribute in ['load']:
            return getattr(self.importer, 'apply')
        elif attribute in ['save']:
            return getattr(self.exporter, 'apply')
        elif attribute in ['initialize_writer']:
            return getattr(self.exporter, attribute)
        elif attribute in ['add_default_kwargs']:
            return getattr(self.pathifier.kwargifier, 'add')

    """ Public Methods """

    def add_book_folder(self, prefix: Optional[str] = 'book') -> None:
        """Sets the book folder and corresponding attribute.

        Args:
            prefix (Optional[str]): either prefix to datetime naming or the
                book folder name. Defaults to 'book'.

        """
        if self.datetime_naming:
            self.active_book = '_'.join(
                [prefix, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
        else:
            self.active_book = prefix
        self.results.add_folders(
            root_folder = self.results['root'],
            subfolders = self.active_book)
        return self

    def add_chapter_folder(self,
            chapter: 'Chapter',
            name: Optional[str] = None) -> None:
        """Creates folder path for iterable-specific exports.

        Args:
            chapter (Chapter): an instance of SimpleBook.
            name (string): name of attribute for the folder path to be stored
                and the prefix of the folder to be created on disk.

        """
        if not name:
            name = 'chapter'
        self.active_chapter = ''.join([name, '_'])
        try:
            for step in listify(self.naming_classes):
                self.active_chapter += '_'.join(
                    [self.active_chapter, chapter.pages[step].technique])
        except AttributeError:
            pass
        subfolder = ''.join([subfolder, str(chapter.metadata['number'])])
        self.results.add_folders(
            root_folder = self.active_book,
            subfolders = self.active_chapter)
        return self

    def change_stage(self, new_stage: str) -> None:
        """Updates Inventory state for appropriate dict values to be chosen."""
        self.stage = new_stage
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates default folder and file settings."""
        self.root_folder = Path(self.root_folder)
        self.data_folder = self.root_folder.joinpath(self.data_folder)
        self.results_folder = self.root_folder.joinpath(self.results_folder)
        return self

    def publish(self) -> None:
        """Creates core folder trees and file options."""
        self.folders = Folders(
            root_folder = self.results_folder,
            subfolders = {
                self.results_folders: [],
                self.data_folders: [self.data_subfolders]},
            related = self)
        self.data_importer = Importer(
            file_names = self.import_file_names,
            folders = self.import_data_folders,
            file_formats = self.import_file_formats,
            related = self)
        self.data_exporter = Exporter(
            file_names = self.export_file_names,
            folders = self.export_data_folders,
            file_formats = self.export_file_formats,
            related = self)
        self.file_formats = FileFormats(related = self)
        self.pathifier = Pathifier(related = self)
        self.kwargifier = Kwargifier(related = self)
        return self

    def apply(self, action: str, **kwargs) -> Callable:
        """Leverages __getattr__ delegation to allow 'apply' method to be used.
        """
        return getattr(self, action)(**kwargs)


@dataclass
class DataFolders(SimplePath):
    """Creates and stores data folder paths.

    Args:
        inventory ('Inventory): related Inventory instance.
        folder (str): folder where 'names' are or should be.
        names (Dict[str, str]): dictionary where keys are names of states and
            values are Path objects linked to those states.

    """
    inventory: 'Inventory'
    folder: str
    names: Dict[str, str]

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        self.active = 'subfolders'
        self.data_subfolders = ['raw', 'interim', 'processed', 'external']
        super().__post_init__()
        self.publish()
        return self

    """ Private Methods """

    def _validate(self) -> None:
        """Validates type of passed 'bundle' argument.

        Raises:
            TypeError: if 'bundle' is neither a dictionary nor NestedBundle
                instance or subclass.

        """
        if issubclass(self.bundle, NestedBundle):
            self = self.bundle
        elif not isinstance(self.bundle, (list, dict, str)):
            raise TypeError(
                'bundle must be a dict, list, str, or SimpleContents type')
        return self

    def _pathlibify(self, path: Union[str, Path]) -> Path:
        """Converts string 'path' to pathlib Path object.

        Args:
            path (Union[str, Path]): either a string representation of a path,
                a key to the active stored dictionary, or a Path object.

        Returns:
            Path object.

        """
        try:
            path = getattr(self, self.state)[path]
        except (ValueError, TypeError, KeyError):
            if isinstance(path, str):
                return Path(path)
            elif isinstance(path, Path):
                return path
            else:
                raise TypeError('path must be str or Path type')

    def _publish_path(self, path: Union[str, Path]) -> None:
        """Finalizes, stores, and writes folder path.

        Args:
            path (Union[str, Path]): either a string representation of a path,
                a key to the active stored dictionary, or a Path object.

        """
        pathlib_path = self._pathlibify(path = path)
        getattr(self, self.state)[pathlib_path.parts[-1]] = pathlib_path
        self._write_folder(folder = pathlib_path)
        return self

    def _write_folder(self, folder: Path) -> None:
        """Creates folder if it doesn't already exist.

        Args:
            folder (Path): the path of the folder.

        """
        path.mkdir(parents = True, exist_ok = True)
        return self

    """ Public Methods """

    def add_folders(self,
            root_folder: Union[str, Path],
            subfolders: Union[List[str], Dict[str, str], str]) -> None:
        """Adds a list of subfolders to an existing root_folder.

        For every subfolder created, an attribute with the same name will
        also be created with its value corresponding to the full path of that
        new subfolder.

        Args:
            root_folder (Union[str, Path]): path of folder where subfolders
                should be created or name of attribute containing path of a
                folder.
            subfolders (Union[List[str], Dict[str, str], str]): subfolder
                name(s) to be created.

        """
        root = self._pathlibify(path = root_folder)
        if isinstance(subfolders, dict):
            self.add_tree(root_folder = root, subfolders = subfolders)
        elif isinstance(subfolders, list) or isinstance(subfolders, str):
            self.add_subfolders(root_folder = root, subfolders = subfolders)
        else:
            raise TypeError('subfolders must be list, dict, or str type')
        return self

    def add_subfolders(self,
            root_folder: Union[str, Path],
            subfolders: Union[List[str], str]) -> None:
        """Creates a set of 'subfolders' off of 'root_folder'.

        Each created folder name is also stored as a local attribute with the
        same name as the created folder.

        Args:
            root_folder (str): the folder from which the tree branch should be
                created.
            subfolders (Union[List[str], str]): subfolder names to be created
                off of 'root_folder'.

        """
        root = self._pathlibify(path = root_folder)
        self._publish_path(path = root)
        for subfolder in listify(subfolders):
            pathlib_subfolder = self._pathlibify(path = subfolder)
            self._publish_path(path = root.joinpath(pathlib_subfolder))
        return self

    def add_tree(self,
            root_folder: Union[str, Path],
            subfolders: Dict[str, Union[str, Dict]]) -> None:
        """Adds folder tree to disk and adds corresponding attributes.

        Args:
            root_folder (Union[str, Path]): path of folder where subfolders
                should be created or name of attribute containing path of a
                folder.
            subfolders (Dict[str, Union[str, Dict]]): a folder tree to be
                created with corresponding attributes to the inventory instance.

        """
        root = self._pathlibify(path = root_folder)
        self._publish_path(path = root)
        for root_folder, folders in subfolders.items():
            if isinstance(folders, dict):
                self.add_tree(root_folder = root_folder, subfolders = folders)
            else:
                self.add_subfolders(
                    root_folder = root,
                    subfolders = folders)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates root folder for instance."""
        self.root_folder = self._pathlibify(path = self.root_folder)
        getattr(self, self.state)['root'] = self.root_folder
        self._publish_path(path = self.root_folder)
        return self

    def publish(self) -> None:
        """Creates core folder tree."""
        self.add_folders(
            root_folder = self.root_folder,
            subfolders = self.bundle)
        return self


@dataclass
class Pathifier(object):
    """Builds completed file_paths.

    Args:
        related ('Inventory'): related Inventory instance.

    """
    related: 'Inventory' = None

    def __post_init__(self) -> None:
        return self

    """ Private Methods """

    def _set_folder(self,
            distributor: 'SimpleDistributor',
            file_format: Optional[str] = None) -> Path:
        """Selects 'folder' or default value.

        Args:
            folder (Optional[str]): name of target folder. Defaults to None.

        Returns:
            str: completed folder.

        """
        if not folder:
            folder = distributor.folders[self.related.stage]
        else:
            try:
                folder = self.related.folders[folder]
            except AttributeError:
                pass
        return folder

    def _set_file_name(self,
            distributor: 'SimpleDistributor',
            file_format: Optional[str] = None) -> str:
        """Selects 'file_name' or default values.

        Args:
            file_name (Optional[str]): name of file. Defaults to None.

        Returns:
            str: completed file_name.

        """
        if not file_name:
            file_name = distributor.file_names[self.related.stage]
        return file_name

    def _set_file_format(self,
            distributor: 'SimpleDistributor',
            file_format: Optional[str] = None) -> str:
        """Selects 'file_format' or default value.

        Args:
            file_format (Optional[str]): name of file format. Defaults to None.

        Returns:
            str: completed file_format.

        """
        if not file_format:
            file_format = distributor.file_formats[self.related.stage]
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
        return folder.joinpath(''.join(
            [file_name, self.file_formats[file_format].extension]))

    """ Core siMpLify Methods """

    def apply(self,
            distributor: 'SimpleDistributor',
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> Path:
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
        if isinstance(file_path, Path):
            return file_path
        elif isinstance(file_path, str):
            return Path(file_path)
        else:
            return self._make_path(
                folder = self._set_folder(
                    distributor = distributor,
                    folder = folder),
                file_name = self._set_file_name(
                    distributor = distributor,
                    file_name = file_name),
                file_format = self._set_file_format(
                    distributor = distributor,
                    file_format = file_format))

@dataclass
class Kwargifier(object):
    """Builds completed file_paths.

    Args:
        related ('Inventory'): related Inventory instance.

    """
    related: 'Inventory' = None

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
        self.library(dict(zip(kwargs, settings)))
        return self


@dataclass
class Importer(SimpleDistributor):
    """Manages file importing for siMpLify.

    Args:
        inventory ('Inventory'): related Inventory instance.

    """
    inventory: 'Inventory' = None

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_defaults(self) -> None:
        self.data_folders = SimplePath(
            inventory = self,
            folder = self.inventory.data_folder,
            names = {
                'sow': 'raw',
                'reap': 'raw',
                'clean': 'interim',
                'bale': 'interim',
                'deliver': 'interim',
                'chef': 'processed',
                'actuary': 'processed',
                'critic': 'processed',
                'artist': 'processed'})
        self.results_folders = SimplePath(
            inventory = self,
            folder = self.inventory.results_folder,
            names = {
                'book': 'book',
                'chapter': 'chapter'})
        self.file_names = {
            'sow': None,
            'harvest': None,
            'clean': 'harvested_data',
            'bale': 'cleaned_data',
            'deliver': 'baled_data',
            'chef': 'final_data',
            'critic': 'final_data'}
        self.file_formats = {
            'sow': 'source_format',
            'harvest': 'source_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'interim_format',
            'chef': 'final_format',
            'critic': 'final_format'}
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
            include_subfolders (Optional[bool]): whether to include files in
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
                globals()[self.library[file_format.associated_package]],
                file_format.import_method)(
                    file_path, **kwargs)
        except AttributeError:
            raise AttributeError(' '.join(
                [file_format.import_method,
                 'is not a recognized import method']))


@dataclass
class Exporter(SimpleDistributor):
    """Manages file exporting for siMpLify.

    Args:
        related ('Inventory'): related Inventory instance.

    """
    related: 'Inventory' = None

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

    def _draft_defaults(self) -> None:
        self.folders = {
            'sow': 'raw',
            'reap': 'interim',
            'clean': 'interim',
            'bale': 'interim',
            'deliver': 'processed',
            'chef': 'processed',
            'critic': 'recipe'}
        self.file_names = {
            'sow': None,
            'harvest': 'harvested_data',
            'clean': 'cleaned_data',
            'bale': 'baled_data',
            'deliver': 'final_data',
            'chef': 'final_data',
            'critic': 'predicted_data'}
        self.file_formats: {
            'sow': 'source_format',
            'harvest': 'interim_format',
            'clean': 'interim_format',
            'bale': 'interim_format',
            'deliver': 'final_format',
            'chef': 'final_format',
            'critic': 'final_format'}
        return self

    """ Public Methods """

    def initialize_writer(self,
            file_path: str,
            columns: List[str],
            encoding: Optional[str] = None,
            dialect: Optional[str] = 'excel') -> None:
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
class FileFormat(object):
    """File format container."""

    name: Optional[str] = 'file_format'
    extension: Optional[str] = None
    import_method: Optional[str] = None
    export_method: Optional[str] = None
    addtional_kwargs: Optional[List[str]] = None
    required: Optional[Dict[str, Any]] = None
    test_size_parameter: Optional[str] = None


@dataclass
class FileFormats(SimpleContents):
    """Creates and stores file formats and file extensions.

    Args:
        related ('Inventory'): related Inventory instance.

    """
    related: 'Pathifier' = None

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        super().__post_init__()
        return self

    """ Core siMpLify Methods """
    def draft(self) -> None:
        self.bundle = {
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
                export_method = '_export_text')}
        return self