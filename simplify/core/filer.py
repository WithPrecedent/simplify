"""
.. module:: filer
:synopsis: data science file management made simple
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from abc import ABC
from collections.abc import MutableMapping
import csv
from dataclasses import dataclass
from dataclasses import field
import datetime
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import pandas as pd

from simplify.core.base import SimpleComponent
from simplify.core.states import create_states
from simplify.core.utilities import datetime_string
from simplify.core.utilities import deduplicate
from simplify.core.utilities import listify


@dataclass
class Filer(MutableMapping):
    """Manages files and folders for siMpLify.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and provides methods for loading and saving siMpLify,
    pandas, and numpy objects.

    Args:
        idea ('Idea'): an Idea instance with file-management related settings.
        root_folder (Optional[str]): the complete path from which the other
            paths and folders used by Filer should be created. Defaults to
            None. If not passed, the parent folder of the parent folder of the
            current working directory is used.
        data_folder (Optional[str]): the data subfolder name or a complete path
            if the 'data_folder' is not off of 'root_folder'. Defaults to
            'data'.
        data_subfolders (Optional[List[str]]): subfolders for data to be
            saved and loaded at different stages of a siMpLify project. Defaults
            to None. If not provided, the CookieCutter data science list is used
            (https://drivendata.github.io/cookiecutter-data-science/): ['raw',
            'interim', 'processed', 'external'].
        results_folder (Optional[str]): the results subfolder name or a complete
            path if the 'results_folder' is not off of 'root_folder'. Defaults
            to 'results'.
        states (Optional[Union[List[str], 'SimpleState']]): diffrent Project
            states or a 'SimpleState' instance. Defaults to None.

    """
    root_folder: Optional[Union[str, List[str]]] = field(
        default_factory = ['..', '..'])
    data_folder: Optional[str] = field(default_factory = lambda: 'data')
    data_subfolders: Optional[List[str]] = field(default_factory = list)
    results_folder: Optional[str] = field(default_factory = lambda: 'results')
    states: Optional[Union[List[str], 'SimpleState']] = None
    idea: Optional['Idea'] = None

    def __post_init__(self) -> None:
        """Creates initial attributes."""
        # Uses defaults for 'data_subfolders, if not passed.
        if not self.data_subfolders:
            self.data_subfolders = [
                'raw',
                'interim',
                'processed',
                'external']
        # Injects attributes from 'idea'.
        self.idea_sections = ['files']
        self = self.idea.apply(instance = self)
        # Initializes internal 'folders' dictionary to which dunder access
        # methods are directed.
        self.folders = {}
        # Automatically calls 'draft' method to complete initialization.
        self.draft()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            filer: Optional[Union[str, Path, List[str]]] = None,
            idea: Optional['Idea'] = None,
            **kwargs) -> 'Filer':
        """Creates an Filer instance from passed arguments.

        Args:
            filer (Optional[Union[str, Path, List[str]]]): Filer
                instance or root folder for one.
            idea (Optional['Idea']): an Idea instance.

        Returns:
            Filer: instance, properly configured.

        Raises:
            TypeError if filer is neither an Filer instance, string
                folder path, nor list to create a folder path.

        """
        if isinstance(filer, Filer):
            return filer
        elif isinstance(filer, (str, Path, List)):
            return cls(root_folder = filer, idea = idea, **kwargs)
        elif filer is None:
            return cls(idea = idea, **kwargs)
        else:
            raise TypeError('filer must be Filer type or folder path')

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Path:
        """Returns value for 'key' in 'folders'.

        Args:
            key (str): name of key in 'folders'.

        Returns:
            Path: item stored as a 'folders'.

        Raises:
            KeyError: if 'key' is not in 'folders'.

        """
        try:
            return self.folders[key]
        except KeyError:
            raise KeyError(' '.join([key, 'is not found in Filer']))

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' entry in 'folders'.

        Args:
            key (str): name of key in 'folders'.

        """
        try:
            del self.folders[key]
        except KeyError:
            pass
        return self

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets 'key' in 'folders' to 'value'.

        Args:
            key (str): name of key in 'folders'.
            value (Any): value to be paired with 'key' in 'folders'.

        """
        self.folders[key] = value
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'folders'.

        Returns:
            Iterable stored in 'folders'.

        """
        return iter(self.folders.items())

    def __len__(self) -> int:
        """Returns length of 'folders'.

        Returns:
            Integer: length of 'folders'.

        """
        return len(self.folders)

    """ Private Methods """

    def _draft_root(self) -> None:
        """Turns 'root_folder' into a Path object."""
        self.root_folder = self.root_folder or ['..', '..']
        if isinstance(self.root_folder, list):
            root = Path.cwd()
            for item in self.root_folder:
                root = root.joinpath(item)
            self.folders['root'] = root
        else:
            self.folders['root'] = Path(self.root_folder)
        return self

    def _draft_core_folders(self) -> None:
        """Drafts initial data and results folders."""
        self.folders['results'] = self.folders['root'].joinpath(
            self.results_folder)
        self._write_folder(folder = self.folders['results'])
        self.folders['data'] = self.folders['root'].joinpath(
            self.data_folder)
        self._write_folder(folder = self.folders['data'])
        for folder in self.data_subfolders:
            self.folders['folder'] = self.folders['data'].joinpath(folder)
            self._write_folder(folder = self.folders['folder'])
        return self

    def _draft_file_formats(self) -> None:
        """Drafts supported file formats and state-related mappings."""
        self.file_formats = {
            'csv': FileFormat(
                name = 'csv',
                module = 'pandas',
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
                module = 'pandas',
                extension = '.xlsx',
                import_method = 'read_excel',
                export_method = 'to_excel',
                additional_kwargs = ['index_col', 'header', 'usecols'],
                test_size_parameter = 'nrows'),
            'feather': FileFormat(
                name = 'feather',
                module = 'pandas',
                extension = '.feather',
                import_method = 'read_feather',
                export_method = 'to_feather',
                required = {'nthreads': -1}),
            'hdf': FileFormat(
                name = 'hdf',
                module = 'pandas',
                extension = '.hdf',
                import_method = 'read_hdf',
                export_method = 'to_hdf',
                additional_kwargs = ['columns'],
                test_size_parameter = 'chunksize'),
            'json': FileFormat(
                name = 'json',
                module = 'pandas',
                extension = '.json',
                import_method = 'read_json',
                export_method = 'to_json',
                additional_kwargs = ['encoding', 'columns'],
                test_size_parameter = 'nrows'),
            'stata': FileFormat(
                name = 'stata',
                module = 'pandas',
                extension = '.dta',
                import_method = 'read_stata',
                export_method = 'to_stata',
                test_size_parameter = 'chunksize'),
            'text': FileFormat(
                name = 'text',
                module = None,
                extension = '.txt',
                import_method = '_import_text',
                export_method = '_export_text'),
            'png': FileFormat(
                name = 'png',
                module = 'seaborn',
                extension = '.png',
                export_method = 'save_fig',
                required = {'bbox_inches': 'tight', 'format': 'png'}),
            'pickle': FileFormat(
                name = 'pickle',
                module = None,
                extension = '.pickle',
                import_method = '_pickle_object',
                export_method = '_unpickle_object')}
        self.import_format_states = {
            'acquire': 'source_format',
            'parse': 'source_format',
            'clean': 'interim_format',
            'merge': 'interim_format',
            'deliver': 'interim_format',
            'analyze': 'final_format',
            'summarize': 'final_format',
            'criticize': 'final_format',
            'visualize': 'final_format'}
        self.export_format_states = {
            'acquire': 'source_format',
            'parse': 'interim_format',
            'clean': 'interim_format',
            'merge': 'interim_format',
            'deliver': 'final_format',
            'analyze': 'final_format',
            'summarize': 'final_format',
            'criticize': 'final_format',
            'visualize': 'final_format'}
        return self

    def _draft_file_names(self) -> None:
        """Drafts default import and export file names for data."""
        self.import_file_names = {
            'acquire': None,
            'parse': None,
            'clean': 'harvested_data',
            'merge': 'cleaned_data',
            'deliver': 'baled_data',
            'analyze': 'final_data',
            'summarize': 'final_data',
            'criticize': 'final_data',
            'visualize': 'predicted_data'}
        self.export_file_names = {
            'acquire': 'source_format',
            'parse': 'interim_format',
            'clean': 'interim_format',
            'merge': 'interim_format',
            'deliver': 'final_format',
            'analyze': 'final_format',
            'summarize': 'final_data',
            'criticize': 'predicted_data',
            'visualize': 'predicted_data'}
        return self

    def _draft_folders(self) -> None:
        """Drafts default import and export folder names for data.

        These folder names correspond to the data science CookieCutter tree
        structure. If using an alternative method, these mappings need to
        be replaced either through a subclass or by modifying the attributes.

        """
        self.import_folders = {
            'acquire': 'raw',
            'reap': 'raw',
            'clean': 'interim',
            'merge': 'interim',
            'deliver': 'interim',
            'analyze': 'processed',
            'summarize': 'processed',
            'criticize': 'processed',
            'visualize': 'processed'}
        self.export_folders = {
            'acquire': 'raw',
            'reap': 'interim',
            'clean': 'interim',
            'merge': 'interim',
            'deliver': 'processed',
            'analyze': 'processed',
            'summarize': 'processed',
            'criticize': 'processed',
            'visualize': 'processed'}
        return self

    def _make_unique_path(self, folder: Path, name: str) -> Path:
        """Creates a unique path to avoid overwriting a file or folder.

        Thanks to RealPython for this bit of code:
        https://realpython.com/python-pathlib/.

        Args:
            folder (Path): the folder where the file or folder will be located.
            name (str): the basic name that should be used.

        Returns:
            Path: with a unique name. If the original name conflicts with an
                existing file/folder, a counter is used to find a unique name
                with the counter appended as a suffix to the original name.

        """
        counter = 0
        while True:
            counter += 1
            path = folder / name.format(counter)
            if not path.exists():
                return path

    def _pathlibify(self,
            folder: str,
            name: Optional[str] = None,
            extension: Optional[str] = None) -> Path:
        """Converts strings to pathlib Path object.

        If 'name' and 'extension' are passed, a file path is created. Otherwise,
        a folder path is created.

        Args:
            folder (str): folder for file location.
            name (Optional[str]): the name of the file.
            extension (Optional[str]): the extension of the file.

        Returns:
            Path: formed from string arguments.

        """
        try:
            folder = self.folders[folder]
        except (KeyError, TypeError):
            try:
                if folder.is_dir():
                    pass
            except (AttributeError, TypeError):
                folder = self.folders['root'].joinpath(folder)
        if name and extension:
            return folder.joinpath('.'.join([name, extension]))
        elif isinstance(folder, Path):
            return folder
        else:
            return Path(folder)

    def _write_folder(self, folder: Union[str, Path]) -> None:
        """Writes folder to disk.

        Parent folders are created as needed.

        Args:
            folder (Union[str, Path]): intended folder to write to disk.

        """
        Path.mkdir(folder, parents = True, exist_ok = True)
        return self

    """ File Input/Output Methods """

    def load(self,
            file_path: Optional[Union[str, Path]] = None,
            folder: Optional[Union[str, Path]] = None,
            file_name: Optional[str] = None,
            file_format: Optional[Union[str, 'FileFormat']] = None,
            **kwargs) -> Any:
        """Imports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path (Optional[Union[str, Path]]): a complete file path.
                Defaults to None.
            folder (Optional[Union[str, Path]]): a complete folder path or the
                name of a folder stored in 'filer'. Defaults to None.
            file_name (Optional[str]): file name without extension. Defaults to
                None.
            file_format (Optional[Union[str, 'FileFormat']]): object with
                information about how the file should be loaded or the key to
                such an object stored in 'filer'. Defaults to None
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Any: depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        """
        if self.file_formats[file_format].module in ['pandas', 'numpy']:
            importer = self.data_importer
        else:
            importer = self.results_importer
        return importer.apply(
                file_path = file_path,
                folder = folder,
                file_name = file_name,
                file_format = file_format,
                **kwargs)

    def save(self,
            variable: Any,
            file_path: Optional[Union[str, Path]] = None,
            folder: Optional[Union[str, Path]] = None,
            file_name: Optional[str] = None,
            file_format: Optional[Union[str, 'FileFormat']] = None,
            **kwargs) -> None:
        """Exports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable (Any): object to be save to disk.
            file_path (Optional[Union[str, Path]]): a complete file path.
                Defaults to None.
            folder (Optional[Union[str, Path]]): a complete folder path or the
                name of a folder stored in 'filer'. Defaults to None.
            file_name (Optional[str]): file name without extension. Defaults to
                None.
            file_format (Optional[Union[str, 'FileFormat']]): object with
                information about how the file should be loaded or the key to
                such an object stored in 'filer'. Defaults to None
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        """
        if self.file_formats[file_format].module in ['pandas', 'numpy']:
            exporter = self.data_exporter
        else:
            exporter = self.results_exporter
        exporter.apply(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format,
            **kwargs)
        return self

    def set_project_folder(self, name: Optional[str] = None) -> None:
        """Sets project folder for results for a Project instance to be saved.

        Args:
            name (Optional[str]): name of folder to use. Defaults to None. If
                not passed, a unique folder name will be created with the
                prefix of 'project_' and the suffix of the current date and
                time.

        """
        if name is None:
            name = '_'.join('project', datetime_string())
        self.folders['project'] = self.folders['results'].joinpath(name)
        self._write_folder(folder = self.folders['project'])
        return self

    def set_chapter_folder(self,
            prefix: Optional[str] = None,
            name: Optional[str] = None) -> None:
        """Sets chapter folder for results for a Chapter instance to be saved.

        Args:
            prefix (Optional[str]): prefix of folder to use. Defaults to None.
                If not passed, the prefix 'chapter_' is used.
            name (Optional[str]): suffix to chapter name to use. Defaults to
                None. If not passed, the '_make_unique_path' method is called
                to dynamically create a path.

        """
        prefix = prefix or 'chapter'
        if name:
            return self.folders['project'].joinpath('_'.join([prefix, name]))
        else:
            return self._make_unique_path(
                folder = self.folders['project'],
                name = '_'.join([prefix, '{:03d}']))

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Initializes core paths and attributes."""
        # Initializes 'state' for state management.
        self.state = create_states(states = self.states)
        # Transforms root folder path into a Path object.
        self._draft_root()
        # Creates basic folder structure and writes folders to disk.
        self._draft_core_folders()
        # Creates catalogs of file formats, folders, and file names.
        self._draft_file_formats()
        self._draft_folders()
        self._draft_file_names()
        # Creates importer and exporter instances for file management.
        self.data_importer = Importer(
            filer = self,
            root_folder = self.data_folder,
            folders = self.import_folders,
            file_format_states = self.import_format_states,
            file_names = self.import_file_names)
        self.data_exporter = Exporter(
            filer = self,
            root_folder = self.data_folder,
            folders = self.export_folders,
            file_format_states = self.export_format_states,
            file_names = self.export_file_names)
        self.results_importer = Importer(
            filer = self,
            root_folder = self.data_folder)
        self.results_exporter = Exporter(
            filer = self,
            root_folder = self.results_folder)
        return self


@dataclass
class SimpleDistributor(ABC):
    """Base class for siMpLify Importer and Exporter.

    Args:
        filer ('Filer'): a related Filer instance.

    """

    filer: 'Filer'

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Creates 'Pathifier' instance for dynamic path creation.
        self.pathifier = Pathifier(
            filer = self.filer,
            distributor = self)
        return self

    """ Private Methods """

    def _check_kwargs(self,
            file_format: 'FileFormat',
            passed_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Selects kwargs for particular methods.

        If a needed argument was not passed, default values are used.

        Args:
            file_format ('FileFormat'): an instance with information about
                additional kwargs to search for.
            passed_kwargs (Dict[str, Any]): kwargs passed to method.

        Returns:
            Dict[str, Any]: kwargs with only relevant parameters.

        """
        new_kwargs = passed_kwargs
        for variable in file_format.addtional_kwargs:
            if not variable in passed_kwargs:
                if variable in self.filer.default_kwargs:
                    new_kwargs.update(
                        {variable: self.filer.default_kwargs[variable]})
                elif hasattr(self.filer, variable):
                    new_kwargs.update(
                        {variable: getattr(self.filer, variable)})
        return new_kwargs

    def _check_file_format(self,
            file_format: Union[str, 'FileFormat']) -> 'FileFormat':
        """Selects 'file_format' or default value.

        Args:
            file_format (Union[str, 'FileFormat']): name of file format or a
                'FileFormat' instance.

        Returns:
            str: completed file_format.

        """
        if isinstance(file_format, FileFormat):
            return file_format
        elif isinstance(file_format, str):
            return self.filer.file_formats[file_format]
        else:
            return self.filer.file_formats[
                self.file_formats[self.filer.state]]

    def _make_parameters(self,
            file_format: 'FileFormat',
            **kwargs) -> Dict[str, Any]:
        """Creates complete parameters for a file input/output method.

        Args:
            file_format ('FileFormat'): an instance with information about the
                needed and optional parameters.
            kwargs: additional parameters to pass to an input/output method.

        Returns:
            Dict[str, Any]: parameters to be passed to an input/output method.

        """
        parameters = self._check_kwargs(
            file_format = file_format,
            passed_kwargs = kwargs)
        try:
            parameters.update(file_format.required)
        except TypeError:
            pass
        if kwargs:
            parameters.update(**kwargs)
        return parameters


@dataclass
class Importer(SimpleDistributor):
    """Manages file importing for siMpLify.

    Args:
        filer ('Filer'): related Filer instance.
        root_folder (Optional[str]): the root folder for files to be loaded.
            This should usually be the data or results folder from 'filer'.
            Defaults to None.
        folders (Optional[Dict[str, str]]): mapping with keys of Project ststes
            and values corresponding to folders stored in 'filer'. Defaults
            to an empty dictionary.
        file_format_states (Optional[Dict[str, str]]): mapping with keys of
            Project states and values corresponding to keys of 'file_formats'
            in filer. This mapping is used if different file formats are
            used at different stages of the project (most often when the
            original data format is not desired for long-term use). Defaults to
            an empty dictionary.
        file_names (Optional[Dict[str, str]]): mapping with keys of Project
            states and values of default file names. Defaults to an empty
            dictionary.

    """
    filer: 'Filer'
    root_folder: Optional[str] = None
    folders: Optional[Dict[str, str]] = field(default_factory = dict)
    file_format_states: Optional[Dict[str, str]] = field(default_factory = dict)
    file_names: Optional[Dict[str, str]] = field(default_factory = dict)

    """ Public Methods """

    def load(self, **kwargs):
        """Calls 'apply' method with **kwergs."""
        return self.apply(**kwargs)

    def make_batch(self,
            folder: Optional[Union[str, Path]] = None,
            file_format: Optional[Union[str, 'FileFormat']] = None,
            include_subfolders: Optional[bool] = True) -> Iterable:
        """Creates an iterable of paths for importing files.

        If 'include_subfolders' is True, subfolders are searched as well for
        matching 'file_format' files.

        Args:
            folder (Optional[Union[str, Path]]): path of folder or string
                corresponding to class attribute with path. Defaults to None.
            file_format (Optional[Union[str, 'FileFormat']]): file format name
                or a FileFormat instance. Defeaults to None.
            include_subfolders (Optional[bool]): whether to include files in
                subfolders when creating a batch. Defaults to True

        Returns:
            Iterable: matching file paths.

        """
        folder = folder or self.filer[self.folders[self.filer.stage]]
        file_format = self._check_file_format(file_format = file_format)
        if include_subfolders:
            return Path(folder).rglob('.'.join(['*', file_format.extension]))
        else:
            return Path(folder).glob('.'.join(['*', file_format.extension]))

    # def iterate_batch(self,
    #         chapters: List[str],
    #         dataset: 'Dataset' = None,
    #         return_dataset: Optional[bool] = True):
    #     """Iterates through a list of files contained in self.batch and
    #     applies the chapters created by a book method (or subclass).
    #     Args:
    #         chapters(list): list of chapter types (Recipe, Harvest, etc.)
    #         dataset(Dataset): an instance of Dataset or subclass.
    #         return_dataset(bool): whether dataset should be returned by
    #         this method.
    #     Returns:
    #         If 'return_dataset' is True: an in instance of Dataset.
    #         If 'return_dataset' is False, no value is returned.
    #     """
    #     if dataset:
    #         for file_path in self.batch:
    #             dataset.source = self.load(file_path = file_path)
    #             for chapter in chapters:
    #                 data = chapter.produce(data = dataset)
    #         if return_dataset:
    #             return dataset
    #         else:
    #             return self
    #     else:
    #         for file_path in self.batch:
    #             for chapter in chapters:
    #                 chapter.produce()
    #         return self

    """ Core siMpLify Methods """

    def apply(self,
            file_path: Optional[Union[str, Path]] = None,
            folder: Optional[Union[str, Path]] = None,
            file_name: Optional[str] = None,
            file_format: Optional[Union[str, 'FileFormat']] = None,
            **kwargs) -> Any:
        """Imports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            file_path (Optional[Union[str, Path]]): a complete file path.
                Defaults to None.
            folder (Optional[Union[str, Path]]): a complete folder path or the
                name of a folder stored in 'filer'. Defaults to None.
            file_name (Optional[str]): file name without extension. Defaults to
                None.
            file_format (Optional[Union[str, 'FileFormat']]): object with
                information about how the file should be loaded or the key to
                such an object stored in 'filer'. Defaults to None
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        Returns:
            Any: depending upon method used for appropriate file format, a new
                variable of a supported type is returned.

        """
        file_format = self._check_file_format(file_format = file_format)
        file_path = self.pathifier.apply(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        if file_format.module:
            tool = file_format.load('import_method')
        else:
            tool = getattr(self, file_format.import_method)
        parameters = self._make_parameters(file_format = file_format, **kwargs)
        if sample_size:
            parameters[file_format.sample_size_parameter] = sample_size
        return tool(file_path, **parameters)


@dataclass
class Exporter(SimpleDistributor):
    """Manages file exporting for siMpLify.

    Args:
        filer ('Filer'): related Filer instance.
        root_folder (Optional[str]): the root folder for files to be loaded.
            This should usually be the data or results folder from 'filer'.
            Defaults to None.
        folders (Optional[Dict[str, str]]): mapping with keys of Project ststes
            and values corresponding to folders stored in 'filer'. Defaults
            to an empty dictionary.
        file_format_states (Optional[Dict[str, str]]): mapping with keys of
            Project states and values corresponding to keys of 'file_formats'
            in filer. This mapping is used if different file formats are
            used at different stages of the project (most often when the
            original data format is not desired for long-term use). Defaults to
            an empty dictionary.
        file_names (Optional[Dict[str, str]]): mapping with keys of Project
            states and values of default file names. Defaults to an empty
            dictionary.

    """
    filer: 'Filer' = None
    root_folder: Optional[str] = None
    folders: Optional[Dict[str, str]] = field(default_factory = dict)
    file_format_states: Optional[Dict[str, str]] = field(default_factory = dict)
    file_names: Optional[Dict[str, str]] = field(default_factory = dict)

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
        if not self.filer.boolean_out:
            data.replace({True: 1, False: 0}, inplace = True)
        return data

    """ Public Methods """

    # def initialize_writer(self,
    #         file_path: str,
    #         columns: List[str],
    #         encoding: Optional[str] = None,
    #         dialect: Optional[str] = 'excel') -> None:
    #     """Initializes writer object for line-by-line exporting to a .csv file.

    #     Args:
    #         file_path (str): a complete path to the file being written to.
    #         columns (List[str]): column names to be added to the first row of
    #             the file as column headers.
    #         encoding (str): a python encoding type.
    #         dialect (str): the specific type of csv file created. Defaults to
    #             'excel'.

    #     """
    #     if not columns:
    #         error = 'initialize_writer requires columns as a list type'
    #         raise TypeError(error)
    #     with open(file_path, mode = 'w', newline = '',
    #               encoding = self.file_encoding) as self.output_series:
    #         self.writer = csv.writer(self.output_series, dialect = dialect)
    #         self.writer.writerow(columns)
    #     return self

    # def iterate_writer(self):
    #     return self

    def save(self, **kwargs):
        """Calls 'apply' method with **kwargs."""
        return self.apply(**kwargs)

    """ Core siMpLify Methods """

    def apply(self,
            variable: Any,
            file_path: Optional[Union[str, Path]] = None,
            folder: Optional[Union[str, Path]] = None,
            file_name: Optional[str] = None,
            file_format: Optional[Union[str, 'FileFormat']] = None,
            **kwargs) -> None:
        """Exports file by calling appropriate method based on file_format.

        If needed arguments are not passed, default values are used. If
        file_path is passed, folder and file_name are ignored.

        Args:
            variable (Any): object to be save to disk.
            file_path (Optional[Union[str, Path]]): a complete file path.
                Defaults to None.
            folder (Optional[Union[str, Path]]): a complete folder path or the
                name of a folder stored in 'filer'. Defaults to None.
            file_name (Optional[str]): file name without extension. Defaults to
                None.
            file_format (Optional[Union[str, 'FileFormat']]): object with
                information about how the file should be loaded or the key to
                such an object stored in 'filer'. Defaults to None
            **kwargs: can be passed if additional options are desired specific
                to the pandas or python method used internally.

        """
        file_format = self._check_file_format(file_format = file_format)
        file_path = self.pathifier.apply(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        # Changes boolean values to 1/0 if 'boolean_out' is False.
        if file_format.module in ['pandas']:
            variable = self._check_boolean_out(data = variable)
        if file_format.module:
            tool = file_format.load('export_method')
        else:
            tool = getattr(self, file_format.export_method)
        parameters = self._make_parameters(file_format = file_format, **kwargs)
        tool(variable, file_path, **parameters)
        return self


@dataclass
class Pathifier(object):
    """Builds file_paths based upon state.

    Args:
        filer ('Filer): related 'Filer' instance.
        distributor ('SimpleDistributor'): related 'SimpleDistributor' instance.

    """
    filer: 'Filer'
    distributor: 'SimpleDistributor'

    def __post_init__(self) -> None:
        return self

    """ Private Methods """

    def _check_folder(self, folder: Optional[str] = None) -> str:
        """Selects 'folder' or default value.

        Args:
            folder (Optional[str]): name of folder. Defaults to None.

        Returns:
            str: completed folder.

        """
        if not folder:
            return self.filer.folders[self.distributor.folders[
                self.filer.state]]
        else:
            try:
                return self.filer.folders[folder]
            except AttributeError:
                if isinstance(folder, str):
                    return Path(folder)
                else:
                    return folder

    def _check_file_name(self, file_name: Optional[str] = None) -> str:
        """Selects 'file_name' or default value.

        Args:
            file_name (Optional[str]): name of file. Defaults to None.

        Returns:
            str: completed file_name.

        """
        if not file_name:
            return self.distributor.file_names[self.filer.state]
        else:
            return file_name

    def _make_path(self,
            folder: str,
            file_name: str,
            file_format: 'FileFormat') -> Path:
        """Creates completed file_path from passed arguments.

        Args:
            folder (str): name of target folder.
            file_name (str): name of file.
            file_format ('FileFormat'): instance with instructions about the
                selected file format.

        Returns:
            Path: completed file path.

        """
        return folder.joinpath('.'.join([file_name, file_format.extension]))

    """ Core siMpLify Methods """

    def apply(self,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: 'FileFormat' = None) -> Path:
        """Creates file path from passed arguments.

        Args:
            file_path (Optional[str]): full file path. Defaults to None.
            folder (Optional[str]): name of target folder (not used if
                'file_path' passed). Defaults to None.
            file_name (Optional[str]): name of file (not used if 'file_path'
                passed). Defaults to None.
            file_format (Optional['FileFormat']): instance with instructions
                about the selected file format. Defaults to None.

        Returns:
            str of completed file path.

        """
        if isinstance(file_path, Path):
            return file_path
        elif isinstance(file_path, str):
            return Path(file_path)
        else:
            return self._make_path(
                folder = self._check_folder(folder = folder),
                file_name = self._check_file_name(file_name = file_name),
                file_format = file_format)


@dataclass
class FileFormat(SimpleComponent):
    """File format information and instructions

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        extension (Optional[str]): actual file extension to use. Defaults to
            None.
        import_method (Optional[str]): name of import method in 'module' to
            use. If module is None, the SimpleDistributor looks for the method
            as a local attribute. Defaults to None.
        export_method (Optional[str]): name of export method in 'module' to
            use. If module is None, the SimpleDistributor looks for the method
            as a local attribute. Defaults to None.
        additional_kwargs (Optional[List[str]]): names of commonly used kwargs
            for either the import or export method. Defaults to None.
        required (Optional[Dict[str, Any]]): any required parameters that should
            be passed to the import or export methods. Defaults to None.
        test_size_parameter (Optional[str]): the name of the parameter for
            loading a sample of data for the particular import method. Defaults
            to None.

    """

    name: str
    module: str
    extension: Optional[str] = None
    import_method: Optional[str] = None
    export_method: Optional[str] = None
    additional_kwargs: Optional[List[str]] = None
    required: Optional[Dict[str, Any]] = None
    test_size_parameter: Optional[str] = None

