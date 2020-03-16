"""
.. module:: siMpLify project
:synopsis: data science projects made simple
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections.abc import Iterable
from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)
import warnings

import numpy as np
import pandas as pd

from simplify.core.base import SimpleSystem
from simplify.core.base import SimpleComponent
from simplify.core.dataset import Dataset
from simplify.core.filer import Filer
from simplify.core.idea import Idea
from simplify.core.library import Book
from simplify.core.library import Chapter
from simplify.core.library import Library
from simplify.core.manager import Manager
from simplify.core.manager import Worker
from simplify.core.overview import Overview
from simplify.core.repository import Repository
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify
from simplify.core.utilities import subsetify


@dataclass
class Project(SimpleSystem):
    """Controller class for siMpLify projects.

    Args:
        idea (Optional[Union[Idea, str]]): an instance of Idea or a string
            containing the file path or file name (in the current working
            directory) where a file of a supported file type with settings for
            an Idea instance is located. Defaults to None.
        filer (Optional[Union['Filer', str]]): an instance of Filer or a string
            containing the full path of where the root folder should be located
            for file output. A filer instance contains all file path and
            import/export methods for use throughout siMpLify. Defaults to None.
        dataset (Optional[Union['Dataset', pd.DataFrame, np.ndarray, str]]): an
            instance of Dataset, an instance of Data, a string containing the
            full file path where a data file for a pandas DataFrame is located,
            a string containing a file name in the default data folder (as
            defined in the shared Filer instance), a full folder path where raw
            files for data to be extracted from, a string
            containing a folder name which is an attribute in the shared Filer
            instance, a DataFrame, or numpy ndarray. If a DataFrame, Data
            instance, ndarray, or string is
            passed, the resultant data object is stored in the 'data' attribute
            in a new Dataset instance as a DataFrame. Defaults to None.
        packages (Optional[Union[List[str], Dict[str, 'Package'], Dict[str,
            'Worker'], 'Repository', 'Manager']]): mapping of 'Package'
            instances or the information needed to create one and store it in
            a 'Manager' instance. Defaults to an empty 'Repository' instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'project'.
        identification (Optional[str]): a unique identification name for this
            'Project' instance. The name is used for creating file folders
            related to the 'Project'. If not provided, a string is created from
            the date and time.
        auto_draft (Optional[bool]): whether to call the 'draft' method when
            instanced. Defaults to True.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.

    """
    idea: Optional['Idea'] = None
    filer: Optional['Filer'] = None
    dataset: Optional[Union[
        'Dataset',
        pd.DataFrame,
        np.ndarray,
        str,
        Dict[str, Union[
            pd.DataFrame,
            np.ndarray,
            str]]]] = None
    packages: Optional[Union[
        List[str],
        Dict[str, 'Package'],
        Dict[str, 'Worker'],
        'Repository',
        'Manager']] = field(default_factory = Repository)
    name: Optional[str] = field(default_factory = lambda: 'project')
    identification: Optional[str] = field(default_factory = datetime_string)
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Validates 'Idea' instance.
        self.idea = Idea.create(idea = self.idea)
        # Validates 'Filer' instance.
        self.filer = Filer.create(root_folder = self.filer, idea = self.idea)
        # Validates 'Dataset' instance.
        self.dataset = Dataset.create(data = self.dataset, idea = self.idea)
        # Adds general attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Creates a 'Manager' instance for storing 'Worker' instances.
        self.manager = Manager.create(
            packages=self.packages,
            idea=self.idea)
        # Validates 'packages' or creates it from 'idea' and default packages.
        self.packages = self._initialize_packages(packages = self.packages)
        # Creats an 'Overview' instance, providing an outline of the overall
        # project from 'Worker' instances stored in 'manager'.
        self.overview = Overview.create(manager = self.manager)
        # Creates a 'Library' instance for storing 'Book' instances.
        self.library = Library.create(
            manager = self.manager,
            catalog = self.overview)
        # Initializes 'stage' and validates core siMpLify objects.
        super().__post_init__()
        # Calls 'draft' method if 'auto_draft' is True.
        if self.auto_draft:
            self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        # Calls 'apply' method if 'auto_apply' is True.
        if self.auto_apply:
            self.apply()
        return self

    """ Factory Method """

    @classmethod
    def create(cls,
            idea: Optional[Union['Idea', str]] = None,
            filer: Optional[Union['Filer', str]] = None,
            dataset: Optional[Union[
                'Dataset',
                pd.DataFrame,
                np.ndarray,
                str,
                Dict[str, Union[
                    pd.DataFrame,
                    np.ndarray,
                    str]]]] = None,
            packages: Optional[Union[
                List[str],
                Dict[str, 'Package'],
                Dict[str, 'Worker'],
                'Repository',
                'Manager']] = field(default_factory = Repository),
            name: Optional[str] = field(default_factory = lambda: 'project'),
            identification: Optional[str] = field(
                default_factory = datetime_string),
            auto_draft: Optional[bool] = True,
            auto_publish: Optional[bool] = True,
            auto_apply: Optional[bool] = False) -> 'Project':
        """Creates a 'Project' instance from passed arguments.

        Args:
            idea (Optional[Union[Idea, str]]): an instance of Idea or a string
                containing the file path or file name (in the current working
                directory) where a file of a supported file type with settings
                for an Idea instance is located. Defaults to None.
            filer (Optional[Union['Filer', str]]): an instance of Filer or a
                string containing the full path of where the root folder should
                be located for file output. A filer instance contains all file
                path and import/export methods for use throughout siMpLify.
                Defaults to None.
            dataset (Optional[Union['Dataset', pd.DataFrame, np.ndarray, str]]):
                an instance of Dataset, an instance of Data, a string containing
                the full file path where a data file for a pandas DataFrame is
                located, a string containing a file name in the default data
                folder (as defined in the shared Filer instance), a full folder
                path where raw files for data to be extracted from, a string
                containing a folder name which is an attribute in the shared
                Filer instance, a DataFrame, or numpy ndarray. If a DataFrame,
                Data instance, ndarray, or string is passed, the resultant data
                object is stored in the 'data' attribute in a new Dataset
                instance as a DataFrame. Defaults to None.
            packages (Optional[Union[List[str], Dict[str, 'Package'], Dict[str,
                'Worker'], 'Repository', 'Manager']]): mapping of 'Package'
                instances or the information needed to create one and store it
                in a 'Manager' instance. Defaults to an empty 'Repository'
                instance.
            name (Optional[str]): designates the name of the class instance used
                for internal referencing throughout siMpLify. If the class
                instance needs settings from the shared 'Idea' instance, 'name'
                should match the appropriate section name in that 'Idea'
                instance. When subclassing, it is a good idea to use the same
                'name' attribute as the base class for effective coordination
                between siMpLify classes. 'name' is used instead of
                __class__.__name__ to make such subclassing easier. Defaults to
                'project'.
            identification (Optional[str]): a unique identification name for
                this 'Project' instance. The name is used for creating file
                folders related to the 'Project'. If not provided, a string is
                created from the date and time.
            auto_draft (Optional[bool]): whether to call the 'draft' method when
                instanced. Defaults to True.
            auto_publish (Optional[bool]): whether to call the 'publish' method
                when instanced. Defaults to True.
            auto_apply (Optional[bool]): whether to call the 'apply' method when
                instanced. For auto_apply to have an effect, 'dataset' must also
                be passed. Defaults to False.

        """
        # Validates 'Idea' instance.
        idea = Idea.create(idea = idea)
        # Validates 'Filer' instance.
        filer = Filer.create(root_folder = filer, idea = idea)
        # Validates 'Dataset' instance.
        dataset = Dataset.create(data = dataset, idea = idea)
        return cls(
            idea = idea,
            filer = filer,
            dataset = dataset,
            packages = packages,
            name = name,
            identificaiton = identification,
            auto_draft = auto_draft,
            auto_publish = auto_publish,
            auto_apply = auto_apply)

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable for class instance, depending upon 'stage'.

        Returns:
            Iterable: different depending upon stage.

        """
        if self.stage in ['outline']:
            return iter(self.overview)
        if self.stage in ['draft']:
            return iter(self.manager)
        elif self.stage in ['publish', 'apply']:
            return iter(self.library)

    """ Other Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies Project.

        Calling Project as a function is compatible with and used by the
        command line interface.

        """
        self.auto_apply = True
        self.__post__init()
        return self

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return f'Project {self.identification}: {str(self.overview)}'

    """ Core siMpLify Methods """

    def add(self,
            item: Union[
                'Package',
                'Library',
                'Book',
                'Chapter',
                'Manager',
                'Worker',
                'Dataset',
                 str],
            name: Optional[str] = None,
            overwrite: Optional[bool] = False) -> None:
        """Adds 'worker' to 'manager' or 'book' to 'library'.

        Args:
            item (Union['Package', 'Library', 'Book', 'Chapter', 'Manager',
                'Worker', 'Dataset', str]): a siMpLify object to add
            name (Optional[str]): key to use for the passed item in either
                'library' or 'manager'. Defaults to None. If not passed, the
                'name' attribute of item will be used as the key for item.
           overwrite (Optional[bool]): whether to overwrite an existing
                attribute with the imported object (True) or to update the
                existing attribute with the imported object, if possible
                (False). Defaults to True.

        """
        if name is None:
            try:
                name = item.name
            except (AttributeError, TypeError):
                name = item
        if isinstance(item, str):
            self.manager.add(worker = self.options[item].load())
        elif isinstance(item, Worker):
            self.manager.add(worker = item)
        elif isinstance(item, Package):
            self.options[name] = item
            self.packages[name] = item
            self.manager.add(worker = item.load())
        elif isinstance(item, Book):
            self.library.add(book = item)
        else:
            raise TypeError(
                'add requires a Worker, Book, Package, or string type')
        return self

    def draft(self) -> None:
        """Initializes 'workers' and drafts a 'Library' instance."""
        self.change_stage(new_stage = 'draft')
        # Iterates through 'workers' and creates Book instances in 'library'.
        for name, worker in self.manager.items():
            self.library = worker.publisher.draft(library = self.library)
        return self

    def publish(self) -> None:
        """Finalizes 'Book' instances in 'Library'."""
        self.change_stage(new_stage = 'publish')
        # Iterates through 'workers' and finalizes each Book instance. The
        # finalized instances are stored in 'library'.
        for name, worker in self.manager.items():
            self.library = worker.publisher.publish(library = self.library)
        return self

    def apply(self, data: Optional['Dataset'] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Optional['Dataset']): data object for methods to be
                applied. If not passed, data stored in the 'dataset' is
                used.
            kwargs: any other parameters to pass to the 'apply' method of a
                'Scholar' instance.

        """
        # Changes state.
        self.change_stage(new_stage = 'apply')
        # Assigns 'data' to 'dataset' attribute and validates it.
        if data:
            self.dataset = Dataset.create(data = data)
        # Iterates through each worker, creating and applying needed Books,
        # Chapters, and Techniques for each worker in the Library.
        for name, book in self.library.items():
            self.library = self.workers[name].scholar.apply(
                worker = name,
                library = self.library,
                data = self.dataset,
                **kwargs)
        return self

    """ File Import/Export Methods """

    def load(self,
            file_path: Union[str, Path],
            overwrite: Optional[bool] = True) -> None:
        """Loads a siMpLify object and stores it in the appropriate attribute.

        Args:
            file_path (Union[str, Path]): path to saved 'Library' instance.
            overwrite (Optional[bool]): whether to overwrite an existing
                attribute with the imported object (True) or to update the
                existing attribute with the imported object, if possible
                (False). Defaults to True.

        """
        loaded = self.filer(file_path = file_path)
        if isinstance(loaded, Project):
            self = loaded
        elif isinstance(loaded, Library):
            if overwrite:
                self.library = loaded
            else:
                self.library.update(loaded)
        elif isinstance(loaded, Book):
            self.library.add(book = loaded)
        elif isinstance(loaded, Manager):
            if overwrite:
                self.manager = loaded
            else:
                self.manager.update(loaded)
        elif isinstance(loaded, Worker):
            self.manager.add(worker = loaded)
        elif isinstance(loaded, Dataset):
            if overwrite:
                self.dataset = loaded
            else:
                self.dataset.add(data = loaded)
        else:
            raise TypeError(
                'loaded object must be Projecct, Library, Book, Dataset, \
                     Manager, or Worker type')
        return self

    def save(self,
            attribute: Union[str, object],
            file_path: Optional[Union[str, Path]]) -> None:
        """Saves a siMpLify object.

        Args:
            attribute (Union[str, object]): either the name of the attribute or
                siMpLify object to save.
            file_path (Optional[Union[str, Path]]): path to save 'attribute'.

        Raises:
            AttributeError: if 'attribute' is a string and cannot be found in
                the 'Project' subclass or its 'manager' and 'library'
                attributes.

        """
        if isinstance(attribute, str):
            try:
                attribute = getattr(self, attribute)
            except AttributeError:
                try:
                    attribute = getattr(self.manager, attribute)
                except AttributeError:
                    try:
                        attribute = getattr(self.library, attribute)
                    except AttributeError:
                        AttributeError(f'attribute not found in {self.name}')
        else:
            self.filer.save(attribute)
        return self

    """ Private Methods """

    def _initialize_packages(self,
            packages: Optional[List[str]]) -> Dict[str, 'Package']:
        """Validates 'packages' or converts them to the appropriate type.

        Args:
            packages (Optional[List[str]]): a list

        Returns:
            Dict[str, 'Package']:

        """
        self.options = DEFAULT_OPTIONS
        if not packages:
            try:
                outer_key = self.__class__.__name__.lower()
                inner_key = f'{self.__class__.__name__.lower()}_packages'
                packages = listify(self.idea[outer_key][inner_key])
            except KeyError:
                pass
        if isinstance(packages, MutableMapping):
            return packages
        else:
            for package in packages:
                self.add(item = package)
                # new_packages[package] = self.options[package]
            if new_packages:
                return new_packages
            else:
                return self.packages


@dataclass
class Package(SimpleComponent):
    """Lazy loader for 'Worker' instances.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to use is located.
        worker (Optional[str]): name of 'Worker' subclass in 'module' to load.

    """
    name: Optional[str] = None
    module: Optional[str] = None
    worker: Optional[str] = None

    """ Core siMpLify Methods """

    def load(self) -> 'Worker':
        """Returns 'Worker' from 'module'.

        Returns:
            'Worker': from 'module'.

        """
        return getattr(import_module(self.module), self.worker)


DEFAULT_OPTIONS = {
    'wrangler': Package(
        name = 'wrangler',
        module = 'simplify.wrangler.wrangler',
        worker = 'Wrangler'),
    'explorer': Package(
        name = 'explorer',
        module = 'simplify.explorer.explorer',
        worker = 'Explorer'),
    'analyst': Package(
        name = 'analyst',
        module = 'simplify.analyst.analyst',
        worker = 'Analyst'),
    'critic': Package(
        name = 'critic',
        module = 'simplify.critic.critic',
        worker = 'Critic'),
    'artist': Package(
        name = 'artist',
        module = 'simplify.artist.artist',
        worker = 'Artist')}
