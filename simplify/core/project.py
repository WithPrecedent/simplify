"""
.. module:: siMpLify project
:synopsis: controller class for siMpLify projects
:publisher: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)
import warnings

import numpy as np
import pandas as pd

from simplify.core.base import SimpleLoader
from simplify.core.base import SimpleSettings
from simplify.core.book import Book
from simplify.core.dataset import Dataset
from simplify.core.filer import Filer
from simplify.core.idea import Idea
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify


@dataclass
class ProjectManager(object):
    """Controller class for siMpLify projects.

    Args:
        idea (Optional[Union[Idea, str]]): an instance of Idea or a string
            containing the file path or file name (in the current working
            directory) where a file of a supported file type with settings for
            an Idea instance is located. Defaults to None.
        filer (Optional[Union['Filer', str]]): an instance of Filer
            or a string containing the full path of where the root folder should
            be located for file output. A filer instance contains all file
            path and import/export methods for use throughout the siMpLify
            package. Default is None.
        dataset (Optional[Union['Dataset', 'Data', pd.DataFrame,
            np.ndarray, str]]): an instance of Dataset, an instance of
            Data, a string containing the full file path where a data file
            for a pandas DataFrame is located, a string containing a file name
            in the default data folder (as defined in the shared Filer
            instance), a full folder path where raw files for data to be
            extracted from, a string containing a folder name which is an
            attribute in the shared Filer instance, a DataFrame, or numpy
            ndarray. If a DataFrame, Data instance, ndarray, or string is
            passed, the resultant data object is stored in the 'data' attribute
            in a new Dataset instance as a DataFrame. Default is None.
        workers (Optional[Union[Dict[str, 'Worker']], List[str]]]): dictionary
            with keys as strings and values of 'Worker' instances, or a list of
            workers corresponding to keys in 'default_workers' to use. Defaults
            to an empty dictionary. If nothing is provided, Project attempts to
            construct workers from 'idea' and 'default_workers'.
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
    idea: Optional[Union['Idea', str]] = None
    filer: Optional[Union['Filer', str]] = None
    dataset: Optional[Union[
        'Dataset',
        pd.DataFrame,
        np.ndarray,
        str,
        Dict[str, Union[
            pd.DataFrame,
            np.ndarray,
            str]]]] = None
    workers: Optional[Union[Dict[str, 'Worker'], List[str]]] = field(
        default_factory = dict)
    name: Optional[str] = field(default_factory = lambda: 'project')
    identification: Optional[str] = field(default_factory = datetime_string)
    auto_draft: Optional[bool] = True
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Validates 'Idea' instance and adds attributes from it.
        self.idea = Idea.create(idea = self.idea)
        self = self.idea.apply(instance = self)
        # Validates 'Filer' instance.
        self.filer = Filer.create(root_folder = self.filer, idea = self.idea)
        # Injects 'idea' and 'filer' into base 'SimpleSettings' class.
        SimpleSettings.idea = self.idea
        SimpleSettings.filer = self.filer
        # Validates 'Dataset' instance.
        self.dataset = Dataset.create(data = self.dataset)
        # Validates 'workers' attribute.
        self.workers = self._check_workers(workers = self.workers)
        # Creates a project 'overview' and 'Project' instance.
        self.project = Project(
            identification = self.identification,
            overview = self.outline(name = self.name))
        # Calls 'draft' method if 'auto_draft' is True.
        if self.auto_draft:
            self.project = self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.project = self.publish()
        # Calls 'apply' method if 'auto_apply' is True.
        if self.auto_apply:
            self.dataset = self.apply()
        return self

    """ Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies ProjectManager.

        This requires an dataset argument to be passed to work properly.

        Calling ProjectManager as a function is compatible with and used by the
        command line interface.

        Raises:
            ValueError: if 'dataset' is not passed when ProjectManager is called
                as a function.

        """
        # Validates 'dataset'.
        if self.dataset is None:
            raise ValueError(
                'Calling ProjectManager as a function requires a dataset')
        else:
            self.auto_apply = True
            self.__post__init()
        return self

    """ Private Methods """

    def _check_workers(self,
            workers: Union[
                Dict[str, 'Worker'], List[str]]) -> Dict[str, 'Worker']:
        """Creates or validates 'workers'.

        Args:
            workers (Union[Dict[str, 'Worker'], List[str]]): set of 'Worker'
                instances stored in dict or a list to create one from
                'default_workers'.

        Returns:
            Dict[str, 'Worker']: mapping with stored 'Worker' instances.

        """
        # Sets 'default_workers' to use in 'workers' construction or later
        # addition.
        self._set_defaults()
        if isinstance(workers, list) and workers:
            new_workers = {}
            for worker in workers:
                new_workers[worker] = self.default_workers[worker]
            return new_workers
        else:
            return workers

    def _set_defaults(self) -> None:
        """Sets 'default_workers' to use."""
        self.default_workers = {
            'wrangler': Worker(
                name = 'wrangler',
                module = 'simplify.wrangler.wrangler',
                book = 'Manual',
                scholar = 'Wrangler',
                options = 'Mungers',
                import_folder = 'raw',
                export_folder = 'raw'),
            'explorer': Worker(
                name = 'explorer',
                module = 'simplify.explorer.explorer',
                book = 'Ledger',
                scholar = 'Explorer',
                options = 'Measures'),
            'analyst': Worker(
                name = 'analyst',
                module = 'simplify.analyst.analyst',
                book = 'Cookbook',
                chapter = 'Recipe',
                technique = 'AnalystTechnique',
                scholar = 'Analyst',
                options = 'Tools'),
            'critic': Worker(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Anthology',
                chapter = 'Review',
                technique = 'Evaluator',
                scholar = 'Critic',
                options = 'Evaluators',
                data = 'analyst'),
            'artist': Worker(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas',
                scholar = 'Artist',
                options = 'Mediums',
                data = 'critic')}
        return self

    def _get_settings(self,
            section: str,
            prefix: str,
            suffix: str) -> List[str]:
        """Returns settings from 'idea' based on 'name' and 'suffix'.

        Args:
            section (str): outer key name in 'idea' section.
            prefix (str); prefix for an inner key name.
            suffix (str): suffix to inner key name in 'idea'.

        Returns:
            List[str]: names of matching workers, steps, or techniques.

        """
        return listify(self.idea[section]['_'.join([prefix, suffix])])

    def _initialize_workers(self,
            workers: Dict[str, 'Worker']) -> Dict[str, 'Worker']:
        """Creates instances for attributes for each Worker in 'workers'.

        Args:
            workers (Dict[str, 'Worker']): stored Worker instances.

        Returns:
            Dict[str, 'Worker']: instance with instances added.

        """
        for key, worker in workers.items():
            workers[key].steps = self.project.overview[key]
            workers[key].publisher = worker.load('publisher')(worker = worker)
            workers[key].scholar = worker.load('scholar')(worker = worker)
        return workers

    """ Public Methods """

    def add(self,
            name: Optional[str] = None,
            worker: Optional['Worker'] = None,
            **kwargs) -> None:
        """Adds subpackage to 'workers'.

        Args:
            name (Optional[str]): name of subpackage. This is used as both the
                key to the created Worker in 'workers' and as the 'name'
                attribute in the Worker. Defaults to None. If not provided, the
                'worker' will be added to 'workers' with the key being the
                'name' attribute of 'worker'.
            worker (Optional['Worker']): a completed instance. If not provided,
                the method will assume all of the parameters needed to construct
                a 'Worker' instance are in 'kwargs'.
            **kwargs: other attributes of a 'Worker' instance to pass.

        """
        if worker is not None:
            if name is not None:
                self.workers[name] = worker
            else:
                self.workers[worker.name] = worker
        elif name in self.default_workers:
            self.workers[name] = self.default_workers[name]
        else:
            self.workers[name] = Worker(name = name, **kwargs)
        return self

    """ Core siMpLify Methods """

    def outline(self, name: Optional[str] = None) -> (
            Dict[str, Dict[str, List[str]]]):
        """Creates nested dictinoary outlining a siMpLify project.

        Args:
            name (Optional[str]): name of project which should match a section
                in 'idea'. Defaults to None. If passed, 'name' is assigned as
                an instance attribute.

        Returns:
            Dict[str, Dict[str, List[str]]]: nested dictionary of workers,
                steps, and techniques for a siMpLify project.

        """
        if name is not None:
            self.name = name
        overview = {}
        # Uses passed 'workers' or gets 'workers' from 'idea'.
        workers = self.workers or self._check_workers(self._get_settings(
            section = self.name,
            prefix = self.name,
            suffix = 'workers'))
        for worker in workers:
            overview[worker] = {}
            steps = self._get_settings(
                section = worker,
                prefix = worker,
                suffix = 'steps')
            for step in steps:
                overview[worker][step] = self._get_settings(
                    section = worker,
                    prefix = step,
                    suffix = 'techniques')
        return overview

    def draft(self, project: Optional['Project'] = None) -> 'Project':
        """Initializes 'workers' and drafts a 'Project' instance.

        Args:
            project (Optional['Project']): an instance to be modified. Defaults
                to None. If not passed, the local 'project' attribute is used.

        Returns:
            'Project': with modifications made.

        """
        if project is None:
            project = self.project
        # Initializes 'workers' by loading appropriate class objects.
        self.workers = self._initialize_workers(workers = self.workers)
        # Iterates through 'workers' and creates Book instances in 'workers'.
        for name, worker in self.workers.items():
            # Drafts a Book instance for 'worker'.
            project = worker.publisher.draft(project = project)
        return project

    def publish(self, project: Optional['Project'] = None) -> 'Project':
        """Finalizes 'Book' instances in 'Project'.

        Args:
            project (Optional['Project']): an instance to be modified. Defaults
                to None. If not passed, the local 'project' attribute is used.

        Returns:
            'Project': with modifications made.

        """
        if project is None:
            project = self.project
        # Changes state.
        self.state = 'publish'
        # Iterates through 'workers' and finalizes each Book instance. The
        # finalized instances are then placed in 'library'.
        for name, worker in self.workers.items():
            project = worker.publisher.publish(project = project)
        return project

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
        self.state = 'apply'
        # Assigns 'data' to 'dataset' attribute and validates it.
        if data:
            self.dataset = Dataset.create(data = data)
        # Iterates through each worker, creating and applying needed Books,
        # Chapters, and Techniques for each worker in the Project.
        for name, book in self.project.library.items():
            self.project = self.workers[name].scholar.apply(
                worker = name,
                project = self.project,
                data = self.dataset,
                **kwargs)
        return self


@dataclass
class Project(MutableMapping):
    """Serializable object containing complete siMpLify project.

    Args:
        identification (Optional[str]): a unique identification name for this
            'Project' instance. The name is used for creating file folders
            related to the 'Project'. If not provided, a string is created from
            the date and time.
        overview (Optional[Dict[str, Dict[str, List[str]]]]): nested dictionary
            of workers, steps, and techniques for a siMpLify project. Defaults
            to an empty dictionary. An overview is not strictly needed for
            object serialization, but provides a good summary of the various
            options selected in a particular 'Project'. As a result, it is used
            by the '__repr__' and '__str__' methods.
        library (Optional[Dict[str, 'Book']]): stored 'Book' instances. Defaults
            to an empty dictionary.

    """
    identification: Optional[str] = field(default_factory = datetime_string)
    overview: Optional[Dict[str, Dict[str, List[str]]]] = field(
        default_factory = dict)
    library: Optional[Dict[str, 'Book']] = field(default_factory = dict)

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> 'Book':
        """Returns key from 'library'.
        Args:
            key (str): key to item in 'library'.

        Returns:
            'Book': from 'library'.

        """
        return self.library[key]

    def __setitem__(self, key: str, value: 'Book') -> None:
        """Sets 'key' in 'library' to 'value'.

        Args:
            key (str): key to item in 'library' to set.
            value ('Book'): instance to place in 'library'.

        """
        self.library[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'library'.

        Args:
            key (str): key in 'library'.

        """
        try:
            del self.library[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'library'.

        Returns:
            Iterable: of 'library'.

        """
        return iter(self.library)

    def __len__(self) -> int:
        """Returns length of 'library'.

        Returns:
            int: length of 'library'.
        """
        return len(self.library)

    """ Other Dunder Methods """

    def __repr__(self) -> str:
        """Returns string representation of a class instance."""
        return self.__str__()

    def __str__(self) -> str:
        """Returns string representation of a class instance."""
        return f'Project {self.identification}: {str(self.overview)}'

    """ Public Methods """

    def add(self, book: 'Book') -> None:
        """Adds 'book' to 'library'.

        Args:
            book ('Book'): an instance to be added.

        Raises:
            ValueError: if 'book' is not a 'Book' instance.

        """
        # Validates 'book' type as 'Book' before adding to 'library'.
        if isinstance(book, Book):
            self.library[book.name] = book
        else:
            raise ValueError('book must be a Book instance to add to a Library')
        return self


@dataclass
class Worker(SimpleLoader):
    """Object construction instructions used by a Project instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (Optional[str]): name of module where object to use is located
            (can either be a siMpLify or non-siMpLify module). Defaults to
            'simplify.core'.
        book (Optional[str]): name of Book object in 'module' to load. Defaults
            to 'Book'.
        chapter (Optional[str]): name of Chapter object in 'module' to load.
            Defaults to 'Chapter'.
        technique (Optional[str]): name of Book object in 'module' to load.
            Defaults to 'Technique'.
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to 'Publisher'.
        scholar (Optional[str]): name of Scholar class in 'module' to load.
            Defaults to 'Scholar'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union[str, Dict[str, Any]]]): a dictionary containing
            options for the 'Worker' instance to utilize or a string
            corresponding to a dictionary in 'module' to load. Defaults to an
            empty dictionary.
        data (Optional[str]): name of attribute or key in a 'Project' instance
            'library' to use as a data object to apply methods to. Defaults to
            'dataset'.
        import_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'filer' which
            contains the path to the default folder for exporting data objects.
            Defaults to 'processed'.

    """
    name: str
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')
    default_module: Optional[str] = field(
        default_factory = lambda: 'simplify.core')
    book: Optional[str] = field(default_factory = lambda: 'Book')
    chapter: Optional[str] = field(default_factory = lambda: 'Chapter')
    technique: Optional[str] = field(default_factory = lambda: 'Technique')
    publisher: Optional[str] = field(default_factory = lambda: 'Publisher')
    scholar: Optional[str] = field(default_factory = lambda: 'Scholar')
    steps: Optional[List[str]] = field(default_factory = list)
    options: Optional[Union[str, Dict[str, Any]]] = field(
        default_factory = dict)
    data: Optional[str] = field(default_factory = lambda: 'dataset')
    import_folder: Optional[str] = field(default_factory = lambda: 'processed')
    export_folder: Optional[str] = field(default_factory = lambda: 'processed')

    def __post_init__(self) -> None:
        # Declares 'default_components' in case specified components are not
        # found.
        self.default_components = {}
        for attribute in [
                'book',
                'chapter',
                'technique',
                'publisher',
                'scholar']:
            self.default_components[attribute] = getattr(self, attribute)
        return self