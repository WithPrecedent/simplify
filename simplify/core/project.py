"""
.. module:: siMpLify project
:synopsis: controller class for siMpLify projects
:publisher: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.core.book import Book
from simplify.core.book import Outline
from simplify.core.dataset import Dataset
from simplify.core.publisher import Publisher
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify
from simplify.core.worker import Worker


@dataclass
class Project(MutableMapping):
    """Controller class for siMpLify projects.

    Args:
        idea (Optional[Union[Idea, str]]): an instance of Idea or a string
            containing the file path or file name (in the current working
            directory) where a file of a supported file type with settings for
            an Idea instance is located. Defaults to None.
        inventory (Optional[Union['Inventory', str]]): an instance of Inventory
            or a string containing the full path of where the root folder should
            be located for file output. A inventory instance contains all file
            path and import/export methods for use throughout the siMpLify
            package. Default is None.
        dataset (Optional[Union['Dataset', 'Data', pd.DataFrame,
            np.ndarray, str]]): an instance of Dataset, an instance of
            Data, a string containing the full file path where a data file
            for a pandas DataFrame is located, a string containing a file name
            in the default data folder (as defined in the shared Inventory
            instance), a full folder path where raw files for data to be
            extracted from, a string containing a folder name which is an
            attribute in the shared Inventory instance, a DataFrame, or numpy
            ndarray. If a DataFrame, Data instance, ndarray, or string is
            passed, the resultant data object is stored in the 'data' attribute
            in a new Dataset instance as a DataFrame. Default is None.
        tasks (Optional[Union[Dict[str, 'Task']], List[str]]]): dictionary with
            keys as strings and values of 'Task' instances, or a list of tasks
            corresponding to keys in 'default_tasks' to use. Defaults to an
            empty dictionary. If nothing is provided, Project attempts to
            construct tasks from 'idea' and 'default_tasks'.
        library (Optional[Dict[str, 'Task']):  dictionary with keys as strings
            and values of 'Book' instances. Defaults to an empty dictionary. If
            not provided (the normal case), 'library' will be constructed from
            'tasks'.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'dataset' must also
            be passed. Defaults to False.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'project'.
        identification (Optional[str]): a unique identification name for this
            'Project' instance. The name is used for creating file folders
            related to the 'Project'. If not provided, a string is created from
            the date and time.

    """
    idea: Optional[Union['Idea', str]] = None
    inventory: Optional[Union['Inventory', str]] = None
    dataset: Optional[Union[
        'Dataset',
        pd.DataFrame,
        np.ndarray,
        str,
        Dict[str, Union[
            pd.DataFrame,
            np.ndarray,
            str]]]] = None
    tasks: Optional[Union[Dict[str, 'Task'], List[str]]] = field(
        default_factory = dict)
    library: Optional[Dict[str, 'Book']] = field(default_factory = dict)
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False
    name: Optional[str] = field(default_factory = lambda: 'project')
    identification: Optional[str] = field(default_factory = datetime_string)

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Validates 'Idea' instance and injects it into base classes.
        self.idea = Idea.create(idea = self.idea)
        self._spread_idea()
        # Validates 'Inventory' instance and injects it into base classes.
        self.inventory = Inventory.create(root_folder = self.inventory)
        self._spread_inventory()
        # Validates 'Dataset' instance.
        self.dataset = Dataset.create(data = self.dataset)
        # Initializes 'state' for use by various access methods.
        self.state = 'draft'
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        # Calls 'apply' method if 'auto_apply' is True.
        if self.auto_apply:
            self.apply()
        return self

    """ Required ABC Methods """

    def __getitem__(self, key: str) -> Union['Task', 'Book']:
        """Returns key from 'tasks' or 'library' depending upon 'state'.

        Args:
            key (str): key to item in 'tasks' or 'library'.

        Returns:
            Union['Task', 'Book']: 'Task' from 'tasks' or 'Book' from 'library'
                depending upon 'state'.

        """
        if self.state in ['draft', 'publish']:
            return self.tasks[key]
        elif self.state in ['apply']:
            return self.library[key]
        else:
            raise ValueError(
                'to get an item, state must be "draft", "publish", or "apply"')

    def __setitem__(self, key: str, value: Union['Task', 'Book']) -> None:
        """Sets key in 'tasks' or 'library', depending upon 'state'.

        Args:
            key (str): key to item in 'tasks' or 'library'.
            value (Union['Task', 'Book']): 'Task' or 'Book' instance to place
                in 'tasks' or 'library', depending upon 'state'.

        """
        if self.state in ['draft', 'publish']:
            self.tasks[key] = value
        elif self.state in ['apply']:
            self.library[key] = value
        else:
            raise ValueError(
                'to set an item, state must be "draft", "publish", or "apply"')
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes 'key' in 'tasks' or 'library', depending upon 'state'.

        Args:
            key (str): key in 'tasks' or 'library'.

        """
        try:
            if self.state in ['draft', 'publish']:
                del self.tasks[key]
            elif self.state in ['apply']:
                del self.library[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'tasks' or 'library', depending upon 'state'.

        Returns:
            Iterable: 'tasks' or 'library', depending upon 'state'.

        """
        if self.state in ['draft', 'publish']:
            return iter(self.tasks)
        elif self.state in ['apply']:
            return iter(self.library)
        else:
            raise ValueError(
                'to iterate, state must be "draft", "publish", or "apply"')

    def __len__(self) -> int:
        """Returns length of 'tasks' or 'library', depending upon 'state'.

        Returns:
            int: length of 'tasks' or 'library', depending upon 'state'.

        """
        if self.state in ['draft', 'publish']:
            return len(self.tasks)
        elif self.state in ['apply']:
            return len(self.library)
        else:
            raise ValueError(
                'to get length, state must be "draft", "publish", or "apply"')

    """ Other Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies Project.

        This requires an dataset argument to be passed to work properly.

        Calling Project as a function is compatible with and used by the
        command line interface.

        Raises:
            ValueError: if 'dataset' is not passed when Project is called as
                a function.

        """
        # Validates 'dataset'.
        if self.dataset is None:
            raise ValueError('Calling Project as a function requires a dataset')
        else:
            self.auto_apply = True
            self.__post__init()
        return self

    """ Private Methods """

    def _spread_idea(self) -> None:
        """Injects 'idea' into select base classes."""
        for simplify_class in [Inventory, Dataset, Task, Book]:
            setattr(simplify_class, 'idea', self.idea)
        return self

    def _spread_inventory(self) -> None:
        """Injects 'inventory' into select base classes."""
        for simplify_class in [Dataset, Book]:
            setattr(simplify_class, 'inventory', self.inventory)
        return self

    def _set_defaults(self) -> None:
        """Sets 'default_tasks' to use."""
        self.default_tasks = {
            'wrangle': Task(
                name = 'wrangler',
                module = 'simplify.wrangler.wrangler',
                book = 'Manual',
                worker = 'Wrangler',
                options = 'Mungers',
                import_folder = 'raw',
                export_folder = 'raw'),
            'explore': Task(
                name = 'explorer',
                module = 'simplify.explorer.explorer',
                book = 'Ledger',
                worker = 'Explorer',
                options = 'Measures'),
            'analyze': Task(
                name = 'analyst',
                module = 'simplify.analyst.analyst',
                book = 'Cookbook',
                chapter = 'Recipe',
                technique = 'AnalystTechnique',
                worker = 'Analyst',
                options = 'Tools'),
            'evaluate': Task(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Anthology',
                worker = 'Critic',
                options = 'Evaluators'),
            'visualize': Task(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas',
                worker = 'Artist',
                options = 'Mediums')}
        return self

    def _create_tasks(self,
            tasks: Union[Dict[str, 'Task'], List[str]]) -> Dict[str, 'Task']:
        """Creates or validates 'tasks'.

        Args:
            tasks (Union[Dict[str, 'Task'], List[str]]): set of 'Task' instances
                stored in dict or a list to create one from 'default_tasks'.

        Returns:
            Dict[str, 'Task']: mapping with stored 'Task' instances.

        """
        if not tasks:
            try:
                tasks = listify(
                    self.idea[self.name]['_'.join([self.name, 'tasks'])],
                    default_empty = True)
            except KeyError:
                pass
        if isinstance(tasks, dict) and tasks:
            return tasks
        elif not tasks:
            return self.default_tasks
        elif isinstance(tasks, list):
            new_tasks = {}
            for task in tasks:
                new_tasks[task] = self.default_tasks[task]
            return new_tasks

    def _load_task_attribute(self,
            tasks: Dict[str, 'Task'], attribute: str) -> Dict[str, 'Task']:
        """Creates instances for 'attribute' for each Task in 'tasks'.

        Args:
            tasks (Dict[str, 'Task']): stored Task instances.
            attribute (str): name of attribute to load and instance.

        Returns:
            Dict[str, 'Task']: instance with instances added.

        """
        # For each task, creates an instance at 'attribute'.
        new_tasks = {}
        for name, task in tasks.items():
            loaded = task.load(attribute)
            new_tasks[name] = loaded(idea = self.idea, task = task)
        return new_tasks

    """ Public Methods """

    def add(self,
            name: Optional[str] = None,
            task: Optional['Task'] = None,
            **kwargs) -> None:
        """Adds subpackage to 'tasks'.

        Args:
            name (Optional[str]): name of subpackage. This is used as both the
                key to the created Task in 'tasks' and as the 'name'
                attribute in the Task. Defaults to None. If not provided, the
                'task' will be added to 'tasks' with the key being the
                'name' attribute of 'task'.
            task (Optional['Task']): a completed instance. If not provided,
                the method will assume all of the parameters needed to construct
                a 'Task' instance are in 'kwargs'.
            **kwargs: other attributes of a 'Task' instance to pass.

        """
        if task is not None:
            if name is not None:
                self.tasks[name] = task
            else:
                self.tasks[task.name] = task
        elif name in self.default_tasks:
            self.tasks[name] = self.default_tasks[name]
        else:
            self.tasks[name] = Task(name = name, **kwargs)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Injects attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Sets 'default_tasks' to use in 'tasks' construction.
        self._set_defaults()
        # Creates 'Task' instances for each selected stage.
        self.tasks = self._create_tasks(tasks = self.tasks)
        self.tasks = self._load_task_attribute(
            tasks = self.tasks,
            attribute = 'publisher')
        # Iterates through 'tasks' and creates Book instances in 'tasks'.
        for name, task in self.tasks.items():
            # Drafts a Book instance for 'task'.
            self.tasks[name] = task.publisher.draft(task = task)
        return self

    def publish(self,
            tasks: Optional[Union[
                Dict[str, 'Task'], List[str]]] = None) -> None:
        """Finalizes iterable by creating Book instances.

        Args:
            tasks (Optional[Union[Dict[str, 'Task'], List[str]]]):
                alternative tasks to use. If not passed, the existing
                'tasks' attribute will be used. If passed, these will replace
                the local 'tasks' attribute. Defaults to None.

        """
        # Changes state.
        self.state = 'publish'
        # Assigns 'tasks' argument to 'tasks' attribute, if passed.
        if tasks is not None:
            self.tasks = self._create_tasks(tasks = tasks)
        # Iterates through 'tasks' and finalizes each Book instance. The
        # finalized instances are then placed in 'library'.
        for name, task in self.tasks.items():
            self.library[name] = task.publisher.publish(task = task)
        return self

    def apply(self, data: Optional['Dataset'] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Optional['Dataset']): data object for methods to be
                applied. If not passed, data stored in the 'dataset' is
                used.

        """
        # Changes state.
        self.state = 'apply'
        # Assigns 'data' to 'dataset' attribute and validates it.
        if data:
            self.dataset = Dataset.create(data = data)
        # Creates an iterable of 'workers' to apply Book instances to 'data'.
        self.tasks = self._load_task_attribute(
            tasks = self.tasks,
            attribute = 'worker')
        # Iterates through each task, creating and applying needed Books,
        # Chapters, and Techniques for each task in the Project.
        for name, book in self.library.items():
            self.library[name], self.dataset = self.tasks[name].worker.apply(
                book = book,
                data = self.dataset,
                library = self.library,
                **kwargs)
        return self


@dataclass
class Task(object):
    """Object construction instructions used by a Project instance.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
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
        worker (Optional[str]): name of Worker class in 'module' to load.
            Defaults to 'Worker'.
        steps (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union[str, Dict[str, Any]]]): a dictionary containing
            options for the 'Task' instance to utilize or a string corresponding
            to a dictionary in 'module' to load. Defaults to an empty
            dictionary.
        import_folder (Optional[str]): name of attribute in 'inventory' which
            contains the path to the default folder for importing data objects.
            Defaults to 'processed'.
        export_folder (Optional[str]): name of attribute in 'inventory' which
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
    worker: Optional[str] = field(default_factory = lambda: 'Worker')
    steps: Optional[List[str]] = field(default_factory = list)
    techniques: Optional[Dict[str, List[str]]] = field(default_factory = dict)
    options: Optional[Union[str, Dict[str, Any]]] = field(
        default_factory = dict)
    import_folder: Optional[str] = field(default_factory = lambda: 'processed')
    export_folder: Optional[str] = field(default_factory = lambda: 'processed')

    """ Public Methods """

    def load(self, component: str) -> object:
        """Returns 'component' from 'module'.

        Args:
            component (str): name of object to load from 'module'.

        Returns:
            object: from 'module'.

        """
        try:
            return getattr(
                import_module(self.module),
                getattr(self, component))
        except (ImportError, AttributeError):
            try:
                return getattr(
                    import_module(self.default_module),
                    getattr(self, component))
            except (ImportError, AttributeError):
                raise ImportError(' '.join(
                    [getattr(self, component), 'is neither in', self.module,
                        'nor', self.default_module]))