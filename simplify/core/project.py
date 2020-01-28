"""
.. module:: siMpLify project
:synopsis: controller class for siMpLify projects
:publisher: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.core.definitions import Outline
from simplify.core.ingredients import create_ingredients
from simplify.core.repository import Plan
from simplify.core.repository import Repository
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify
from simplify.core.worker import Worker


@dataclass
class Project(Iterable):
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
        ingredients (Optional[Union['Ingredients', 'Ingredient', pd.DataFrame,
            np.ndarray, str]]): an instance of Ingredients, an instance of
            Ingredient, a string containing the full file path where a data file
            for a pandas DataFrame is located, a string containing a file name
            in the default data folder (as defined in the shared Inventory
            instance), a full folder path where raw files for data to be
            extracted from, a string containing a folder name which is an
            attribute in the shared Inventory instance, a DataFrame, or numpy
            ndarray. If a DataFrame, Ingredient instance, ndarray, or string is
            passed, the resultant data object is stored in the 'data' attribute
            in a new Ingredients instance as a DataFrame. Default is None.
        tasks (Optional[Union['Plan', Dict[str, 'Task']], List[str]]]):
            MutableMapping with keys as strings and values of 'Task' instances,
            or a list of tasks corresponding to keys in 'DefaultTasks' to use.
            Defaults to an empty Plan. If nothing is provided, Project attempts
            to construct tasks from 'idea' and 'DefaultTasks'.
        library (Optional[Union['Plan', Dict[str, 'Task']]]):  MutableMapping
            with keys as strings and values of 'Book' instances. Defaults to
            en empty Plan. If not provided (the normal case), 'library' will
            be constructed from 'tasks'.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            instanced. Defaults to True.
        auto_apply (Optional[bool]): whether to call the 'apply' method when
            instanced. For auto_apply to have an effect, 'ingredients' must also
            be passed. Defaults to False.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'project'.

    """
    idea: Optional[Union['Idea', str]] = None
    inventory: Optional[Union['Inventory', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        'Ingredient',
        pd.DataFrame,
        np.ndarray,
        str,
        Dict[str, Union[
            'Ingredient',
            pd.DataFrame,
            np.ndarray,
            str]],
        List[Union[
            'Ingredient',
            pd.DataFrame,
            np.ndarray,
            str]]]] = None
    tasks: Optional[Union['Plan', Dict[str, 'Task'], List[str]]] = field(
        default_factory = Plan)
    library: Optional[Union['Plan', Dict[str, 'Book']]] = field(
        default_factory = Plan)
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False
    name: Optional[str] = field(default_factory = lambda: 'project')
    identification: Optional[str] = field(default_factory = datetime_string)

    def __post_init__(self) -> None:
        """Initializes class attributes and calls selected methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Validates 'idea', 'inventory', and 'ingredients'.
        self.idea, self.inventory, self.ingredients = (
            simplify.startup(
                idea = self.idea,
                inventory = self.inventory,
                ingredients = self.ingredients,
                project = self))
        # Initializes 'state' for use by the '__iter__' method.
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

    def __iter__(self) -> Iterable:
        """Returns iterable of 'tasks' or 'library' depending upon 'state'.

        Returns:
            Iterable: 'tasks' or 'library' depending upon 'state'.

        """
        if self.state in ['draft', 'publish']:
            return iter(self.tasks)
        elif self.state in ['apply']:
            return iter(self.library)
        else:
            raise ValueError(
                'to iterate, state must be "draft", "publish", or "apply"')

    """ Other Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies Project.

        This requires an ingredients argument to be passed to work properly.

        Calling Project as a function is compatible with and used by the
        command line interface.

        Raises:
            ValueError: if 'ingredients' is not passed when Project is called as
                a function.

        """
        # Validates 'ingredients'.
        if self.ingredients is None:
            raise ValueError(
                'Calling Project as a function requires ingredients')
        else:
            self.auto_apply = True
            self.__post__init()
        return self

    """ Private Methods """

    def _draft_tasks(self,
            tasks: Union['Plan', Dict[str, 'Task'], List[str]]) -> 'Plan':
        """Creates or validates 'tasks'.

        Args:
            tasks (tasks: Union['Plan', Dict[str, 'Task'], List[str]]):
                Plan instance or information for making one.

        Returns:
            'Plan': instance with iterable Task instances.

        """
        if isinstance(tasks, Plan):
            tasks.project = self
            return tasks
        elif isinstance(tasks, dict):
            return Plan(contents = tasks, project = self)
        elif isinstance(tasks, list):
            return DefaultTasks(steps = tasks, project = self)
        else:
            return DefaultTasks(project = self)

    def _draft_editors(self, tasks: 'Plan') -> 'Plan':
        """Creates Publisher and Worker instances for each Task.

        Args:
            tasks ('Plan'): stored Task instances.

        Returns:
            'Plan': instance with Publisher and Worker instances added to
                each stored 'Task' instance.

        """
        # For each task, creates an Publisher and Worker instance.
        for task in tasks.keys():
            publisher = tasks[task].load('publisher')
            tasks[task].publisher = publisher(project = self, task = task)
            worker = tasks[task].load('worker')
            tasks[task].worker = worker(project = self, task = task)
        return tasks

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
        # Creates 'Task' instances for each selected stage.
        self.tasks = self._draft_tasks(tasks = self.tasks)
        self.tasks = self._draft_publishers(tasks = self.tasks)
        # Iterates through 'tasks' and creates Book instances in 'tasks'.
        for task in self.tasks.keys():
            # Drafts a Book instance for 'task'.
            self.tasks[task].publisher.draft()
        return self

    def publish(self,
            tasks: Optional[Union[
                'Plan', Dict[str, 'Task'], List[str]]] = None) -> None:
        """Finalizes iterable by creating Book instances.

        Args:
            tasks (Optional[Union['Plan', Dict[str, 'Task'], List[str]]]):
                alternative tasks to use. If not passed, the existing
                'tasks' attribute will be used. If passed, these will replace
                the local 'tasks' attribute. Defaults to None.

        """
        # Changes state.
        self.state = 'publish'
        # Assigns 'tasks' argument to 'tasks' attribute, if passed.
        if tasks is not None:
            self.tasks = self._draft_tasks(tasks = tasks)
        # Iterates through 'tasks' and finalizes each Book instance.
        for task in self.tasks.keys():
            self.tasks[task].publisher.publish()
        return self

    def apply(self, data: Optional['Ingredients'] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Optional['Ingredients']): data object for methods to be
                applied. If not passed, data stored in the 'ingredients' is
                used.

        """
        # Changes state.
        self.state = 'apply'
        # Deletes 'tasks' to save memory, if option is selected.
        # if self.conserve_memory:
        #     del self.tasks
        # Assigns 'data' to 'ingredients' attribute and validates it.
        if data:
            self.ingredients = create_ingredients(ingredients = data)
        # Iterates through each task, creating and applying needed Books,
        # Chapters, and Techniques for each task in the Project.
        for key, book in self.library.items():
            self.library[key] = self.tasks[key].worker.apply(
                book = book,
                data = self.ingredients,
                **kwargs)
        return self


@dataclass
class Task(Outline):
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
        publisher (Optional[str]): name of Publisher class in 'module' to load.
            Defaults to
            'Publisher'.
        worker (Optional[str]): name of Worker class in 'module' to load.
            Defaults to 'Worker'.
        steps: (Optional[List[str]]): list of steps to execute. Defaults to an
            empty list.
        options (Optional[Union['Repository', str]]): a 'Repositor'y containing
            options for the Task instance to utilize or a string corresponding
            to a 'Repository' subclass in 'module' to load. Defaults to an
            empty 'Repository' instance.

    """
    name: str
    module: Optional[str] = field(default_factory = lambda: 'simplify.core')
    book: Optional[str] = field(default_factory = lambda: 'Book')
    publisher: Optional[str] = field(default_factory = lambda: 'Publisher')
    worker: Optional[str] = field(default_factory = lambda: 'Worker')
    steps: Optional[List[str]] = field(default_factory = list)
    techniques: Optional[Dict[str, List[str]]] = field(default_factory = dict)
    options: Optional[Union['Repository', str]] = field(
        default_factory = Repository)


@dataclass
class DefaultTasks(Plan):
    """Default tasks for a Project.

    To limit the options selected, pass a list of selected tasks to 'steps'.

    Args:
        steps (Optional[List[str]]): an ordred set of steps. Defaults to an
            empty list. All items in 'steps' should correspond to keys in
            'contents' before iterating.
        contents (Optional[Union['Repository', Dict[str, Any]]]): a 'Repository'
            instance or a dictionary that can be used to create one. Defaults to
            an empty Repository.
        defaults (Optional[List[str]]): a list of keys in 'contents' which
            will be used to return items when 'default' is sought. If not
            passed, 'default' will be set to all keys.
        iterable (Optional[str]): the name of the attribute that should be
            iterated when a class instance is iterated. Defaults to 'contents'.
        project ('Project'): a related 'Project' instance.

    """
    steps: Optional[List[str]] = field(default_factory = list)
    contents: Optional[Union['Repository', Dict[str, Any]]] = field(
        default_factory = Repository)
    defaults: Optional[List[str]] = field(default_factory = list)
    iterable: Optional[str] = field(default_factory = lambda: 'steps')
    project: 'Project' = None

    def _create_contents(self) -> None:
        self.contents = {
            'organize': Task(
                name = 'wrangler',
                module = 'simplify.wrangler.wrangler',
                book = 'Manual',
                worker = 'Wrangler',
                options = 'Mungers'),
            'analyze': Task(
                name = 'analyst',
                module = 'simplify.analyst.analyst',
                book = 'Cookbook',
                worker = 'Analyst',
                options = 'Tools'),
            'summarize': Task(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                book = 'Ledger',
                worker = 'Actuary',
                options = 'Measures'),
            'criticize': Task(
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