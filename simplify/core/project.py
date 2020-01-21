"""
.. module:: siMpLify project
:synopsis: controller class for siMpLify projects
:editor: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.core.book import Book
from simplify.core.editors import Author
from simplify.core.editors import Publisher
from simplify.core.ingredients import create_ingredients
from simplify.core.repository import Sequence
from simplify.core.scholar import Scholar
from simplify.core.types import Outline
from simplify.core.utilities import datetime_string
from simplify.core.utilities import listify


@dataclass
class Project(MutableMapping):
    """Controller class for siMpLify projects.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supported file type with settings for an Idea instance is
            located. Even though a default value is provided (None) to lessen
            subclass MRO problems, 'idea' is required for a Project to work.
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
        workers (Optional[List[str], str]): ordered list of workers to call.
            Each worker should match a key in 'workers'. Defaults to an empty
            list. If no 'workers' are provided, a Project instance attempts to
            find them in 'idea'.
        library (Optional[Dict[str, Union['Worker', 'Book']]]): attribute
            which stores the siMpLify objects which are iterated by Project. If
            not passed or later populated, the internally stored
            'default_workers' is used when 'publish' is called. Defaults to an
            empty dictionary.
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
    idea: Union['Idea', str] = None
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
    workers: Optional[Union['Sequence', List[str], str]] = field(
        default_factory = Sequence)
    library: Optional['Sequence'] = field(default_factory = Sequence)
    auto_publish: Optional[bool] = True
    auto_apply: Optional[bool] = False
    name: Optional[str] = 'project'
    identification: Optional[str] = field(default_factory = datetime_string)

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods.

        Raises:
            ValueError: if 'idea' is None.

        """
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Checks 'idea' to make sure it was passed.
        if self.idea is None:
            raise ValueError('Project requires an idea argument')
        # Validates 'idea', 'inventory', and 'ingredients'.
        self.idea, self.inventory, self.ingredients = (
            simplify.startup(
                idea = self.idea,
                inventory = self.inventory,
                ingredients = self.ingredients,
                project = self))
        # Initializes 'state' for use by lookup methods.
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

    def __getitem__(self, key: str) -> 'Worker':
        """Returns item in 'workers' or 'library', depending upon 'state'.

        Args:
            key (str): name of key in 'workers' or 'library', depending upon
                'state'.

        Returns:
            'Worker': item stored in 'workers' or 'library', depending upon
                'state'.

        Raises:
            KeyError: if 'key' is not found in 'workers' or 'library', depending
                upon 'state'.

        """
        if self.state in ['draft']:
            contents = self.workers
        else:
            contents = self.library
        try:
            return contents[key]
        except KeyError:
            raise KeyError(' '.join([key, 'is not in', self.name]))

    def __setitem__(self, key: str, value: 'Worker') -> None:
        """Sets 'key' to 'value' in 'workers' or 'library'.

        Args:
            key (str): name of key in 'workers' or 'library', depending upon
                'state'.
            value ('Worker'): value to be paired with 'key' in 'workers' or
                'library', depending upon 'state'.

        """
        if self.state in ['draft']:
            contents = self.workers
        else:
            contents = self.library
        contents[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes item in 'workers' or 'library', depending upon 'state'.

        Args:
            key (str): name of key in 'workers' or 'library', depending upon
                'state'.

        """
        if self.state in ['draft']:
            contents = self.workers
        else:
            contents = self.library
        try:
            del contents[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'workers'.

        Returns:
            Iterable stored in 'workers'.

        """
        if self.state in ['draft']:
            contents = self.workers
        else:
            contents = self.library
        return iter(contents)

    def __len__(self) -> int:
        """Returns length of 'workers'.

        Returns:
            Integer of length of 'workers'.

        """
        if self.state in ['draft']:
            contents = self.workers
        else:
            contents = self.library
        return len(contents)

    """ Other Dunder Methods """

    def __call__(self) -> Callable:
        """Drafts, publishes, and applies Project.

        This requires idea and ingredients arguments to be passed to work
        properly.

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ' '.join(['Project Library:', *list(self.keys())])

    """ Private Methods """

    def _create_workers(self,
            workers: Union['Sequence', List[str], str]) -> None:
        """Creates or validates 'workers'.

        Args:
            workers (Union['Sequence', List[str], str]): key(s) for Worker
                instances in 'default_workers' or a completed Sequence
                instance with Worker instances.

        """
        if not workers.contents:
            try:
                # Attempts to get 'workers' from 'idea'.
                workers = listify(self.idea[self.name]['_'.join(
                    [self.name, 'workers'])])
            except KeyError:
                pass
        if isinstance(workers, (list, str)) and workers:
            new_workers = {}
            for worker in listify(workers):
                new_workers[worker] = self.default_workers[worker]
            return Sequence(contents = new_workers)
        elif isinstance(workers, Sequence) and workers:
            return workers
        else:
            return Sequence(contents = self.default_workers)

    def _create_editors(self, workers: 'Sequence') -> None:
        """Creates Editor instances for each Worker.

        Args:
            workers ('Sequence'): stored Worker instances.

        """
        for name, worker in workers.items():
            # For each worker, creates an Author and Publisher instance.
            author = worker.load('author')
            workers[name].author = author(project = self, worker = name)
            publisher = worker.load('publisher')
            workers[name].publisher = publisher(project = self, worker = name)
        return workers

    """ Public Methods """

    def add(self,
            name: str,
            worker: Optional['Worker'] = None,
            **kwargs) -> None:
        """Adds subpackage to 'workers'.

        Args:
            name (str): name of subpackage. This is used as both the key to the
                created Worker in 'workers' and as the 'name' attribute in the
                Worker.
            worker (Optional['Worker']): a completed instance. If not provided,
                the method will assume all of the parameters needed to construct
                a 'Worker' instance are in 'kwargs'.
            **kwargs: other attributes of a 'Worker' instance to pass.

        """
        if worker:
            self.workers[name] = worker
        else:
            self.workers[name] = Worker(name = name, **kwargs)
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Sets default package options available to Project.
        self.default_workers = {
            'chef': Worker(
                name = 'chef',
                module = 'simplify.chef.chef',
                book = 'Cookbook',
                options = 'Cookware'),
            'farmer': Worker(
                name = 'farmer',
                module = 'simplify.farmer.farmer',
                book = 'Almanac',
                options = 'Mungers'),
            'actuary': Worker(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                book = 'Ledger',
                options = 'Measures'),
            'critic': Worker(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Collection',
                options = 'Evaluators'),
            'artist': Worker(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas',
                options = 'Mediums')}
        # Creates 'Worker' instances for each selected stage.
        self.workers = self._create_workers(workers = self.workers)
        self.workers = self._create_editors(workers = self.workers)
        # Iterates through 'workers' and creates a skeleton of each Book.
        for name, worker in self.workers.items():
            # Drafts a Book instance for 'worker' and places it in 'library'.
            worker.author.draft()
        return self

    def publish(self, workers: Optional[Union[List[str], str]] = None) -> None:
        """Finalizes iterable by creating Book instances.

        Args:
            workers (Optional[Union[List[str], str]]): option(s) to publish. If
                not passed, the existing 'workers' attribute will be used.

        """
        # Changes state.
        self.state = 'publish'
        # Assigns 'workers' argument to 'workers' attribute, if passed.
        if workers is not None:
            self.workers = self._create_workers(workers = workers)
        # Injects attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Iterates through 'workers' and finalizes each Book instance.
        for name, worker in self.workers.items():
            worker.publisher.publish()
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
        # Creates a Scholar instance to apply Book instances to 'data'.
        scholar = Scholar(project = self)
        # Assigns 'data' to 'ingredients' attribute and validates it.
        if data:
            self.ingredients = create_ingredients(ingredients = data)
        # Iterates through each worker, creating and applying needed Books,
        # Chapters, and Techniques for each worker in the Project.
        for name, book in self.library.items():
            self.library[name] = scholar.apply(
                book = book,
                data = self.ingredients)
        return self


@dataclass
class Worker(Outline):
    """Object construction techniques used by Editor instances.

    Ideally, this class should have no additional methods beyond the lazy
    loader ('load' method) and __contains__ dunder method.

    Users can use the idiom 'x in Option' to check if a particular attribute
    exists and is not None. This means default values for optional arguments
    should generally be set to None to allow use of that idiom.

    Args:
        name (str): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier.
        module (str): name of module where object to incorporate is located
            (can either be a siMpLify or non-siMpLify module).
        book (str): name of Book object in 'module' to load. Defaults to None.

    """
    name: str
    module: Optional[str] = 'simplify.core'
    book: Optional['Book'] = 'Book'
    author: Optional['Author'] = 'Author'
    publisher: Optional['Publisher'] = 'Publisher'
    scholar: Optional['Scholar'] = 'Scholar'
    steps: Optional[List[str]] = field(default_factory = list)
    options: Optional['Sequence'] = field(default_factory = Sequence)
    techniques: Dict[str, List[str]] = field(default_factory = dict)