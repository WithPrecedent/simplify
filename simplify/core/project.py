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
from simplify.core.base import SimpleCatalog
from simplify.core.base import SimpleProgression
from simplify.core.book import Book
from simplify.core.editors import Author
from simplify.core.editors import Publisher
from simplify.core.ingredients import create_ingredients
from simplify.core.scholars import Scholar
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
            str],
            List[Union[
                'Ingredient',
                pd.DataFrame,
                np.ndarray,
                str]]]]] = None
    workers: Optional[Union['Workers', List[str], str]] = field(
        default_factory = list)
    library: Optional['Library'] = field(default_factory = Library)
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
        self.idea, self.inventory, self.ingredients, self.workers = (
            simplify.startup(
                idea = self.idea,
                inventory = self.inventory,
                ingredients = self.ingredients,
                workers = self.workers,
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
            dictionary = self.workers
        else:
            dictionary = self.library
        try:
            return dictionary[key]
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
            dictionary = self.workers
        else:
            dictionary = self.library
        dictionary[key] = value
        return self

    def __delitem__(self, key: str) -> None:
        """Deletes item in 'workers' or 'library', depending upon 'state'.

        Args:
            key (str): name of key in 'workers' or 'library', depending upon
                'state'.

        """
        if self.state in ['draft']:
            dictionary = self.workers
        else:
            dictionary = self.library
        try:
            del dictionary[key]
        except KeyError:
            pass
        return self

    def __iter__(self) -> Iterable:
        """Returns iterable of 'workers'.

        Returns:
            Iterable stored in 'workers'.

        """
        if self.state in ['draft']:
            dictionary = self.workers
        else:
            dictionary = self.library
        return (iter(dictionary.items()))

    def __len__(self) -> int:
        """Returns length of 'workers'.

        Returns:
            Integer of length of 'workers'.

        """
        if self.state in ['draft']:
            dictionary = self.workers
        else:
            dictionary = self.library
        return len(dictionary)

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

    """ Private Methods """

    def _create_workers(self,
            workers: Optional[Union[List[str], str]] = None) -> None:
        """Creates SimpleEditor instances for each Worker."""
        workers = workers or self.workers
        for name, worker in workers.items():
            # For each worker, create an Author, Publisher, and Scholar instance
            # to draft, publish, and apply Book instances.
            workers[name].author = self.workers[name].author(
                project = self,
                worker = worker)
            workers[name].publisher = self.workers[name].publisher(
                project = self,
                worker = worker)
            workers[name].scholar = self.workers[name].scholar(
                project = self,
                worker = worker)
        return workers

    """ Public Methods """

    def add_worker(self, name: str, **kwargs) -> None:
        """Adds subpackage to 'workers'.

        Args:
            name (str): name of subpackage. This is used as both the key to the
                created Worker in 'workers' and as the 'name' attribute in the
                Worker.
            **kwargs: other attributes of a 'Worker' instance to pass.

        """
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
                scholar = 'Chef',
                options = 'ChefCatalog'),
            'farmer': Worker(
                name = 'farmer',
                module = 'simplify.farmer.farmer',
                book = 'Almanac',
                scholar = 'Farmer',
                options = 'FarmerCatalog'),
            'actuary': Worker(
                name = 'actuary',
                module = 'simplify.actuary.actuary',
                book = 'Ledger',
                scholar = 'Actuary',
                options = 'ActuaryCatalog'),
            'critic': Worker(
                name = 'critic',
                module = 'simplify.critic.critic',
                book = 'Collection',
                scholar = 'Critic,
                options = 'CriticCatalog'),
            'artist': Worker(
                name = 'artist',
                module = 'simplify.artist.artist',
                book = 'Canvas',
                scholar = 'Artist',
                options = 'ArtistCatalog')}
        self.editors = {
            'author' : 'draft',
            'publisher': 'publish',
            'scholar': 'apply'}
        # Creates 'Worker' instances for each selected stage.
        self.workers = self._create_workers()
        # Iterates through 'workers' and creates a skeleton of each Book.
        for name, worker in self.workers.items():
            # Drafts a Book instance for 'worker' and places it in 'library'.
            self.library[name] = worker.author.draft()
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
            self.workers = create_workers(workers = workers)
        # Injects attributes from 'idea'.
        self = self.idea.apply(instance = self)
        # Iterates through 'workers' and finalizes each Book instance.
        for name, worker in self.workers.items():
            self.library[name] = worker.publisher.publish()
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
        # Assigns 'data' to 'ingredients' attribute and validates it.
        if data:
            self.ingredients = create_ingredients(ingredients = data)
        # Iterates through each worker, creating and applying needed Books,
        # Chapters, and Techniques for each worker in the Project.
        for name, worker in self.workers.items():
            if worker.book.returns_data:
                self.ingredients = worker.scholar.apply(
                    book = self.library[name],
                    data = self.ingredients,
                    **kwargs)
            else:
                worker.scholar.apply(
                    book = self.library[name],
                    data = self.ingredients,
                    **kwargs)
        return self



# @dataclass
# class Library(MutableMapping):
#     """A state-dependent dictionary.

#     Args:


#     """

#     project: 'Project'
#     workers: Optional[Dict[str, 'Worker']] = field(default_factory = dict)
#     books: Optional[Dict[str, 'Book']] = field(default_factory = dict)

#     def __post_init__(self) -> None:
#         self.stages = {
#             'draft': 'workers',
#             'publish': 'books',
#             'apply': 'books'}
#         return self

#     """ Required ABC Methods """

#     def __getitem__(self, key: str) -> Any:
#         """Returns value for 'key' in the active dictionary.

#         Args:
#             key (str): name of key in the active dictionary.

#         Returns:
#             Any: item stored the active dictionary.

#         """
#         try:
#             return getattr(self, self.stages[self.project.stage])[key]
#         except KeyError:
#             if key in ['all']:
#                 return getattr(self, self.stages[self.project.stage])
#             else:
#                 raise KeyError(' '.join([key, 'is not in the library']))

#     def __delitem__(self, key: str) -> None:
#         """Deletes 'key' entry in the active dictionary.

#         Args:
#             key (str): name of key in the active dictionary.

#         """
#         try:
#             del getattr(self, self.stages[self.project.stage])[key]
#         except KeyError:
#             pass
#         return self

#     def __setitem__(self, key: str, value: Any) -> None:
#         """Sets 'key' in the active dictionary to 'value'.

#         Args:
#             key (str): name of key in the active dictionary.
#             value (Any): value to be paired with 'key' in the active dictionary.

#         """
#         getattr(self, self.stages[self.project.stage])[key] = value
#         return self

#     def __iter__(self) -> Iterable:
#         """Returns iterable of the active dictionary.

#         Returns:
#             Iterable stored in the active dictionary.

#         """
#         return iter(getattr(self, self.stages[self.project.stage]).items())

#     def __len__(self) -> int:
#         """Returns length of the active dictionary.

#         Returns:
#             Integer: length of the active dictionary.

#         """
#         return len(getattr(self, self.stages[self.project.stage]))

#     """ Other Dunder Methods """

#     def __add__(self,
#             other: Union[
#                 'Library', Dict[str, Union['Worker', 'Book']]]) -> None:
#         """Combines argument with the active dictionary.

#         Args:
#             other (Union['Library', Dict[str, Union['Worker', 'Book']]]):
#                 another 'Library' instance or compatible dictionary.

#         """
#         self.add(library = other)
#         return self

#     def __iadd__(self,
#             other: Union[
#                 'Library', Dict[str, Union['Worker', 'Book']]]) -> None:
#         """Combines argument with the active dictionary.

#         Args:
#             other (Union['Library', Dict[str, Union['Worker', 'Book']]]):
#                 another 'Library' instance or compatible dictionary.

#         """
#         self.add(library = other)
#         return self

#     """ Public Methods """

#     def add(self,
#             key: Optional[str] = None,
#             value: Optional[Any] = None,
#             options: Optional[Union[
#                 'Library', Dict[str, Union['Worker', 'Book']]]] = None) -> None:
#         """Combines arguments with the active dictionary.

#         Args:
#             key (Optional[str]): options key for 'value' to use. Defaults to
#                 None.
#             value (Optional[Any]): item to store in the active dictionary.
#                 Defaults to None.
#             library (Optional[Union['Library', Dict[str, Union['Worker',
#                 'Book']]]]): another 'Library' instance or compatible
#                 dictionary. Defaults to None.

#         """
#         if key is not None and value is not None:
#             getattr(self, self.stages[self.project.stage])[key] = value
#         if library is not None:
#             try:
#                 getattr(self, self.stages[self.project.stage]).update(
#                     getattr(library, self.stages[self.project.stage])
#             except AttributeError:
#                 try:
#                     getattr(self, self.stages[self.project.stage]).update(
#                         library)
#                 except (TypeError, AttributeError):
#                     pass
#         return self

@dataclass
def Library(SimpleProgression):

    options: Optional[Dict[str, 'Book']] = field(default_factory = dict)
    order: Optional[List[str]] = field(default_factory = list)

    """ Other Dunder Methods """

    def __add__(self, other: 'Book') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Book'): a 'Book' instance.

        """
        self.add(book = other)
        return self

    def __iadd__(self, other: 'Book') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Book'): a 'Book' instance.

        """
        self.add(book = other)
        return self

    """ Public Methods """

    def add(self, book: 'Book', key: Optional[str] = None) -> None:
        """Combines arguments with 'options'.

        Args:
            book ('Book'): a 'Book' instance.
            key (Optional[str]): key name to link to 'book'. If not passed,
                 the 'name' attribute of 'book' will be used.

        """
        if key:
            self.options[key] = book
        else:
            self.options[book.name] = book
        self.order.append(key)
        return self


@dataclass
def Workers(SimpleProgression):

    options: Optional[Dict[str, 'Worker']] = field(default_factory = dict)
    order: Optional[List[str]] = field(default_factory = list)

    """ Other Dunder Methods """

    def __add__(self, other: 'Worker') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Worker'): a 'Worker' instance.

        """
        self.add(worker = other)
        return self

    def __iadd__(self, other: 'Worker') -> None:
        """Combines argument with 'options'.

        Args:
            other ('Worker'): a 'Worker' instance.

        """
        self.add(worker = other)
        return self

    """ Public Methods """

    def add(self, worker: 'Worker', key: Optional[str] = None) -> None:
        """Combines arguments with 'options'.

        Args:
            worker ('Worker'): a 'Worker' instance.
            key (Optional[str]): key name to link to 'worker'. If not passed,
                 the 'name' attribute of 'worker' will be used.

        """
        if key:
            self.options[key] = worker
        else:
            self.options[worker.name] = worker
        return self


@dataclass
def Worker(SimpleOutline):
    """Object construction techniques used by SimpleEditor instances.

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
    module: str
    book: Optional['Book'] = Book
    author: Optional['Author'] = Author
    publisher: Optional['Publisher'] = Publisher
    scholar: Optional['Scholar'] = Scholar
    steps: Optional[List[str]] = field(default_factory = list)
    options: Optional['SimpleCatalog'] = field(default_factory = SimpleCatalog)
    techniques: Dict[str, List[str]] = field(default_factory = dict)


""" Validator Function """

def create_workers(
        workers: Union['Workers', List[str], str],
        project: 'Project') -> 'Workers':
    """Creates or validates 'workers'.

    Args:
        workers: (Union['Workers', List[str], str]): either a 'Workers' instance,
            a list of workers, or a single worker.
        project ('Project'): a related 'Project' instance with a
            'default_workers' dictionary.

    Returns:
        'Workers': an instance derived from 'workers' and/or 'project'.

    """
    if isinstance(workers, Workers):
        return workers
    elif isinstance(workers, (list, str)) and workers:
        new_workers = {}
        for worker in listify(workers):
            new_workers[worker] = project.default_workers[worker]
        return Workers(options = new_workers)
    else:
        return Workers(options = project.default_workers)