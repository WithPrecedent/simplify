"""
.. module:: book
:synopsis: iterable builder, container, and steps
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from simplify import factory
from simplify.core.utilities import create_proxies
from simplify.core.utilities import listify
from simplify.core.utilities import numpy_shield
from simplify.core.utilities import XxYy


@dataclass
class SimpleManuscript(ABC):
    """Base class for data processing classes.

    SimpleManuscript implements a modified composite tree pattern for organizing
    Projects, Books, Chapters, and Pages. The hierarchy between each level is
    fixed, but the core methods are shared by all of the levels in that
    hierarchy.

    """
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        self.draft()
        if hasattr(self, 'proxies'):
            self = create_proxies(instance = self, proxies = self.proxies)
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable 'children'."""
        try:
            return iter(self.children)
        except AttributeError:
            return iter()

    """ Private Methods """

    def _convert_wildcards(self, value: Union[str, List[str]]) -> List[str]:
        """Converts 'all', 'default', or 'none' values to a list of items.

        Args:
            value (Union[str, List[str]]): name(s) of pages.

        Returns:
            If 'all', either the 'all' property or all keys listed in 'options'
                dictionary are returned.
            If 'default', either the 'defaults' property or all keys listed in
                'options' dictionary are returned.
            If some variation of 'none', 'none' is returned.
            Otherwise, 'value' is returned intact.

        """
        if value in ['all', ['all']]:
            return self.all
        elif value in ['default', ['default']]:
            self.default
        elif value in ['none', ['none'], 'None', ['None'], None]:
            return ['none']
        else:
            return listify(value)

    def _draft_options(self) -> None:
        """Declares 'options' dict.

        Subclasses should provide their own '_draft_options' method, if needed.

        """
        self.options = {}
        return self

    """ Import/Export Methods """

    def load(self,
            name: Optional[str] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Loads object from file into the subclass attribute 'name'.

        For any arguments not passed, default values stored in the shared
        Library instance will be used based upon the current 'stage' of the
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
                library.extensions. Defaults to None.

        """
        setattr(self, name, self.library.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format))
        return self

    def save(self,
            variable: Optional[Union['SimpleManuscript', str]] = None,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None,
            file_format: Optional[str] = None) -> None:
        """Exports a variable or attribute to disk.

        If 'variable' is not passed, 'self' will be used.

        For other arguments not passed, default values stored in the shared
        library instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            variable (Optional[Union['SimpleManuscript'], str]): a python object
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
                library.extensions. Defaults to None.

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
        self.library.save(
            variable = variable,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = file_format)
        return self

    """ Composite Management Methods """

    def add_parent(self, parent: 'SimpleManuscript') -> None:
        """Sets 'parent' attribute to 'SimpleManuscript'.

        Setting 'parent' allows different SimpleManuscript instances to access
        attributes from other SimpleManuscript instances that are connected
        through parent and/or child relationships.

        Args:
            parent ('SimpleManuscript'): SimpleManuscript subclass above another
                SimpleManuscript subclass instance in the composite tree.

        """
        self.parent = parent
        return self

    def remove_parent(self) -> None:
        """Sets 'parent' to None."""
        self.parent = None
        return self

    def add_child(self, name: str, child: 'SimpleManuscript') -> None:
        """Adds 'child' instance to 'children'.

        Args:
            name (str): key name for child instance to be accessed from
                'children' dict.
            child ('SimpleManuscript'): SimpleManuscript subclass below another
                SimpleManuscript subclass instance in the composite tree.

        """
        try:
            self.children[name] = child
        except (AttributeError, TypeError):
            self.children = {}
            self.children[name] = child
        return self

    def remove_child(self, name: str) -> None:
        """Removes a child instance link from 'children'.

        Args:
            name (str): key name of child instance to remove.

        """
        try:
            del self.children[name]
        except KeyError:
            pass
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values.

        Subclasses should provide their own 'draft' method.

        """
        return self

    @abstractmethod
    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        return self

    @abstractmethod
    def apply(self, data: 'Ingredients', **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (Ingredients): data object for methods to be applied.

        """
        return self

    """ Properties """

    @property
    def all(self) -> List[str]:
        """Returns list of all 'options'."""
        return list(options.keys())

    @property
    def defaults(self) -> List[str]:
        """Returns default 'options'.

        If no default options have been set, 'all' is returned.

        """
        try:
            self._defaults
        except AttributeError:
            return self.all

    @defaults.setter
    def defaults(self, defaults: List[str]) -> None:
        """Sets default 'options'."""
        self._defaults = defaults
        return self


@dataclass
class Book(SimpleManuscript):
    """Builds and controls Chapters.

    This class contains methods useful to create iterators and iterate over
    passed arguments based upon user-selected options. Book subclasses construct
    iterators and process data with those iterators.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        library (Optional[Union['Library', str]]): an instance of
            library or a string containing the full path of where the root
            folder should be located for file output. A library instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Library instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.

    """
    idea: Union['Idea', str]
    library: Optional[Union['Library', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True
    file_format: str = 'pickle'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        self.parent_type = 'project'
        self.children_type = 'chapters'
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        self.idea, self.library, self.ingredients = factory.startup(
            idea = self.idea,
            library = self.library,
            ingredients = self.ingredients)
        self.draft()
        if self.auto_publish and self.ingredients is not None:
            self.publish(data = self.ingredients)
        return self

    """ Private Methods """

    def _draft_steps(self) -> None:
        """If 'steps' is not passed, gets 'steps' from Idea settings.

        If there are no 'steps' in the shared Idea instance, an empty list is
        created for 'steps'.

        """
        if self.steps is None:
            try:
                self.steps = getattr(self, '_'.join([self.name, 'steps']))
            except AttributeError:
                self.steps = []
        else:
            self.steps = listify(self.steps)
        return self

    def _draft_contributors(self) -> None:
        """Creates 'contributors' containing SimpleContributor instances."""
        self.contributors  = {}
        for step in self.steps:
            try:
                contributor = getattr(
                    import_module(self.options[step][0]),
                    self.options[step][1])
                self.add_contributor(name = step, contributor = contributor)
            except KeyError:
                error = ' '.join(
                    [step, 'does not match an option in', self.name])
                raise KeyError(error)
        return self

    def _draft_plans(self) -> None:
        """Creates cartesian product of all possible 'chapters'."""
        plans = []
        for step in self.steps:
            try:
                key = '_'.join([step, 'techniques'])
                plans.append(listify(self.idea[self.name][key]))
            except AttributeError:
                plans.append(['none'])
        self.plans = list(map(list, product(*plans)))
        return self

    def _draft_chapters(self) -> None:
        """Converts 'plans' from list of lists to Chapter instances."""
        if not hasattr(self, 'chapters'):
            self.chapters = {}
        if not hasattr(self, 'chapter_type'):
            self.chapter_type = Chapter
        for i, plan in enumerate(self.plans):
            pages = dict(zip(self.steps, plan))
            metadata = self._draft_chapter_metadata(number = i)
            self.add_children(name = str(i), pages = pages, metadata = metadata)
        return self

    def _draft_chapter_metadata(self, number: int) -> Dict[str, Any]:
        """Finalizes metadata for Chapter instance.

        Args:
            number (int): chapter number; used for recordkeeping.

        Returns:
            Dict[str, Any]: metadata dict.

        """
        metadata = {'number': number + 1}
        try:
            metadata.update(self.metadata)
        except AttributeError:
            pass
        return metadata

    def _extra_processing(self, chapter: 'Chapter') -> 'Chapter':
        """Extra actions to take for each Chapter processed.

        Subclasses should provide _extra_processing methods, if needed.

        Returns:
            'Chapter' with any modifications made.

        """
        return chapter

    def _publish_contributors(self,
            data: Optional['Ingredients'] = None) -> None:
        """Converts contributor classes into class instances.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        new_contributors = {}
        for key, contributor in self.contributors.items():
            instance = contributor(idea = self.idea, library = self.library)
            instance.book = self
            instance.publish(data = data)
            new_contributors[key] = instance
        self.contributors = new_contributors
        return self

    def _publish_chapters(self, data: Optional['Ingredients'] = None) -> None:
        """Subclasses should provide their own method, if needed.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        if not hasattr(self, 'chapters'):
            self.chapters = {}
        if not hasattr(self, 'chapter_type'):
            self.chapter_type = Chapter
        for i, plan in enumerate(self.plans):
            pages = dict(zip(self.steps, plan))
            metadata = self._draft_chapter_metadata(number = i)
            self.add_chapter(name = str(i), pages = pages, metadata = metadata)
        return self

    """ Public Import/Export Methods """

    def load_chapter(self, file_path: str) -> None:
        """Imports a single recipe from disk and adds it to the class iterable.

        Args:
            file_path (str): a path where the file to be loaded is located.

        """
        try:
            self.chapters.update(
                {str(len(self.chapters)): self.library.load(
                    file_path = file_path,
                    file_format = 'pickle')})
        except (AttributeError, TypeError):
            self.chapters = {}
            self.chapters.update(
                {str(len(self.chapters)): self.library.load(
                    file_path = file_path,
                    file_format = 'pickle')})
        return self

    """ Composite Management Methods """

    def add_contributor(self,
            name: str,
            contributor: 'SimpleContributor') -> None:
        """Creates a SimpleContributor instance and stores it in 'contributors'.

        Args:
            name (str): name of key to access SimpleContributor instance from
                'contributors' dict.
            contributor ([type]): a SimpleContributor class (not instance).

        """

        try:
            self.contributors[name] = contributor
        except (AttributeError, TypeError):
            self.contributors = {}
            self.contributors[name] = contributor
        return self

    def remove_contributor(self, name: str) -> None:
        """Deletes a SimpleContributor from 'contributors'.

        Args:
            name (str): key name for SimpleContributor to remove from the
                'contributors' dict.

        """
        try:
            del self.contributors[name]
        except KeyError:
            pass
        return self

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        if not hasattr(self, 'parent_type'):
            self.parent_type = 'project'
        if not hasattr(self, 'children_type'):
            self.children_type = 'chapters'
        # 'options' should be created before this loop.
        for method in (
                'options',
                'steps',
                'contributors',
                'plans',
                'chapters'):
            getattr(self, '_'.join(['_draft', method]))()
        return self

    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """Finalizes 'contributors' and 'chapters'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        for method in ('contributors', 'chapters'):
            try:
                getattr(self, '_'.join(['_publish', method]))(data = data)
            except AttributeError:
                pass
        return self

    def apply(self, data: 'Ingredients', **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        new_chapters = {}
        for number, chapter in self.chapters.items():
            chapter.apply(data = ingredients, **kwargs)
            new_chapter[number] = self._extra_processing(chapter = chapter)
        self.chapters = new_chapters
        return self

    """ Properties """

    @property
    def project(self) -> None:
        return self.parent

    @project.setter
    def project(self, project: 'SimpleManuscript') -> None:
        self.parent = project
        return self

    @property
    def chapters(self) -> None:
        return self.children

    @chapters.setter
    def chapters(self, chapters: Dict[str, 'SimpleManuscript']) -> None:
        self.children = chapters
        return self

@dataclass
class Chapter(SimpleManuscript):
    """Iterator for a siMpLify process.

    Args:
        pages (Dict[str, str]): information needed to create Page classes.
            Keys are step names and values are Algorithm keys.
        metadata (Optional[Dict[str, Any]], optional): any metadata about
            the chapter. In projects, 'number' is automatically a key
            created for 'metadata' to allow for better recordkeeping.
            Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    pages: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = 'chapter'
    file_format: str = 'pickle'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _get_page(self,
            key: str,
            technique: str,
            ingredients: 'Ingredients') -> 'Page':
        return self.book.contributors[key].publish(
            page = technique,
            data = ingredients)

    """ Import/Export Methods """

    def load(self,
            file_path: Optional[str] = None,
            folder: Optional[str] = None,
            file_name: Optional[str] = None) -> None:
        """Loads 'pages' from disk.

        For any arguments not passed, default values stored in the shared
        Library instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            file_path (Optional[str]): a complete file path for the file to be
                loaded.
            folder (Optional[str]): a path to the folder where the file should
                be loaded from (not used if file_path is passed).
            file_name (Optional[str]): contains the name of the file to be
                loaded without the file extension (not used if 'file_path' is
                passed).

        """
        self.pages = self.library.load(
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    def save(self,
            file_path: Optional[str] = None,
            folder: Optional[str]  = None,
            file_name: Optional[str]  = None) -> None:
        """Exports 'pages' to disk.

        For any arguments not passed, default values stored in the shared
        Library instance will be used based upon the current 'stage' of the
        siMpLify project.

        Args:
            file_path (Optional[str]): a complete file path for the file to be
                saved.
            folder (Optional[str]): a path to the folder where the file should
                be saved (not used if file_path is passed).
            file_name (Optional[str]): contains the name of the file to be saved
                without the file extension (not used if file_path is passed).

        """
        self.library.save(
            variable = self.pages,
            file_path = file_path,
            folder = folder,
            file_name = file_name,
            file_format = 'pickle')
        return self

    """ Composite Management Methods """

    def add_book(self, book: 'Book') -> None:
        """Sets 'book' attribute to 'book'.

        Args:
            'book' ('Book'): Book instance which contains this Chapter instance.

        """
        self.book = book
        return self

    def remove_book(self) -> None:
        """Sets 'book' to None."""
        self.book = None
        return self

    def add_page(self, pages: Dict[str, 'Page']) -> None:
        """Adds page class instances to class instance.

        Args:
            pages (Dict[str, 'Page']): dict with ordered Page instances to
                be applied to passed data.

        """
        try:
            self.pages.update(pages)
        except TypeError:
            self.pages = pages
        return self

    def remove_pages(self, pages: Union[List[str], str]) -> None:
        """Delinks 'pages' from class instance.

        Args:
            pages (List[str], str): key names for Page instances to be
                delinked from the current class instance.

        """
        for name in listify(pages):
            try:
                self.pages[name].book = None
                del self.pages[name]
            except KeyError:
                pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        return self

    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """Finalizes 'pages'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        new_pages = {}
        for key, technique in self.pages.items():
            page = self._get_page(
                key = key,
                technique = technique,
                data = ingredients)
            page.chapter = self
            page.publish(data = ingredients)
            new_pages[key] = page
        self.pages = new_pages
        return self

    def apply(self, data: 'Ingredients' = None, **kwargs) -> None:
        """Applies 'pages' to 'data'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance for 'pages'
                to be applied.
            **kwargs: any paramters to pass to Page 'apply' methods.

        """
        setattr(self, data.name, data)
        for key, page in self.pages.items():
            try:
                self.book.library.stage = key
            except KeyError:
                pass
            setattr(self, data.name, page.apply(
                data = getattr(self, data.name),
                **kwargs))
        return self

    """ Properties """

    @property
    def book(self) -> None:
        return self.parent

    @book.setter
    def book(self, book: 'SimpleManuscript') -> None:
        self.parent = book
        return self

    @property
    def pages(self) -> None:
        return self.pages

    @pages.setter
    def pages(self, pages: Dict[str, 'SimpleManuscript']) -> None:
        self.pages = pages
        return self


@dataclass
class Page(SimpleManuscript):
    """Stores, combines, and applies Algorithm and Parameters instances.

    A SimpleContributor directs the building of the requisite algorithm and
    parameters to be injected into a Page instance. When possible, these Page
    instances are made to be scikit-learn compatible using the included
    'fit', 'transform', and 'fit_transform' methods. A Page instance can also
    be applied to data using the normal siMpLify 'apply' method.

    Args:
        algorithm (Algorithm): finalized algorithm instance.
        parameters (Parameters): finalized parameters instance.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """
    algorithm: 'Algorithm'
    parameters: Optional['Parameters'] = None
    name: str = 'page'
    file_format: str = 'pickle'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Dunder Methods """

    # def __repr__(self) -> Union[object, None]:
    #     """Returns 'algorithm'.

    #     Returns:
    #         'algorithm' (object): finalized algorithm.

    #     """
    #     return self.__str__()

    # def __str__(self) -> Union[object, None]:
    #     """Returns 'algorithm'.

    #     Returns:
    #         'algorithm' (object): finalized algorithm.

    #     """
    #     try:
    #         return self.algorithm
    #     except AttributeError:
    #         return None

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Attaches 'parameters' to the 'algorithm'.

        """
        try:
            self.algorithm = self.algorithm._gizmo(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm._gizmo(self.parameters)
            except AttributeError:
                pass
        except TypeError:
            pass
        return self

    # @numpy_shield
    def publish(self, ingredients: 'Ingredients') -> 'Ingredients':
        """[summary]

        Returns:
            [type]: [description]
        """

        # if self.hyperparameter_search:
        #     self.algorithm = self._search_hyperparameters(
        #         data = ingredients,
        #         data_to_use = data_to_use)
        try:
            self.algorithm.fit(
                getattr(ingredients, ''.join(['x_', ingredients.state])),
                getattr(ingredients, ''.join(['y_', ingredients.state])))
            setattr(
                ingredients, ''.join(['x_', ingredients.state]),
                self.algorithm.transform(getattr(
                    ingredients, ''.join(['x_', ingredients.state]))))
        except AttributeError:
            try:
                data = self.algorithm.publish(
                    data = ingredients)
            except AttributeError:
                pass
        return ingredients

    """ Scikit-Learn Compatibility Methods """

    @XxYy(truncate = True)
    # @numpy_shield
    def fit(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional['Ingredients'] = None) -> None:
        """Generic fit method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Raises:
            AttributeError if no 'fit' method exists for local 'algorithm'.

        """
        if x is not None:
            try:
                if y is None:
                    self.algorithm.process.fit(x)
                else:
                    self.algorithm.process.fit(x, y)
            except AttributeError:
                error = ' '.join([self.design.name,
                                  'algorithm has no fit method'])
                raise AttributeError(error)
        elif data is not None:
            self.algorithm.process.fit(
                getattr(data, ''.join(['x_', data.state])),
                getattr(data, ''.join(['y_', data.state])))
        else:
            error = ' '.join([self.name, 'algorithm has no fit method'])
            raise AttributeError(error)
        return self

    @XxYy(truncate = True)
    # @numpy_shield
    def fit_transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional['Ingredients'] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic fit_transform method for partial compatibility to sklearn

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            TypeError if DataFrame, ndarray, or ingredients is not passed to
                the method.

        """
        self.algorithm.process.fit(x = x, y = y, data = ingredients)
        if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
            return self.algorithm.process.transform(x = x, y = y)
        elif data is not None:
            return self.algorithm.process.transform(data = ingredients)
        else:
            error = ' '.join([self.name,
                              'algorithm has no fit_transform method'])
            raise TypeError(error)

    @XxYy(truncate = True)
    # @numpy_shield
    def transform(self,
            x: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            data: Optional['Ingredients'] = None) -> (
                Union[pd.DataFrame, 'Ingredients']):
        """Generic transform method for partial compatibility to sklearn.

        Args:
            x (Optional[Union[pd.DataFrame, np.ndarray]]): independent
                variables/features.
            y (Optional[Union[pd.Series, np.ndarray]]): dependent
                variable/label.
            data (Optional[Ingredients]): instance of Ingredients containing
                pandas data objects as attributes.

        Returns:
            transformed x or data, depending upon what is passed to the
                method.

        Raises:
            AttributeError if no 'transform' method exists for local
                'process'.

        """
        if hasattr(self.algorithm.process, 'transform'):
            if isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray):
                if y is None:
                    return self.algorithm.process.transform(x)
                else:
                    return self.algorithm.process.transform(x, y)
            elif data is not None:
                return self.algorithm.process.transform(
                    X = getattr(data, 'x_' + data.state),
                    Y = getattr(data, 'y_' + data.state))
        else:
            error = ' '.join([self.name, 'algorithm has no transform method'])
            raise AttributeError(error)

    """ Properties """

    @property
    def chapter(self) -> None:
        return self.chapter

    @chapter.setter
    def chapter(self, chapter: 'SimpleManuscript') -> None:
        self.parent = chapter
        return self
