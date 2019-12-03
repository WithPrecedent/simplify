"""
.. module:: book
:synopsis: composite tree base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
from itertools import product
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.base.filer import SimpleFile
from simplify.base.options import SimpleOptions
from simplify.base.state import SimpleState
from simplify.core.utilities import listify


@dataclass
class SimpleManuscript(SimpleOptions, ABC):
    """Base class for data processing, analysis, and visualization.

    SimpleComposite implements a modified composite tree pattern for organizing
    the various subpackages in siMpLify.

    """
    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Sets default 'name' attribute if none exists.
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()
        # Calls SimpleOptions __post_init__ method.
        SimpleOptions.__post_init__()
        # Injects attributes from Idea instance, if values exist.
        try:
            self = self.idea.apply(instance = self)
        except AttributeError:
            pass
        # Creates proxy names for attributes, if 'proxies' attribute exists.
        self.proxify()
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if hasattr(self, 'auto_publish') and self.auto_publish:
            self.publish()
        return self

    """ Dunder Methods """

    def __iter__(self) -> NotImplementedError:
        raise NotImplementedError(' '.join([
            self.__class__.__name__, 'has no child classes']))

    """ Composite Management Methods """

    def proxify(self, proxies: Optional[Dict[str, str]] = None) -> None:
        """Creates proxy names for attributes and methods.

        Args:
            proxies (Optional[Dict[str, str]]): dictionary with keys of current
                attribute names and values of proxy attribute names. If
                'proxies' is not passed, the method looks for 'proxies'
                attribute in the subclass instance. Defaults to None

        """
        if proxies is None:
            try:
                proxies = self.proxies
            except AttributeError:
                pass
        if proxies is not None:
            proxy_attributes = {}
            for name, proxy in proxies.items():
                for key, value in self.__dict__.items():
                    if proxy in key:
                        proxy_attributes[key.replace(name, proxy)] = value
            self.__dict__.update(proxy_attributes)
        return self

    """ Core siMpLify Methods """

    @abstractmethod
    def draft(self) -> None:
        """Required method that sets default values.

        Subclasses should provide their own 'draft' method.

        """
        return self

    @abstractmethod
    def publish(self, data: Optional[object] = None) -> None:
        """Required method which applies methods to passed data.

        Subclasses should provide their own 'publish' method.

        Args:
            data (Optional[object]): an Ingredients instance.

        """
        return self

    @abstractmethod
    def apply(self, data: object, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Subclasses should provide their own 'apply' method, if needed.

        Args:
            data (object): data object for methods to be applied.

        """
        return self

    """ Properties """

    @property
    def book(self) -> 'Book':
        """Returns '_book' attribute."""
        return self._book

    @book.setter
    def book(self, book: 'Book') -> None:
        """Sets '_book' attribute to 'book' argument.

        Args:
            book (Book): Book class up one level in the composite tree.

        """
        self._book = book
        return self

    @book.deleter
    def book(self) -> None:
        """Sets 'book' to None."""
        self._book = None
        return self

    @property
    def chapters(self) -> Dict[str, Union['Book', 'Page']]:
        """Returns '_chapters' attribute.

        Returns:
            Dict of str access keys and Book or Page values.

        """
        return self._chapters

    @chapters.setter
    def chapters(self,
            chapters: Dict[str, Union['Book', 'Page']],
            override: Optional[bool] = False) -> None:
        """Adds 'chapters' to '_chapters' attribute.

        If 'override' is True, 'chapters' replaces '_chapters'.

        Args:
            chapters (Dict[str, Union['Book', 'Page']]): dictionary with str
                for reference keys and values of either 'Book' or 'Page'.
            override (Optional[bool]): whether to overwrite existing '_chapters'
                (True) or add 'chapters' to '_chapters' (False). Defaults to
                False.

        """
        if override or not hasattr(self, '_chapters') or not self._chapters:
            self._chapters = chapters
        else:
            self._chapters.update(chapters)
        return self

    @chapters.deleter
    def chapters(self, chapters: Union[List[str], str]) -> None:
        """ Removes 'chapters' for '_chapters' attribute.

        Args:
            chapters (Union[List[str], str]): key(s) to chapters classes to
                remove from '_chapters'.

        """
        for child in listify(chapters):
            try:
                del self._chapters[child]
            except (KeyError, AttributeError):
                pass
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
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'library' for
            serialization of subclasses to be saved. Defaults to 'book'.

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
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        self.proxies = {'chapters': 'chapters'}
        self.idea, self.library, self.ingredients = simplify.startup(
            idea = self.idea,
            library = self.library,
            ingredients = self.ingredients)
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable for '_chapters'."""
        try:
            return iter(self._chapters)
        except AttributeError:
            self._chapters = {}
            return iter(self._chapters)

    """ Private Methods """

    def _draft_steps(self) -> None:
        """If 'steps' does not exist, gets 'steps' from Idea settings.

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

    def _draft_authors(self) -> None:
        """Creates 'authors' containing SimpleDirector instances."""
        self.authors  = {}
        for step in self.steps:
            try:
                author = getattr(
                    import_module(self.options[step][0]),
                    self.options[step][1])
                self.add_author(name = step, author = author)
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
            self.add_chapters(name = str(i), pages = pages, metadata = metadata)
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

    def _publish_authors(self,
            data: Optional['Ingredients'] = None) -> None:
        """Converts author classes into class instances.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        new_authors = {}
        for key, author in self.authors.items():
            instance = author(idea = self.idea, library = self.library)
            instance.book = self
            instance.publish(data = data)
            new_authors[key] = instance
        self.authors = new_authors
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

    def add_author(self,
            name: str,
            author: 'SimpleDirector') -> None:
        """Creates a SimpleDirector instance and stores it in 'authors'.

        Args:
            name (str): name of key to access SimpleDirector instance from
                'authors' dict.
            author ([type]): a SimpleDirector class (not instance).

        """

        try:
            self.authors[name] = author
        except (AttributeError, TypeError):
            self.authors = {}
            self.authors[name] = author
        return self

    def remove_author(self, name: str) -> None:
        """Deletes a SimpleDirector from 'authors'.

        Args:
            name (str): key name for SimpleDirector to remove from the
                'authors' dict.

        """
        try:
            del self.authors[name]
        except KeyError:
            pass
        return self

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        for method in (
                'options',
                'steps',
                'authors',
                'plans',
                'chapters'):
            getattr(self, '_'.join(['_draft', method]))()
        return self

    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """Finalizes 'authors' and 'chapters'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        for method in ('authors', 'chapters'):
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
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.

    """
    pages: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = 'chapter'
    file_format: str = 'pickle'
    export_folder: str = 'chapter'

    def __post_init__(self) -> None:
        self.proxies = {'book': 'book', 'chapters': 'pages'}
        super().__post_init__()
        return self

    """ Dunder Methods """

    def __iter__(self) -> Iterable:
        """Returns iterable for 'pages'."""
        try:
            return iter(self._pages)
        except AttributeError:
            self._pages= {}
            return iter(self._pages)

    """ Private Methods """

    def _get_page(self,
            key: str,
            technique: str,
            ingredients: 'Ingredients') -> 'Page':
        return self.book.authors[key].publish(
            page = technique,
            data = ingredients)

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


@dataclass
class Page(SimpleManuscript):
    """Stores, combines, and applies Algorithm and Parameters instances.

    A SimpleDirector directs the building of the requisite algorithm and
    parameters to be injected into a Page instance. When possible, these Page
    instances are made to be scikit-learn compatible using the included
    'fit', 'transform', and 'fit_transform' methods. A Page instance can also
    be applied to data using the normal siMpLify 'apply' method.

    Args:
        components (Dict[str, object])
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.

    """
    components: Dict[str, object]
    name: str = 'page'
    file_format: str = 'pickle'
    export_folder: str = 'chapter'

    def __post_init__(self) -> None:
        self.proxies = {'book': 'chapter'}
        super().__post_init__()
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Attaches 'parameters' to the 'algorithm'.

        """
        try:
            self.algorithm = self.algorithm.process(**self.parameters)
        except AttributeError:
            try:
                self.algorithm = self.algorithm.process(self.parameters)
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


class ObjectFiler(SimpleFile):
    folder_path: str
    file_name: str
    file_format: 'FileFormat'

    def __post_init__(self):
        return self

@dataclass
class Stage(SimpleState):
    """State machine for siMpLify project workflow.

    Args:
        idea (Idea): an instance of Idea.
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
    name: Optional[str] = 'stage_machine'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _set_states(self) -> List[str]:
        """Determines list of possible stages from 'idea'.

        Returns:
            List[str]: states possible based upon user selections.

        """
        states = []
        for stage in listify(self.idea['simplify']['simplify_steps']):
            if stage == 'farmer':
                for step in self.idea['farmer']['farmer_steps']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Initializes state machine."""
        # Sets list of possible states based upon Idea instance options.
        self.options = self._set_states()
        # Sets initial state.
        self.state = self.options[0]
        return self