"""
.. module:: book
:synopsis: composite tree base classes
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from itertools import product
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

import simplify
from simplify.core.chapter import Chapter
from simplify.core.filer import SimpleFile
from simplify.core.options import SimpleOptions
from simplify.core.state import SimpleState
from simplify.core.utilities import listify


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
        techniques (Optional[Union[List[str], str]]): ordered names of Book
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
    techniques: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True
    file_format: Optional[str] = 'pickle'
    export_folder: Optional[str] = 'book'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_plans(self) -> None:
        """Creates cartesian product of all possible 'chapters'."""
        plans = []
        for step in self.techniques:
            try:
                key = '_'.join([step, 'techniques'])
                plans.append(listify(self.options.idea[self.name][key]))
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
            pages = dict(zip(self.techniques, plan))
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
            data: Optional[object] = None) -> None:
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

    def _publish_chapters(self, data: Optional[object] = None) -> None:
        """Subclasses should provide their own method, if needed.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        if not hasattr(self, 'chapters'):
            self.chapters = {}
        if not hasattr(self, 'chapter_type'):
            self.chapter_type = Chapter
        for i, plan in enumerate(self.plans):
            pages = dict(zip(self.techniques, plan))
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

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        super().draft()
        # Injects attributes from Idea instance, if values exist.
        self = self.options.idea.apply(instance = self)
        # Drafts plans based upon settings.
        self._draft_plans()
        self._draft_chapters()
        return self

    def publish(self, data: Optional[object] = None) -> None:
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

    def apply(self, data: object, **kwargs) -> None:
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

    """ Composite Properties """
    
    @property
    def parent(self) -> NotImplementedError:
        return NotImplementedError(
            'Book instances and subclasses cannot have parents')


class BookFiler(SimpleFiler):

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
        for stage in listify(self.options.idea['simplify']['simplify_steps']):
            if stage == 'farmer':
                for step in self.options.idea['farmer']['farmer_steps']:
                    states.append(step)
            else:
                states.append(stage)
        return states

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Initializes state machine."""
        # Sets list of possible states based upon Idea instance options.
        self._options = SimpleOptions(options = self._set_states()
        # Sets initial state.
        self.state = self.options[0]
        return self