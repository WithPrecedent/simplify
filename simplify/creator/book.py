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

import simplify.creator
from simplify.creator.chapter import Chapter
from simplify.library.filer import SimpleFile
from simplify.creator.options import Options
from simplify.library.state import SimpleState
from simplify.library.utilities import listify


@dataclass
class Book(SimpleCodex):
    """Builds and controls Chapters.

    This class contains methods useful to create iterators and iterate over
    passed arguments based upon user-selected options. Book subclasses construct
    iterators and process data with those iterators.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        filer (Optional[Union['Filer', str]]): an instance of
            filer or a string containing the full path of where the root
            folder should be located for file output. A filer instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Default is None.
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Filer instance, a
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
        export_folder (Optional[str]): attribute name of folder in 'filer' for
            serialization of subclasses to be saved. Defaults to 'book'.

    """
    idea: Union['Idea', str]
    filer: Optional[Union['Filer', str]] = None
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
        self.proxies = {'children': 'chapters', 'child': 'chapter'}
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

    def _publish_chapter_metadata(self, number: int) -> Dict[str, Any]:
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

    def _publish_chapters(self, data: Optional[object] = None) -> None:
        """Subclasses should provide their own method, if needed.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        if not hasattr(self, 'chapter_type'):
            self.chapter_type = Chapter
        for i, plan in enumerate(self.plans):
            self.add_chapters(
                chapters = self.chapter_type(
                    name = str(i),
                    techniques = dict(zip(self.techniques, plan)),
                    metadata = self._publish_chapter_metadata(number = i)))
        return self

    def _apply_extra_processing(self, chapter: 'Chapter') -> 'Chapter':
        """Extra actions to take for each Chapter applied.

        Subclasses should provide '_apply_extra_processing' methods, if needed.

        Returns:
            'Chapter' with any modifications made.

        """
        return chapter

    """ Core siMpLify methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        super().draft()
        # Drafts plans based upon settings.
        self._draft_plans()
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Finalizes 'authors' and 'chapters'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        super().publish(data = data)
        self._publish_chapters(data = data)
        return self

    def apply(self, data: Optional[object], **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        if data is not None:
            self.ingredients = data
        new_chapters = []
        for chapter in self.chapters:
            chapter.apply(data = data, **kwargs)
            new_chapters.append(self._apply_extra_processing(chapter = chapter))
        self.chapters = new_chapters
        return self