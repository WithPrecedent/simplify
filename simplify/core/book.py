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

import simplify.core
from simplify.core.chapters import Chapters
from simplify.core.codex import SimpleCodex
from simplify.core.options import CodexOptions
from simplify.core.utilities import listify


@dataclass
class Book(SimpleCodex):
    """Builds and controls Chapters.

    This class contains methods useful to create iterators and iterate over
    passed arguments based upon user-selected options. Book subclasses construct
    iterators and process data with those iterators.

    Args:
        ingredients (Optional[Union['Ingredients', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Ingredients, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Inventory instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Ingredients instance. Default is None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        steps (Optional[List[str], str, Dict[str, str]]): ordered list of
            steps to use with each item matching a key in 'options', a single
            item which matchings a key in 'options' or a dictionary where each
            key matches a key in options and each value is a 'technique'
            parameter to be sent to a child class. Defaults to an empty dict.
        options (Optional[Union['CodexOptions', Dict[str, Any]]]): allows
            setting of 'options' property with an argument. Defaults to an
            empty dict.
        chapters (Optional[Union['Chapters', Dict[str, 'Page']]]): a dictionary
            with stored Page instance or a completed Chapters instance. This
            should not be ordinarly passed, but is made available if users
            wish to pass a set of Chapters instance. Defaults to an empty dict.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'ingredients' must also be passed. Defaults to True.
        file_format (Optional[str]): name of file format for object to be
            serialized. Defaults to 'pickle'.
        export_folder (Optional[str]): attribute name of folder in 'inventory' for
            serialization of subclasses to be saved. Defaults to 'book'.

    """
    chapters: List['Chapters']
    filer: 'SimpleFiler'
    name: Optional[str] = 'simplify'

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        self.proxies = {'children': 'chapters', 'child': 'chapter'}
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_plans(self) -> None:
        """Creates cartesian product of all possible 'chapters'."""
        plans = []
        for step in self.steps:
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

    def _publish_plans(self) -> None:
        """Subclasses should provide their own method, if needed.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.

        """
        if not hasattr(self, 'chapter_type'):
            self.chapter_type = Chapter
        for i, plan in enumerate(self.plans):
            self.chapters.add(
                chapter = self.chapter_type(
                    name = str(i),
                    steps = dict(zip(self.steps, plan)),
                    options = self.options,
                    metadata = self._publish_chapter_metadata(number = i)))
        return self

    def _publish_chapters(self) -> None:
        for chapter in self.children:
            chapter.publish()
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
        self.chapters = Chapters(options = self.options, book = self)
        # Drafts plans based upon settings.
        self._draft_plans()
        return self

    def publish(self) -> None:
        """Finalizes 'authors' and 'chapters'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance.
                'ingredients' needs to be passed if there are any
                'data_dependent' parameters for the included Page instances
                in 'pages'. Otherwise, it need not be passed. Defaults to None.

        """
        if data is None:
            try:
                data = self.ingredients
            except AttributeError:
                pass
        self._publish_plans()
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