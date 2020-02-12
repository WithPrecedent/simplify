"""
.. module:: analyst
:synopsis: machine learning made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from dataclasses import field
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Optional,
    Tuple, Union)

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from simplify.core.book import Book
from simplify.core.utilities import listify
from simplify.core.utilities import subsetify
from simplify.core.worker import Worker


@dataclass
class Cookbook(Book):
    """Standard class for iterable storage in the Analyst subpackage.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. Defaults to
            'cookbook'
        iterable(Optional[str]): name of attribute for storing the main class
            instance iterable (called by __iter___). Defaults to 'recipes'.
        chapters (Optional['Plan']): iterable collection of steps and
            techniques to apply at each step. Defaults to an empty 'Plan'
            instance.

    """
    name: Optional[str] = 'cookbook'
    iterable: Optional[str] = field(default_factory = lambda: 'recipes')
    chapters: Optional[List['Chapter']] = field(default_factory = list)


@dataclass
class Analyst(Worker):
    """Applies a 'Cookbook' instance to data.

    Args:
        idea ('Idea'): an 'Idea' instance with project settings.

    """
    idea: 'Idea'

    """ Private Methods """

    def _iterate_chapter(self,
            book: 'Book',
            chapter: 'Chapter',
            data: Union['Dataset']) -> 'Chapter':
        """Iterates a single chapter and applies 'techniques' to 'data'.

        Args:
            chapter ('Chapter'): instance with 'techniques' to apply to 'data'.
            data (Union['Dataset', 'Book']): object for 'chapter'
                'techniques' to be applied.

        Return:
            'Chapter': with any changes made. Modified 'data' is added to the
                'Chapter' instance with the attribute name matching the 'name'
                attribute of 'data'.

        """
        data.create_xy()
        remaining = list(chapter.techniques.keys())
        for step, techniques in chapter.techniques.items():
            if not techniques in ['none', ['none']]:
                for technique in listify(techniques, default_empty = True):
                    technique = self._finalize_technique(
                        book = book,
                        technique = technique,
                        data = data)
                    if step in ['split']:
                        chapter.techniques = subsetify(
                            chapter.techniques,
                            remaining)
                        data = self._split_loop(
                            book = book,
                            data = data,
                            chapter = chapter)
                    elif step in ['search']:
                        chapter.techniques = subsetify(
                            chapter.techniques,
                            remaining)
                        chapter = self._search_loop(
                            data = data,
                            chapter = chapter)
                    else:
                        data = technique.apply(data = data)
            remaining.remove(step)
        setattr(chapter, 'data', data)
        return chapter

def _finalize_chapters(self, book: 'Book', data: 'Dataset') -> 'Book':
    for chapter in book.chapters:
        for step, techniques in chapter.techniques.items():
            for technique in listify(techniques):
                if technique is not 'none':
                    technique = self._add_conditionals(
                        book = book,
                        technique = technique,
                        data = data)
                    technique = self._add_data_dependents(
                        technique = technique,
                        data = data)
                    technique = self._add_parameters_to_algorithm(
                        technique = technique)
    return book