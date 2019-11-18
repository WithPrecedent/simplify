"""
.. module:: siMpLify project
:synopsis: entry point for implementing multiple siMpLify subpackages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from importlib import import_module
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simplify.core.book import Book
from simplify.core.utilities import listify


@dataclass
class Project(Book):
    """Controller class for siMpLify projects.

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

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_books(self) -> None:
        """Creates 'books' from 'steps' and 'options'."""
        self.books = {}
        for step in self.steps:
            try:
                book = getattr(
                    import_module(self.options[step][0]),
                    self.options[step][1])
                if self.verbose:
                    print('Drafting', self.options[step][1].lower())
                instance = book(idea = self.idea, library = self.library)
                instance.project = self
                self.books[step] = instance
            except KeyError:
                error = ' '.join(
                    [step, 'does not match an option in', self.name])
                raise KeyError(error)
        return self

    def _publish_books(self, data: Optional['Ingredients'] = None) -> None:
        """Finalizes 'books'."""
        new_books = {}
        for key, book in self.books.items():
            try:
                if self.verbose:
                    print('Publishing', book.__class__.__name__.lower())
                book.publish(data = data)
                new_books[key] = book
            except KeyError:
                error = ' '.join([key, 'does not match a Book in', self.name])
                raise KeyError(error)
        return self

    """ Composite Management Methods """

    def add_book(self, name: str, book: 'Book') -> None:
        """Creates a Book instance and stores it in 'books'.

        Args:
            name (str): name of key to access Book instance from 'books' dict.
            book ([type]): a Book class (not instance).

        """

        try:
            self.books[name] = book
        except (AttributeError, TypeError):
            self.books = {}
            self.books[name] = book
        return self

    def remove_book(self, name: str) -> None:
        """Deletes a Book from 'books'.

        Args:
            name (str): key name for Book to remove from the 'books' dict.

        """
        try:
            del self.books[name]
        except KeyError:
            pass
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Creates initial attributes."""
        # Sets step options with information for module importation.
        self.options = {
            'farmer': ('simplify.farmer', 'Almanac'),
            'chef': ('simplify.chef', 'Cookbook'),
            'actuary': ('simplify.actuary', 'Ledger'),
            'critic': ('simplify.critic', 'Collection'),
            'artist': ('simplify.artist', 'Canvas')}
        # Finalizes core attributes.
        for method in ('attributes', 'steps', 'books'):
            getattr(self, '_'.join(['_draft', method]))()
        return self

    def publish(self, data: Optional['Ingredients'] = None) -> None:
        """"Finalizes 'books'.

        Args:
            data (Optional['Ingredients']): an Ingredients instance. 'data'
                needs to be passed if there are any 'data_dependent' parameters
                for the included Page instances in 'pages'. Otherwise, it need
                not be passed. Defaults to None.

        """
        if data is None:
            data = self.ingredients
        self._publish_books(data = data)
        return self

    def apply(self, data: 'Ingredients', **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        for step in self.steps:
            getattr(self, step).apply(data = self.ingredients)
        return self