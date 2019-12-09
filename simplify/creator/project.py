"""
.. module:: siMpLify project
:synopsis: entry point for implementing multiple siMpLify steps
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

import simplify
from simplify.creator.author import SimpleCodex
from simplify.creator.options import SimpleOptions
from simplify.library.utilities import listify
from simplify.creator.worker import Worker


@dataclass
class Project(object):
    """Controller class for siMpLify projects.

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
        _steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Default is None.

    """
    idea: Union['Idea', str]
    filer: Optional[Union['Filer', str]] = None
    ingredients: Optional[Union[
        'Ingredients',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True
    _steps: Optional[Union[List[str], str]] = None

    def __post_init__(self) -> None:
        """Initializes class attributes and calls appropriate methods."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Sets default 'name' attribute if none exists.
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        # Finalizes 'idea', 'filer', and 'ingredients instances.
        self.idea, self.filer, self.ingredients = simplify.startup(
            idea = self.idea,
            filer = self.filer,
            ingredients = self.ingredients)
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        return self

    """ Private Methods """

    def _draft_options(self) -> None:
        """Sets step options with information for module importation."""
        self._options = SimpleOptions(
            options = {
                'farmer': ('simplify.farmer', 'Almanac'),
                'chef': ('simplify.chef', 'Cookbook'),
                'actuary': ('simplify.actuary', 'Ledger'),
                'critic': ('simplify.critic', 'Collection'),
                'artist': ('simplify.artist', 'Canvas')},
            _author = self)
        return self

    def _draft_steps(self) -> None:
        """Creates list of steps to use in this project.

        If there are no 'steps' or there is not a list of 'steps' in 'idea', an
        empty list is created for 'steps'. Users will then have to add steps
        manually.

        """
        if self._steps is None:
            try:
                self._steps = getattr(self.idea, '_'.join([self.name, 'steps']))
            except AttributeError:
                try:
                    self._steps = getattr(
                        self.idea, '_'.join([self.name, 'steps']))
                except AttributeError:
                    self._steps = []
        else:
            self._steps = listify(self._steps)
        return self

    def _set_author(self) -> None:
        """Sets SimpleCodex instance."""
        self.author = SimpleCodex(
            idea = self.idea,
            filer = self.filer,
            ingredients = self.ingredients)
        return self

    def _set_worker(self) -> None:
        """Sets Worker instance."""
        self.worker = Worker(
            idea = self.author.idea,
            filer = self.author.filer,
            ingredients = self.author.ingredients)
        return self

    def _store_options(self) -> None:
        """Stores all used options in local attributes."""
        for step in self.steps:
            setattr(self, step, self.options[step])
        return self

    """ Core siMpLify Methods """

    def draft(self) -> None:
        """Sets initial attributes."""
        # Injects attributes from Idea instance, if values exist.
        self = self.idea.apply(instance = self)
        # Calls methods for setting 'options' and 'steps'.
        self._draft_options()
        self._draft_steps()
        return self

    def publish(self, data: Optional[object] = None) -> None:
        """Finalizes steps by creating Book instances in options.

        Args:
            data (Optional[object]): an optional object needed for a Book's
                'publish' method.

        """
        if data is None:
            data = self.ingredients
        for step in self.steps:
            step.options.publish(
                author = self.author,
                steps = self.steps,
                data = data)
        self._store_options()
        return self

    def apply(self, data: Optional[object] = None, **kwargs) -> None:
        """Applies created objects to passed 'data'.

        Args:
            data (Ingredients): data object for methods to be applied. This can
                be an Ingredients instance, but other compatible objects work
                as well.

        """
        # Sets Worker instance.
        self._set_worker()
        if data is None:
            data = self.ingredients
        for step in self.steps:
            data = self.options.apply(
                worker = self.worker,
                key = step,
                data = data,
                **kwargs)
        return self

    """ Composite Methods and Properties """

    def add_steps(self, steps: Union[List[str], str]) -> None:
        """Adds Book(s) to 'steps' property.

        Args:
            steps (Union[List[str], str]): string(s) matching key(s) in
                'options' connected to Books to add.

        """
        self._steps[key].extend(listify(steps))
        return self

    # def load_book(self, file_path: str) -> None:
    #     """Imports a single Book from disk and adds it to the class iterable.

    #     Args:
    #         file_path (str): a path where the file to be loaded is located.

    #     """
    #     self._steps.append(
    #         self.author.filer.load(
    #             file_path = file_path,
    #             file_format = 'pickle'))
    #     return self

    @property
    def steps(self) -> List[str]:
        """Returns '_steps' attribute.

        Returns:
            List of Books.

        """
        return self._steps

    @steps.setter
    def steps(self, steps: Union[List[str], str]) -> None:
        """Assigns 'steps' to '_steps' attribute.

        Args:
            steps (Union[List[str], str]): steps to to set in the '_steps'
                attribute.

        """
        self._steps = listify(steps)
        return self

    @steps.deleter
    def steps(self, steps: Union[List[str], str]) -> None:
        """ Removes 'steps' for '_steps' attribute.
        Args:
            steps (Union[List[str], str]): key(s) to steps classes to remove
                from '_steps'.
        """
        for step in listify(steps):
            try:
                del self._steps[step]
            except KeyError:
                pass
        return self