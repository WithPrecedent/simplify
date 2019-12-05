"""
.. module:: author
:synopsis: book builder
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.options import SimpleOptions
from simplify.core.utilities import listify


@dataclass
class Author(ABC):
    """Base class for creating Book instances.

    Author subclasses direct the creation of siMpLify classes in the following
    manner.

        Idea -> Options -> Outline -> Content -> Page -> Chapter -> Book

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
    name: Optional[str] = None
    auto_publish: Optional[bool] = True

    def __post_init__(self) -> None:
        """Calls initialization methods and sets class instance defaults."""
        # Automatically calls 'draft' method.
        self.draft()
        # Calls 'publish' method if 'auto_publish' is True.
        if self.auto_publish:
            self.publish()
        return self

    """ Private Methods """

    def _draft_options(self, 
            manuscript: 'SimpleManuscript') -> 'SimpleManuscript':
        """Subclasses should provide their own methods to create 'options'."""
        manuscript._options = SimpleOptions(options = {}, _manuscript = self)
        return manuscript

    def _draft_techniques(self, 
            manuscript: 'SimpleManuscript') -> 'SimpleManuscript':
        """If 'techniques' does not exist, gets 'techniques' from 'idea'.

        If there are no matching 'steps' or 'techniques' in 'idea', an empty
        list is created for 'techniques'.

        """
        manuscript.compare = False
        if manuscript.techniques is None:
            try:
                manuscript.techniques = getattr(
                    self.idea, '_'.join([manuscript.name, 'steps']))
            except AttributeError:
                try:
                    manuscript.compare = True
                    manuscript.techniques = getattr(
                        self.idea, '_'.join([manuscript.name, 'techniques']))
                except AttributeError:
                    manuscript.techniques = ['none']
        else:
            manuscript.techniques = listify(manuscript.techniques)
        return manuscript

    """ Core siMpLify Methods """

    def draft(self,
            manuscript: 'SimpleManuscript') -> 'SimpleManuscript':
        """Creates initial attributes."""
        # Injects attributes from Idea instance, if values exist.
        self = self.idea.apply(instance = self)
        # initializes core attributes.
        manuscript = self._draft_options(manuscript = manuscript)
        manuscript = self._draft_techniques(manuscript = manuscript)
        # Initializes all needed options."""
        manuscript.options.load(manuscript.techniques)
        return manuscript

    def publish(self,
            manuscript: 'SimpleManuscript',
            data: Optional[object] = None) -> 'SimpleManuscript':
        """Finalizes 'options'."""
        if data is None:
            data = self.ingredients
        for technique in manuscript.techniques:
            technique.options.publish(
                techniques = manuscript.techniques,
                data = data)
        return self
        
    def apply(self,
            manuscript: 'SimpleManuscript',
            data: Optional[object] = None, **kwargs) -> 'SimpleManuscript':
        if data is None:
            data = self.ingredients
        for technique in manuscript.techniques:
            data = manuscript.options.apply(
                key = technique,
                data = data, 
                **kwargs)
        return data