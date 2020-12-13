"""
.. module:: canvas
:synopsis: data visualizations
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.chapter import Chapter
from simplify.core.library import Book


@dataclasses.dataclass
class Canvas(Book):
    """Builds tools for data visualization.

    Args:
        idea (Union[Idea, str]): an instance of Idea or a string containing the
            file path or file name (in the current working directory) where a
            file of a supoorted file type with settings for an Idea instance is
            located.
        clerk (Optional[Union['Clerk', str]]): an instance of
            clerk or a string containing the full path of where the root
            folder should be located for file output. A clerk instance
            contains all file path and import/export methods for use throughout
            the siMpLify package. Defaults to None.
        dataset (Optional[Union['Dataset', pd.DataFrame, pd.Series,
            np.ndarray, str]]): an instance of Dataset, a string containing
            the full file path where a data file for a pandas DataFrame or
            Series is located, a string containing a file name in the default
            data folder, as defined in the shared Clerk instance, a
            DataFrame, a Series, or numpy ndarray. If a DataFrame, ndarray, or
            string is provided, the resultant DataFrame is stored at the 'df'
            attribute in a new Dataset instance. Defaults to None.
        steps (Optional[Union[List[str], str]]): ordered names of Book
            subclasses to include. These names should match keys in the
            'options' attribute. If using the Idea instance settings, this
            argument should not be passed. Defaults to None.
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared 'Idea' instance, 'name' should match the appropriate
            section name in 'Idea'. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.
        auto_publish (Optional[bool]): whether to call the 'publish' method when
            a subclass is instanced. For auto_publish to have an effect,
            'dataset' must also be passed. Defaults to True.

    """
    idea: Union['Idea', str]
    clerk: Optional[Union['Clerk', str]] = None
    dataset: Optional[Union[
        'Dataset',
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        str]] = None
    steps: Optional[Union[List[str], str]] = None
    name: Optional[str] = 'simplify'
    auto_publish: Optional[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    """ Private Methods """

    def _draft_options(self) -> None:
        self._options = SimpleRepository(contents = {
            'styler': ('simplify.artist.steps.styler', 'Styler'),
            'painter': ('simplify.artist.steps.paint', 'Painter'),
            'animator': ('simplify.artist.steps.animator', 'Animator')}
        return self

    def _draft_styler(self) -> None:
        if 'styler' not in self.steps:
            self.steps = ['styler'] + self.steps
        return self

    """ Core siMpLify Methods """


    def draft(self) -> None:
        """Creates initial attributes."""
        self.parent_type = 'project'
        self.children_type = 'chapters'
        # 'options' should be created before this loop.
        for method in (
                'options',
                'steps',
                'styler',
                'contributors',
                'plans',
                'chapters'):
            getattr(self, '_'.join(['_draft', method]))()
        return self


@dataclasses.dataclass
class Illustration(Chapter):

    def __post_init__(self) -> None:
        super().__post_init__()
        return self