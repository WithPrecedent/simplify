"""
.. module:: creator
:synopsis: the builder design pattern made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify import startup
from simplify import get_supported_types
from simplify.creator.book import Book
from simplify.creator.chapter import Chapter
from simplify.creator.content import Content
from simplify.creator.page import Page


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = _get_supported_types()

"""
This module uses a traditional builder design pattern, with this module acting
as the defacto builder class, to create algorithms for use in the siMpLify
package. The director is any outside class or function which calls the builder
to create an algorithm and corresponding parameters.
Placing the builder functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the builder just
needs:

    import simplify.creator
    
And then to create any new algorithm, the function calls are very
straightforward and clear. For example, to create a new algorithm and
corresponding parameters, this is all that is required:

    design = creator.make(**parameters)
    
Putting the builder in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.
Using the __init__.py file for builder functions was inspired by a blog post,
which used the file for a builder design pattern, by Luna at:
https://www.bnmetrics.com/blog/builder-pattern-in-python3-simple-version
"""


def make_book(
        idea: Union[Dict[str, Dict[str, Any]], str, 'Idea'],
        library: Optional[Union['Library', str]],
        ingredients: Optional[Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]] = None,
        options: Optional['SimpleOptions', Dict[str, tuple[str, str]]] = None,
        steps: Optional[Union[List[str], str]] = None,
        name: Optional[str] = None,
        auto_publish: Optional[bool] = True) -> 'Book':
    """
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
        options (Optional['SimpleOptions', Dict[str, tuple[str, str]]]): either
            a SimpleOptions instance or a dictionary compatible with a
            SimpleOptions instance. Defaults to None.
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
            'ingredients' and 'options' must also be passed. Defaults to True,
            but the 'publish' method will not be called without 'ingredients'
            and 'options'.


    """
    idea, library, ingredients = startup(
        idea = idea,
        library = library,
        ingredients = ingredients)
    return Book(
        idea = idea,
        library = library,
        ingredients = ingredients,
        options = options,
        steps = steps,
        name = name,
        auto_publish = auto_publish)
    
def make_chapter(self):
    