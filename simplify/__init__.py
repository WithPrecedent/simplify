"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from simplify.__main__ import main as project
from simplify.core import make_idea
from simplify.core.idea import Idea
from simplify.core.utilities import timer
from simplify.project import Project
from simplify.resources import make_ingredients
from simplify.resources import make_library
from simplify.resources.ingredients import Ingredients
from simplify.resources.library import Library


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['project',
           'make_idea',
           'make_library',
           'make_ingredients',
           'Idea',
           'Library',
           'Ingredients',
           'Project',
           'timer']


"""
This module uses a traditional factory design pattern, with this module acting
as the defacto class, to create some of the primary class objects used in the
siMpLify packages. Similar modules exist in __init__.py files throughout the
siMpLify packages.

Placing the factory functions here makes for more elegant code. To use these
functions, the importation line at the top of module using the factory just
needs:

    import simplify

And then to create any new classes, the function calls are very straightforward
and clear. For example, to create a new Project, this is all that is required:

    simplify.make_project(parameters)

Putting the factory in __init__.py takes advantage of the automatic importation
of the file when the folder is imported. As a result, the specific functions
do not need to be imported by name and/or located in another module that needs
to be imported.

Using the __init__.py file for factory functions was inspired by a blog post by
Luna at:
https://www.bnmetrics.com/blog/factory-pattern-in-python3-simple-version

"""

def startup(
        idea: Union[Dict[str, Dict[str, Any]], str, 'Idea'],
        library: Union[str, 'Library'],
        ingredients: Union[
            'Ingredients',
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            str]) -> None:
    """Creates Idea, Library, and Ingredients instances.

    Args:
        idea: Union[Dict[str, Dict[str, Any]], str, 'Idea']: Idea instance or
            needed information to create one.
        library: Union[str, 'Library']: Library instance or root folder for one.
        ingredients: Union['Ingredients', pd.DataFrame, pd.Series, np.ndarray,
            str]: Ingredients instance or information needed to create one.

    Returns:
        Idea, Library, Ingredients instances, published.

    """
    idea = make_idea(idea = idea)
    library = make_library(library = library, idea = idea)
    ingredients = make_ingredients(
        ingredients = ingredients,
        idea = idea,
        library = library)
    return idea, library, ingredients

def create(object_type: str, *args, **kwargs) -> object:
    """Calls appropriate function based upon 'types' passed.

    This method adds nothing to calling the functions directly. It is simply
    included for easier external iteration and/or for those who prefer a generic
    function for all factories.

    Args:
        object_type (str): name of package type to be created. It should
            correspond to the suffix of one of the other functions in this
            module (following the prefix 'make_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            factory function.

    Returns:
        object: instance of new class created by the factory.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance designated in 'types'.

    """
    if object_type in get_supported_types():
        return locals()['make_' + object_type ](*args, **kwargs)
    else:
        error = ' '.join([object_type, ' is not a valid class type'])
        raise TypeError(error)

def make_project(
        idea: 'Idea',
        library: 'Library',
        ingredients: 'Ingredients') -> 'Project':
    """Creates Project from 'idea', 'library', and 'ingredients'.

    Args:
        idea ('Idea'): an Idea instance.
        library ('Library'): a Library instance.
        ingredients ('Ingredients'): an Ingredients instance.

    Returns:
        Project based upon passed attributes.

    """
    return Project(
        idea = idea,
        library = library,
        ingredients = ingredients)

def make_project(
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
    return Project(
        idea = idea,
        library = library,
        ingredients = ingredients,
        options = options,
        steps = steps,
        name = name,
        auto_publish = auto_publish)
    
def get_supported_types() -> List[dtr]:
    """Removes 'make_' from object names in locals() to create a list of
    supported class types.

    Returns:
        List[str]: class types which have a corresponding factory function.

    """
    types = []
    for factory_function in locals().keys():
        if factory_function.startswith('make_'):
            types.append(factory_function[len('make_'):])
    return types