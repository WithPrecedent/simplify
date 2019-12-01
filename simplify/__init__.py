"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from simplify.__main__ import main as project
from simplify.core import create_idea
from simplify.core.idea import Idea
from simplify.core.utilities import timer
from simplify.project import Project
from simplify.resources import create_ingredients
from simplify.resources import create_library
from simplify.resources.ingredients import Ingredients
from simplify.resources.library import Library


__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'

__all__ = ['project',
           'create_idea',
           'create_library',
           'create_ingredients',
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

    simplify.create_project(parameters)

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
    idea = create_idea(idea = idea)
    library = create_library(library = library, idea = idea)
    ingredients = create_ingredients(
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
            module (following the prefix 'create_').
        *args, **kwargs: appropriate arguments to be passed to corresponding
            factory function.

    Returns:
        object: instance of new class created by the factory.

    Raises:
        TypeError: if there is no corresponding function for creating a class
            instance designated in 'types'.

    """
    if object_type in get_supported_types():
        return locals()['create_' + object_type ](*args, **kwargs)
    else:
        error = ' '.join([object_type, ' is not a valid class type'])
        raise TypeError(error)

def create_project(
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
    if idea.configuration['general']['verbose']:
        print('Starting siMpLify Project')
    return Project(
        idea = idea,
        library = library,
        ingredients = ingredients)

def get_supported_types() -> List[dtr]:
    """Removes 'create_' from object names in locals() to create a list of
    supported class types.

    Returns:
        List[str]: class types which have a corresponding factory function.

    """
    types = []
    for factory_function in locals().keys():
        if factory_function.startswith('create_'):
            types.append(factory_function[len('create_'):])
    return types