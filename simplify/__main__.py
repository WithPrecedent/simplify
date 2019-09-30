"""
.. module:: simplify
 :synopsis: data science made simple
 :author: Corey Rayburn Yung
 :copyright: 2019
 :license: CC-BY-NC-4.0
"""

from dataclasses import dataclass
import os
import sys

from simplify import Idea
from simplify import Ingredients
from simplify.core.base import SimpleClass
from simplify.core.decorators import localize


@dataclass
class Simplify(SimpleClass):
    """Controller class for completely automated projects.

    This class is provided for applications that rely exclusively on Idea
    settings and/or subclass attributes. For a more customized application,
    users can access the subpackages ('farmer', 'chef', 'critic', and 'artist')
    directly.

    Args:
        idea(Idea or str): either an instance of Idea or a string containing
            file path of where a settings file for an Idea instance is located.

        ingredients(Ingredients or str): an instance of Ingredients or a string
            containing the file path of where a data file for a pandas
            DataFrame is located.
        depot(Depot): an instance of Depot which contains information about
            file and folder locations as well as methods for loading and saving
            files.
        name(str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify package. This is
            used instead of __class__.__name__ so that subclasses can maintain
            the same string name without altering the formal class name.
        auto_finalize(bool): sets whether to automatically call the 'finalize'
            method when the class is instanced. If you do not plan to make any
            adjustments beyond the Idea configuration, this option should be
            set to True. If you plan to make such changes, 'finalize' should be
            called when those changes are complete.
        auto_produce: sets whether to automatically call the 'produce' method
            when the class is instanced.

    """

    idea: object
    ingredients: object = None
    depot: object = None
    name: str = 'simplify'
    auto_finalize: bool = True
    auto_produce: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    def __call__(self, **kwargs):
        """Calls the class as a function.

        Only keyword arguments are accepted so that they can be properly
        turned into local attributes. Those attributes are then used by the
        various 'produce' methods.

        Args:
            **kwargs(list(Recipe) and/or Ingredients): variables that will
                be turned into localized attributes.
        """
        self.finalize()
        self.produce(**kwargs)
        return self

    """ Private Methods """

    def _produce_artist(self):
        self.options['critic'].produce(
                ingredients = self.ingredients,
                recipes = self.recipes)
        return self

    def _produce_chef(self):
        self.ingredients, self.recipes = self.options['chef'].produce(
                ingredients = self.ingredients)
        return self

    def _produce_critic(self):
        self.ingredients = self.options['critic'].produce(
                ingredients = self.ingredients,
                recipes = self.recipes)
        return self

    def _produce_farmer(self):
        self.ingredients = self.options['farmer'].produce(
                ingredients = self.ingredients)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        self.hierarchy = ['packages', 'plans', 'steps', 'techniques',
                          'algorithms']
        self.options = {
                'farmer': ['simplify.farmer', 'Almanac'],
                'chef': ['simplify.chef', 'Cookbook'],
                'critic': ['simplify.critic', 'Review'],
                'artist': ['simplify.artist', 'Canvas']}
        self.checks = ['depot', 'ingredients']
        return self

    def finalize(self):
        for subpackage in self.subpackages:
            self.options[subpackage].finalize()
        return self

    @localize
    def produce(self, **kwargs):
        for subpackage in self.subpackages:
            getattr(self, '_produce_' + subpackage)()
        return self

def _args_to_dict():
    """Converts command line arguments into 'arguments' dict.

    The dictionary conversion is more forgiving than the typical argparse
    construction. It allows the package to check default options and give
    clearer error coding.

    This handy bit of code, as an alternative to argparse, was found here:
        https://stackoverflow.com/questions/54084892/
        how-to-convert-commandline-key-value-args-to-dictionary

    Returns:
        arguments(dict): dictionary of command line options when the options
            are separated by '='.
    """
    arguments = {}
    for argument in sys.argv[1:]:
        if '=' in argument:
            separated = argument.find('=')
            key, value = argument[:separated], argument[separated + 1:]
            arguments[key] = value
    return arguments


def _get_file(file_reference, default = None):
    """Returns file path based upon arguments passed.

    Args:
        file_reference(str): either full path of file sought or file name of
            file in the current working directory.
        default(str): fallback file name in current working directory if
            file_reference is not found.
    Returns:
        path(str) based upon the 'file_reference' or 'default' if the file
            exists.

    """
    if os.path.isfile(file_reference):
        return file_reference
    elif os.path.isfile(os.path.join(os.getcwd(), file_reference)):
        return os.path.join(os.getcwd(), file_reference)
    elif default:
        return os.path.join(os.getcwd(), default)
    else:
        error = file_reference + 'not found'
        raise FileNotFoundError(error)


def _get_idea(arguments):
    """Creates Idea instance from command line or default options.

        Args:
            arguments(dict): command line options dictionary.

        Returns:
            idea(Idea): instance of Idea with settings loaded from a file.

        Raises:
            FileNotFoundError if passed string for '-idea' option is not found
                or the default file 'settings.ini' is not found in the current
                working folder.
    """
    if 'idea' in arguments:
        return Idea(configuration = _get_file(arguments['-idea'],
                                              'settings.ini'))


def _get_ingredients(arguments):
    """Creates Ingredients instance with or without command line options.

        Args:
            arguments(dict): command line options dictionary.

        Returns:
            ingredients(Ingredients): instance of Ingredients with loaded
                pandas DataFrame as 'df' attribute if '-ingredients' option
                passed and the file was found.

        Raises:
            FileNotFoundError if passed string for '-ingredients' option is not
                found.
    """
    if '-ingredients' in arguments:
        if os.path.isfile(arguments['-ingredients']):
            ingredients = Ingredients(df = arguments['-ingredients'])
        else:
            error = 'ingredients file not found:' + arguments['-ingredients']
            raise FileNotFoundError(error)
    else:
        ingredients = Ingredients()
    return ingredients


if __name__ == '__main__':
    print('Starting siMpLify')
    arguments = _args_to_dict()
    idea = _get_idea(arguments = arguments)
    ingredients = _get_ingredients(arguments = arguments)
    Simplify(idea = idea, ingredients = ingredients)