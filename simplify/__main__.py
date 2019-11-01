"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import os
import sys

from simplify.core.idea import Idea
from simplify.core.depot import Depot
from simplify.core.ingredients import Ingredients
from simplify.project import Project


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

def _get_depot(arguments):
    """Creates Depot instance from command line or default options.

        Args:
            arguments(dict): command line options dictionary.

        Returns:
            depot(Depot): instance of Depot with root folder set to the argument
                passed or default option

    """
    try:
        return Depot(root_folder = arguments['-depot'])
    except KeyError:
        return Depot(root_folder = os.path.join('..', '..'))

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
    try:
        return Idea(options = arguments['-idea'])
    except KeyError:
        return Idea(options = os.path.join(os.getcwd, 'settings.ini'))

def _get_ingredients(arguments):
    """Creates Ingredients instance with or without command line options.

        Args:
            arguments(dict): command line options dictionary.

        Returns:
            ingredients(Ingredients): instance of Ingredients with loaded
                pandas DataFrame as 'df' attribute if '-ingredients' option
                passed and the file was found. Otherwise, Ingredients is
                instanced with no DataFrame (which is the normal case for
                projects using siMpLify to gather data).

    """
    try:
        return Ingredients(df = _get_file(arguments['-ingredients']))
    except KeyError:
        return Ingredients()

def main(idea, depot, ingredients):
    print('Starting siMpLify')
    return Project(idea = idea, depot = depot, ingredients = ingredients)

if __name__ == '__main__':
    arguments = _args_to_dict()
    idea = _get_idea(arguments = arguments)
    depot = _get_depot(arguments = arguments)
    ingredients = _get_ingredients(arguments = arguments)
    main(idea = idea, depot = depot, ingredients = ingredients)