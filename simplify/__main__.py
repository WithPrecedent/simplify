"""
.. module:: siMpLify
:synopsis: data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import os
import sys

from simplify import Idea, Depot, Ingredients
from simplify.core.base import SimpleClass, Simplify


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

def _get_file(file_reference):
    """Returns file path based upon arguments passed.

    Args:
        file_reference(str): either full path of file sought or file name of
            file in the current working directory.

    Returns:
        path(str) based upon the 'file_reference' if the file exists.

    Raises:
        FileNotFoundError if 'file_reference' does not match a file on disc.

    """
    if os.path.isfile(file_reference):
        return file_reference
    elif os.path.isfile(os.path.join(os.getcwd(), file_reference)):
        return os.path.join(os.getcwd(), file_reference)
    else:
        error = file_reference + 'not found'
        raise FileNotFoundError(error)

def _get_depot(arguments):
    """Creates Depot instance from command line or default options.

        Args:
            arguments(dict): command line options dictionary.

        Returns:
            depot(Depot): instance of Depot with root folder set to the argument
                passed or default option

    """
    if 'depot' in arguments:
        return Depot(root_folder = arguments['-depot'])
    else:
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
    if 'idea' in arguments:
        return Idea(configuration = _get_file(arguments['-idea']))
    else:
        return Idea(configuration = os.path.join(os.getcwd, 'settings.ini'))

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
    if '-ingredients' in arguments:
        return Ingredients(df = _get_file(arguments['-ingredients']))
    else:
        return Ingredients()


if __name__ == '__main__':
    print('Starting siMpLify')
    arguments = _args_to_dict()
    idea = _get_idea(arguments = arguments)
    depot = _get_depot(arguments = arguments)
    ingredients = _get_ingredients(arguments = arguments)
    Simplify(idea = idea, depot = depot, ingredients = ingredients)