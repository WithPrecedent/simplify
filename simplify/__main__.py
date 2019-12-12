"""
.. module:: siMpLify main
:synopsis: command-line data science made simple
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import simplify


def _args_to_dict() -> Dict[str, str]:
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

def main(
        idea: 'Idea',
        inventory: 'Inventory',
        ingredients: 'Ingredients') -> 'Project':
    """Creates Project from idea, inventory, and ingredients

    Args:
        idea ('Idea'): an Idea instance.
        inventory ('Inventory'): a Inventory instance.
        ingredients ('Ingredients'): an Ingredients instance.

    Returns:
        Project based upon passed attributes.

    """
    return simplify.make_project(
        idea = idea,
        inventory = inventory,
        ingredients = ingredients)

if __name__ == '__main__':
    # Gets command line arguments and converts them to dict.
    arguments = _args_to_dict()
    # Creates Idea, Inventory, and Ingredients instances from passed arguments.
    idea, inventory, ingredients = simplify.startup(
        idea = arguments.get('-idea'),
        inventory = arguments.get('-inventory'),
        ingredients = arguments.get('-ingredients'))
    # Calls 'main' function to return Project.
    main(idea = idea, inventory = inventory, ingredients = ingredients)