"""
.. module:: simplify
:synopsis: controls projects involving multiple siMpLify packages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.decorators import localize
from simplify.core.iterable import SimpleIterable


# Sets default options for module if 'options' not passed to a subclass.
DEFAULT_OPTIONS = {
    'farmer': ['simplify.farmer', 'Almanac'],
    'chef': ['simplify.chef', 'Cookbook'],
    'critic': ['simplify.critic', 'Review'],
    'artist': ['simplify.artist', 'Canvas']}
DEFAULT_CHECKS = ['ingredients']


@dataclass
class Simplify(SimpleIterable):
    """Controller class for completely automated siMpLify projects.

    This class is provided for applications that rely exclusively on Idea
    settings and/or subclass attributes. For a more customized application,
    users can access the subpackages ('farmer', 'chef', 'critic', and 'artist')
    directly.

    Args:
        idea(Idea or str): an instance of Idea or a string containing the file
            path or file name (in the current working directory) where a
            supoorted settings file for an Idea instance is located. Once an
            Idea instance is created by a subclass of SimpleClass, it is
            automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        ingredients(Ingredients or str): an instance of Ingredients or a string
            containing the file path of where a data file for a pandas
            DataFrame is located.
        depot(Depot or str): an instance of Depot a string containing the full
            path of where the root folder should be located for file output.
            Once a Depot instance is created by a subclass of SimpleClass, it
            is automatically made available to all other SimpleClass subclasses
            that are instanced in the future.
        name(str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify package. This is
            used instead of __class__.__name__ so that subclasses can maintain
            the same string name without altering the formal class name.
        auto_publish(bool): sets whether to automatically call the 'publish'
            method when the class is instanced. If you do not plan to make any
            adjustments beyond the Idea configuration, this option should be
            set to True. If you plan to make such changes, 'publish' should be
            called when those changes are complete.
        auto_implement(bool): sets whether to automatically call the 'implement'
            method when the class is instanced.

    """

    idea: object = None
    ingredients: object = None
    depot: object = None
    steps: object = None
    name: str = 'simplify'
    auto_publish: bool = True
    auto_implement: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    def __call__(self, ingredients = None):
        """Calls the class as a function.

        Only keyword arguments are accepted so that they can be properly
        turned into local attributes. Those attributes are then used by the
        various 'implement' methods.

        Args:
            **kwargs(list(Recipe) and/or Ingredients): variables that will
                be turned into localized attributes.
        """
        self.__post_init__()
        self.implement(ingredients = ingredients)
        return self

    def _implement_dangerous(self):
        first_package = True
        for name, package in self.steps.items():
            if first_package:
                first_package = False
                package.implement(ingredients = self.ingredients)
            else:
                package.implement(previous_package = previous_package)
            previous_package = package
        return self

    def _implement_safe(self):
        for name, package in self.steps.items():
            if name in ['farmer']:
                package.implement(ingredients = self.ingredients)
                self.ingredients = package.ingredients
                del package
            if name in ['chef']:
                package.implement(ingredients = self.ingredients)
                self.ingredients = package.ingredients
                self.recipes = package.recipes
                del package
            if name in ['critic']:
                package.implement(ingredients = self.ingredients,
                                  recipes = self.recipes)
                self.ingredients = package.ingredients
                self.reviews = package.reviews
            if name in ['artist']:
                package.implement(ingredients = self.ingredients,
                                  recipes = self.recipes,
                                  reviews = self.reviews)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        super().draft()
        self.sequence_setting = 'packages'
        return self

    def implement(self, ingredients = None):
        if ingredients:
            self.ingredients = ingredients
        if self.conserve_memory:
            self._implement_safe()
        else:
            self._implement_dangerous()
        return self
