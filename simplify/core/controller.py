"""
.. module:: simplify
:synopsis: controls projects involving multiple siMpLify packages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

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
    """Controller class for siMpLify projects.

    This class is provided for applications that rely on Idea settings and/or
    subclass attributes. For a more customized application, users can access the
    subgetattr(self, name)s ('farmer', 'chef', 'critic', and 'artist') directly.

        name(str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify getattr(self,
            name). This is used instead of __class__.__name__ so that subclasses
            can maintain the same string name without altering the formal class
            name.
        auto_publish(bool): sets whether to automatically call the 'publish'
            method when the class is instanced. If you do not plan to make any
            adjustments beyond the Idea configuration, this option should be
            set to True. If you plan to make such changes, 'publish' should be
            called when those changes are complete.
        auto_implement(bool): sets whether to automatically call the 'implement'
            method when the class is instanced.

    """
    steps: object = None
    name: str = 'simplify'
    auto_publish: bool = True
    auto_implement: bool = False
    sequence_setting: str = 'packages'
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        super().__post_init__()
        return self

    def __call__(self, ingredients = None):
        """Calls the class as a function.

        Args:

            ingredients(Ingredients or str): an instance of Ingredients, a
                string containing the full file path of where a data file for a
                pandas DataFrame is located, or a string containing a file name
                in the default data folder, as defined in a Depot instance.

        """
        self.__post_init__()
        self.implement(ingredients = ingredients)
        return self

    def _implement_dangerous(self):
        """Implements steps without concern for memory consumption."""
        first_step = True
        for name in self.sequence:
            if first_step:
                first_step = False
                getattr(self, name).implement(ingredients = self.ingredients)
            else:
                getattr(self, name).implement(previous_step = previous_step)
            previous_step = getattr(self, name)
        return self

    def _implement_safe(self):
        """Implements steps while attempting to conserve memory."""
        for name in self.sequence:
            if name in ['farmer']:
                getattr(self, name).implement(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                delattr(self, name)
            if name in ['chef']:
                getattr(self, name).implement(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
            if name in ['critic', 'artist']:
                getattr(self, name).implement(
                    recipes = self.recipes)
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
        return self

    """ Core siMpLify Methods """

    def implement(self, ingredients = None):
        """Implements steps in 'sequence'."""
        if ingredients:
            self.ingredients = ingredients
        if self.conserve_memory:
            self._implement_safe()
        else:
            self._implement_dangerous()
        return self
