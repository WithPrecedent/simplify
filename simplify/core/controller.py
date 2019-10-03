"""
.. module:: main
:synopsis: controls all siMpLify packages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

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
        auto_read(bool): sets whether to automatically call the 'read'
            method when the class is instanced.

    """

    idea: object = None
    ingredients: object = None
    depot: object = None
    name: str = 'simplify'
    auto_publish: bool = True
    auto_read: bool = False

    def __post_init__(self):
        super().__post_init__()
        return self

    @localize
    def __call__(self, **kwargs):
        """Calls the class as a function.

        Only keyword arguments are accepted so that they can be properly
        turned into local attributes. Those attributes are then used by the
        various 'read' methods.

        Args:
            **kwargs(list(Recipe) and/or Ingredients): variables that will
                be turned into localized attributes.
        """
        self.__post_init__()
        self.read(**kwargs)
        return self

    """ Private Methods """

    def _artist_read(self):
        self.artist.read(
                ingredients = self.ingredients,
                recipes = self.chef.recipes,
                reviews = self.critic.reviews)
        return self

    def _chef_read(self):
        self.chef.read(ingredients = self.ingredients)
        return self

    def _critic_read(self):
        self.critic.read(
                ingredients = self.ingredients,
                recipes = self.chef.recipes)
        return self

    def _farmer_read(self):
        self.farmer.read(ingredients = self.ingredients)
        return self

    """ Core siMpLify Methods """

    def draft(self):
        self.options = {
                'farmer': ['simplify.farmer', 'Almanac'],
                'chef': ['simplify.chef', 'Cookbook'],
                'critic': ['simplify.critic', 'Review'],
                'artist': ['simplify.artist', 'Canvas']}
        self.checks = ['depot', 'ingredients']
        return self

    def publish(self):
        self.packages = {}
        for name, settings in self.options.items():
            if name in self.subpackages:
                self.packages.update({name: settings})
        return self

#    @localize
    def read(self, **kwargs):
        for package_name, package_class in self.packages.items():
            setattr(self, package_name, package_class())
            getattr(self, '_' + package_name + '_read')()
        return self