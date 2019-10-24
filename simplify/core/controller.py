"""
.. module:: simplify
:synopsis: controls projects involving multiple siMpLify packages
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclasses

from simplify.core.decorators import localize
from simplify.core.package import SimplePackage


@dataclass
class Simplify(SimplePackage):
    """Controller class for siMpLify projects.

    This class is provided for applications that rely on Idea settings and/or
    subclass attributes. For a more customized application, users can access the
    subgetattr(self, name)s ('farmer', 'chef', 'actuary', 'critic', and
    'artist') directly.

        name (str): name of class used to match settings sections in an Idea
            settings file and other portions of the siMpLify getattr(self,
            name). This is used instead of __class__.__name__ so that subclasses
            can maintain the same string name without altering the formal class
            name.

    """
    name: str = 'simplify'
    steps: object = None

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
        self.publish(ingredients = ingredients)
        return self

    def _publish_dangerous(self):
        """Implements steps without concern for memory consumption."""
        first_step = True
        for step in self.order:
            if first_step:
                first_step = False
                getattr(self, step).publish(ingredients = self.ingredients)
            else:
                getattr(self, step).publish(
                    ingredients = self.ingredients,
                    previous_step = previous_step)
            previous_step = getattr(self, step)
        return self

    def _publish_safe(self):
        """Implements steps while attempting to conserve memory."""
        for name in self.order:
            if name in ['farmer']:
                getattr(self, name).publish(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                delattr(self, name)
            if name in ['chef', 'actuary']:
                getattr(self, name).publish(ingredients = self.ingredients)
                self.ingredients = getattr(self, name).ingredients
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
            if name in ['critic', 'artist']:
                getattr(self, name).publish(recipes = self.recipes)
                self.recipes = getattr(self, name).recipes
                delattr(self, name)
        return self

    """ Core siMpLify Methods """

    def publish(self, ingredients = None):
        """Implements steps in 'order'."""
        if ingredients:
            self.ingredients = ingredients
        if self.conserve_memory:
            self._publish_safe()
        else:
            self._publish_dangerous()
        return self
