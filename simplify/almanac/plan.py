
from dataclasses import dataclass


@dataclass
class Plan(object):
    """Defines rules for sowing, reaping, cleaning, bundling, and delivering
    data as part of the siMpLify Almanac subpackage.

    Attributes:
        techniques: a list of Almanac step techniques to complete.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    techniques : object = None
    name : str = 'plan'

    def __post_init__(self):
        if not self.techniques:
            self.techniques = []
        return self

    def _check_attributes(self):
        for technique in self.techniques:
            if not hasattr(self, technique):
                error = technique + ' has not been passed to Plan class.'
                raise AttributeError(error)
        return self

    def prepare(self):
        self._check_attributes()
        return self

    def start(self, ingredients):
        """Applies the Almanac technique classes to the passed ingredients."""
        self.ingredients = ingredients
        for technique in self.techniques:
            self.ingredients = technique.start(self.ingredients)
        return self