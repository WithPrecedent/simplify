
from dataclasses import dataclass

from ..managers import Plan


@dataclass
class Almanac(Plan):
    """Defines rules for sowing, reaping, cleaning, bundling, and delivering
    data as part of the siMpLify Harvest subpackage.

    Attributes:
        steps: dictionary of steps containing the name of the step and
            corresponding classes. Dictionary keys and values should be placed
            in order that they should be completed.
    """
    steps : object = None
    name : str = 'almanac'
    structure : str = 'sequence'

    def __post_init__(self):
        super().__post_init__()
        return self

    def prepare(self):
        for step, step_class in self.steps.items():
            step_class.menu = self.menu
            step_class.inventory = self.inventory
            step_class.prepare(self)
            for technique, technique_class in step_class.options.items():
                technique_class.menu = self.menu
                technique_class.inventory = self.inventory
                technique_class.prepare(self)
        return self

    def start(self, ingredients):
        """Applies the Harvest step classes to the passed ingredients."""
        self.ingredients = ingredients
        for step, step_classes in self.steps.items():
            for step_class in step_classes:
                self.ingredients = step_class.start(self.ingredients, self)
        return self