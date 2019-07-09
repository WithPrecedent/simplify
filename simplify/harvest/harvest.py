"""
harvest.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Harvest class, which handles
the harvest planning and implementation.
"""
from dataclasses import dataclass

from .almanac import Almanac
from ..managers import Planner


@dataclass
class Harvest(Planner):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Attributes:

        menu: an instance of Menu.
        inventory: an instance of Inventory. If one is not passed when Harvest
            is instanced, one will be created with default options.
        steps: a dictionary of step names and corresponding classes.
        almanac: an instance of Almanac that gives rules and guidance for how
            the data should be parsed, munged, and delivered depending upon the
            step of the Harvest instance.
        ingredients: an instance of Ingredients.
        almanac_class: Almanac (or a subclass) which contains settings for
            applying various Harvest step methods.
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    menu : object
    inventory : object = None
    ingredients : object = None
    almanac_class : object = None
    auto_prepare : bool = True
    name : str = 'harvest'

    def __post_init__(self):
        """Sets up the core attributes of Harvest."""
        # Declares default step names and classes in a harvest almanac.
        if not self.almanac_class:
            self.almanac_class = Almanac
        super().__post_init__()
        return self

    def add_techniques(self, step, techniques, algorithms):
        self.almanac[step].add_techniques(techniques, algorithms)
        return self

    def prepare(self):
        """Creates a planner with all sequenced techniques applied at each step
        of the harvest plan. Each set of methods is stored in a list within
        self.almanac.
        """
        if self.verbose:
            print('Preparing data harvester')
        self.almanac = self.almanac_class(menu = self.menu,
                                          inventory = self.inventory,
                                          order = self.almanac_order)
        self.almanac.prepare()
        return self

    def start(self, ingredients = None):
        """Iterates through Almanac."""
        if self.verbose:
            print('Harvesting data')
        if ingredients:
            self.ingredients = ingredients
        else:
            error = 'Harvest.start requires an Ingredients instance.'
            raise AttributeError(error)
        for self.step in self.almanac_order:
            self.almanac.start(step = self.step)
            self.ingredients = self.step.start(ingredients = self.ingredients,
                                               almanac = self.almanac)
        return self