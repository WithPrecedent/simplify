"""
almanac.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Almanac class, which handles
the almanac planning and implementation.
"""
from dataclasses import dataclass

from .stages import Cultivate
from .stages import Reap
from .stages import Thresh
from .stages import Bale
from .stages import Clean
from .blackacre import Blackacre
from ..inventory import Inventory


@dataclass
class Almanac(Blackacre):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Attributes:
        ingredients: an instance of Ingredients.
        menu: an instance of Menu.
        inventory: an instance of Inventory. If one is not passed when Alamanac
            is instanced, one will be created with default options.
    """
    ingredients : object
    menu : object
    inventory : object = None
    recipes : object = None

    def __post_init__(self):
        """Sets up the core attributes of Almanac."""
        # Local attributes are added from the Menu instance.
        self.menu.localize(instance = self, sections = ['general',
                                                        'harvesters'])
        # Declares possible classes and steps in a cookbook recipe.
        self.stages = {'plant' : Plant,
                       'reap' : Reap,
                       'thresh' : Thresh,
                       'bale' : Bale,
                       'clean' : Clean}
        # Calls method to set various default or user options.
        self._set_defaults()
        return self

    def _prepare_stages(self):
        """Initializes the stage classes for use by the Almanac."""
        for stage, class_name in self.stages.items():
            setattr(self, stage, self._listify(getattr(self, stage)))
        return self

    def _set_defaults(self):
        """Sets default attributes depending upon arguments passed when the
        Almanac is instanced.
        """
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        return self

    def create(self):
        """Iterates through each of the possible stages."""
        return self

    def prepare(self):
        """Creates the almanac plan with selected data preparation methods."""
        self._prepare_stages()
        return self