"""
harvest.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Harvest class, which handles
the harvest planning and implementation.
"""
from dataclasses import dataclass

from .steps import Sow
from .steps import Reap
from .steps import Clean
from .steps import Bundle
from .steps import Deliver
from ..planner import Planner


@dataclass
class Harvest(Planner):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Attributes:
        ingredients: an instance of Ingredients.
        menu: an instance of Menu.
        inventory: an instance of Inventory. If one is not passed when Alamanac
            is instanced, one will be created with default options.
        instructions: an instance of Instructions that gives rules and guidance
            for how the data should be parsed, munged, and delivered depending
            upon the step of the Harvest instance.
        steps: string or list containing steps of the Harvest to be
            implemented. This parameter should only be used if the user wants
            to override the menu settings (which will be used if left as None).
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
    """

    menu : object
    inventory : object = None
    steps : object = None
    ingredients : object
    instructions : object = None
    steps : object = None
    auto_prepare : bool = True
    name : str = 'harvest'

    def __post_init__(self):
        """Sets up the core attributes of Harvest."""
        # Declares possible classes and steps in a cookbook recipe.
        self.steps = {'sow' : Sow,
                      'reap' : Reap,
                      'clean' : Clean,
                      'bundle' : Bundle,
                      'deliver' : Deliver}
        super().__post_init__()
        return self

    def _parse_instructions(self, step, technique):
        return kwargs

    def start(self):
        """Iterates through each of the possible steps."""
        for step in self._listify(self.steps):
            self.step = self.steps[step](menu = self.menu,
                                         inventory = self.inventory,
                                         ingredients = self.ingredients,
                                         instructions = self.instructions,
                                         auto_prepare = self.auto_prepare)
            self.step.prepare()
            self.step.start()
            if self.conserve_memory:
                del(self.step)
        return self

    def prepare(self):
        return self