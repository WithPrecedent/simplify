"""
harvest.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Harvest class, which handles
the harvest planning and implementation.
"""
from dataclasses import dataclass

from .almanac import Almanac
from .steps import Sow, Reap, Clean, Bundle, Deliver
from ..managers import Planner


@dataclass
class Harvest(Planner):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Attributes:

        menu: an instance of Menu or a string containing the path where a menu
            settings file exists.
        inventory: an instance of Inventory. If one is not passed when Harvest
            is instanced, one will be created with default options.
        steps: a dictionary of step names and corresponding classes. steps
            should only be passed if the user wants to override the options
            selected in the menu settings.
        ingredients: an instance of Ingredients (or a subclass).
        almanac: an instance of Almanac that gives rules and guidance for how
            the data should be parsed, munged, and delivered depending upon the
            step of the Harvest instance.
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    menu : object
    inventory : object = None
    steps : object = None
    ingredients : object = None
    almanac : object = None
    auto_prepare : bool = True
    name : str = 'harvest'

    def __post_init__(self):
        """Sets up the core attributes of Harvest."""
        self.plan_class = Almanac
        self._set_defaults()
        super().__post_init__()
        return self

    @property
    def almanac(self):
        return self.plans

    def _check_sections(self):
        if not hasattr(self, 'sections') or not self.sections:
            self.sections = self.default_sections
        return self

    def _conform(self):
        self._conform_datatypes()
        return self

    def _conform_datatypes(self):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the active step.
        """
        for section, datatype in self.sections.items():
            if self.step in ['reaper', 'cleaner']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.sections[section] = str
            elif self.step in ['bundler', 'delivery']:
                if datatype in ['list', 'pattern']:
                    self.sections[section] = 'category'
        return self

    def _localize(self):
        """Adds menu and inventory options where needed to step and technique
        classes.
        """
        self._check_steps()
        self.menu.localize(instance = self, sections = ['general'])
        for step, step_class in self.steps.items():
            self.menu.localize(instance = step_class,
                               sections = ['general', 'files'])
            step_class.inventory = self.inventory
            for technique, algorithm in step_class.options.items():
                if not hasattr(self, 'default_' + technique):
                    setattr(self, 'default_' + technique, {})
                self.menu.localize(instance = algorithm,
                                   sections = ['general', 'files'])
                algorithm.inventory = self.inventory
        return self

    def _set_defaults(self):
        """ Declares default step names and classes in a harvest almanac plan.
        """
        self.options = {'sower' : Sow,
                              'reaper' : Reap,
                              'cleaner' : Clean,
                              'bundler' : Bundle,
                              'delivery' : Deliver}
        self.default_sections = {}
        self.plan_class = Almanac
        self.planner_type = 'sequence'
        return self

    def add_sections(self, sections):
        self.almanac._check_sections()
        self.almanac.sections.update(sections)
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
        self.set_order()
        if self.almanac == None:
            self.almanac = self.plan_class(menu = self.menu,
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
        for self.step in self.order:
            self.almanac.start(step = self.step)
            self.ingredients = self.step.start(ingredients = self.ingredients,
                                               almanac = self.almanac)
        return self