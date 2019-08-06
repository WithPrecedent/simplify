"""
almanac.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Almanac class, which handles
the planning and implementation for data gathering and preparation.
"""
from dataclasses import dataclass
from itertools import chain

from .plan import Plan
from .steps import Sow, Harvest, Clean, Bundle, Deliver
from ..implements.tools import listify
from ..implements.planner import Planner


@dataclass
class Almanac(Planner):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Attributes:

        menu: an instance of Menu or a string containing the path where a menu
            settings file exists.
        inventory: an instance of Inventory. If one is not passed when Almanac
            is instanced, one will be created with default options.
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the Menu
            steps.
        ingredients: an instance of Ingredients (or a subclass).
        plans: a list of instances of Plan which Almanac creates through
            the prepare method and applies through the start method.
            Ordinarily, a list of plans is not passed when Almanac is
            instanced, but the argument is included if the user wishes to
            reexamine past plans or manually add plans to an existing set.
            Alternatively, if plans can be a dictionary of settings if the user
            prefers not to subclass Almanac and/or use .csv file imports,
            and instead pass the needed settings in dictionary form (with the
            keys corresponding to the names of techniques used and the values
            including the parameters to be used).
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    menu : object
    inventory : object = None
    steps : object = None
    ingredients : object = None
    plans : object = None
    auto_prepare : bool = True
    name : str = 'almanac'

    def __post_init__(self):
        """Sets up the core attributes of Almanac."""
        super().__post_init__()
        return self

    def _check_defaults(self):
        for name in self.__dict__.copy().keys():
            if name.startswith('default_'):
                new_name = name.lstrip('default_')
                if not hasattr(self, new_name):
                    setattr(self, new_name, getattr(self, name))
        return self

    def _check_plans(self):
        if isinstance(self.plans, dict):
            for key, value in self.plans:
                setattr(self, key, value)
        return self

    def _check_sections(self):
        if not hasattr(self, 'sections') or not self.sections:
            if hasattr(self, 'default_sections'):
                self.sections = self.default_sections
            else:
                self.sections = {}
        return self

    def _conform(self):
        self._conform_datatypes()
        return self

    def _conform_datatypes(self):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the active step.
        """
        for section, datatype in self.sections.items():
            if self.step in ['harvest', 'cleane']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.sections[section] = str
            elif self.step in ['bundle', 'deliver']:
                if datatype in ['list', 'pattern']:
                    self.sections[section] = 'category'
        return self

    def _prepare_one_loop(self):
        self.plan = self.plan_class()
        for step_technique in self.plan_steps:
            print(step_technique)
            step_instance = self.technique_dict[step_technique](
                    technique = step_technique,
                    parameters = getattr(self, step_technique))
            self.plan.techniques.append(step_instance)
        return self

    def _prepare_plan(self):
        """Initializes the step classes for use by the Cookbook."""
        self.step_lists = []
        self.technique_dict = {}
        for step in listify(getattr(self, self.name + '_steps')):
            for technique in listify(getattr(self, step + '_techniques')):
                # Stores each step attribute in a dictionary
                setattr(self, step, listify(getattr(self, technique)))
                # Adds step to a list of all step lists
                self.step_lists.append(getattr(self, step))
                # Updates dict with information about techniques within step.
                self.technique_dict.update({technique : self.options[step]})
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is a 'plan'
        self.plan_steps = chain(*self.step_lists)
        return self

    def _set_defaults(self):
        """ Declares default step names and classes in a harvest almanac plan.
        """
        super()._set_defaults()
        self.options = {'sow' : Sow,
                        'harvest' : Harvest,
                        'clean' : Clean,
                        'bundle' : Bundle,
                        'deliver' : Deliver}
        self.plan_class = Plan
        self.checks.extend(['plans', 'sections', 'defaults'])
        return self

    def add_sections(self, sections):
        self.plan._check_sections()
        self.plan.sections.update(sections)
        return self

    def add_step(self, step_name, option, parameters):
        self.plan.techniques.append(self.options[step_name](
                technique = option,
                parameters = parameters))
        return self

    def add_step_class(self, step_name, step_class):
        self.options.update({step_name, step_class})
        return self

    def prepare(self):
        """Creates a harvest plan with all sequenced techniques applied at each
        step. Each set of methods is stored in a list within self.almanac.
        """
        if self.verbose:
            print('Preparing data harvester')
        self._prepare_plan_class()
        self._prepare_steps()
        self._prepare_plan()
        self._prepare_one_loop()
        return self

    def start(self, ingredients = None):
        """Completes an iteration of a Harvest."""
        if ingredients:
            self.ingredients = ingredients
        for step in listify(self.plan_steps):
            self._conform()
            self.ingredients = self.plan.start(ingredients = self.ingredients)
        return self