"""
almanac.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Almanac class, which handles
the planning and implementation for data gathering and preparation.
"""
from dataclasses import dataclass

from .plan import Plan
from .steps import Sow, Harvest, Clean, Bundle, Deliver
from ..implements import listify
from ..managers import Planner


@dataclass
class Almanac(Planner):
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
        self._set_defaults()
        super().__post_init__()
        return self

    @property
    def almanac(self):
        return self.plans

    @almanac.setter
    def almanac(self, plans):
        self.plans = plans

    def _check_defaults(self):
        for name, attribute in self.__dict__.items():
            if name.startswith('default_'):
                new_name = name.lstrip('default_')
                if not hasattr(self, new_name):
                    setattr(self, new_name, attribute)
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
            if self.step in ['reaper', 'cleaner']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.sections[section] = str
            elif self.step in ['bundler', 'delivery']:
                if datatype in ['list', 'pattern']:
                    self.sections[section] = 'category'
        return self

    def _get_plans(self, step):
        step_plans = []
        for technique, algorithm in self.options[step].options.items():
            if hasattr(self, technique):
                for section, parameters in getattr(self, technique):
                    instance = self.options[step](technique = technique,
                                                  parameters = parameters)
                    step_plans.append(instance)
        return step_plans

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
                if not hasattr(self, technique ):
                    if hasattr(self, 'default_' + technique):
                        setattr(self, technique , getattr(
                                self, 'default_' + technique))
                    else:
                        setattr(self, 'default_' + technique, {})
                self.menu.localize(instance = algorithm,
                                   sections = ['general', 'files'])
                algorithm.inventory = self.inventory
        return self

    def _prepare_one_loop(self):
        for i, plan in enumerate(self.all_plans):
            plan_instance = self.plan_class(techniques = self.steps)
            for j, step in enumerate(self.options.keys()):
                setattr(plan_instance, step, self.options[step](plan[j]))
            plan_instance.prepare()
            self.plans.append(plan_instance)
        return self

    def _prepare_one_step(self):
        for i, plan in enumerate(self.all_plans):
            plan_instance = self.plan_class(techniques = self.steps)
            self.plans.append(plan_instance)
        return self

    def _prepare_parameters(self, step):
        for technique, technique_class in step.options.items():
            for section, parameters in getattr(self, technique).items():
                final_parameters = parameters
                final_parameters.update({'section' : section})
                self.plans.append(technique_class(**final_parameters))
        return self

    def _prepare_plans(self):
        self.plans = []
        self.all_plans = []
        for step in self.options.keys():
            if step in self.steps:
                # Stores each step attribute in a list
                setattr(self, step, self.plan_class(
                        techniques = self._get_plans(step = step)))
                # Adds step to a list of all step lists
                self.all_plans.append(getattr(self, step))
        return self

    def _set_defaults(self):
        """ Declares default step names and classes in a harvest almanac plan.
        """
        super()._set_defaults()
        self.options = {'sower' : Sow,
                        'harvester' : Harvest,
                        'cleaner' : Clean,
                        'bundler' : Bundle,
                        'delivery' : Deliver}
        self.plan_class = Plan
        self.checks.extend(['sections', 'defaults'])
        return self

    def add_sections(self, sections):
        self.plans._check_sections()
        self.plans.sections.update(sections)
        return self

    def add_techniques(self, step, techniques, algorithms):
        self.plans[step].add_techniques(techniques, algorithms)
        return self

    def prepare(self):
        """Creates a harvest plan with all sequenced techniques applied at each
        step. Each set of methods is stored in a list within self.almanac.
        """
        if self.verbose:
            print('Preparing data harvester')
        self._prepare_plan_class()
        self._prepare_steps()
        self._prepare_plans()
        return self

    def start(self, ingredients = None):
        """Completes an iteration of a Harvest."""
        if ingredients:
            self.ingredients = ingredients
        for self.step in listify(self.steps):
            self._conform()
            self.ingredients = plan.start(ingredients = self.ingredients)
        return self