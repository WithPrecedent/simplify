"""
almanac.py is the primary control file for the data gathering and processing
portions of the siMpLify package. It contains the Almanac class, which handles
the planning and implementation for data gathering and preparation.
"""
from dataclasses import dataclass

from simplify.almanac.plan import Plan
from simplify.almanac.steps import Sow, Harvest, Clean, Bundle, Deliver
from simplify.core.base import SimpleClass


@dataclass
class Almanac(SimpleClass):
    """Implements data parsing, wrangling, munging, merging, engineering, and
    cleaning methods for the siMpLify package.

    Parameters:

        ingredients: an instance of Ingredients (or a subclass).
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the steps
            listed in menu.configuration.
        plans: a list of instances of steps which Almanac creates
            through the prepare method and applies through the start method.
            Ordinarily, a list of plan is not passed when Almanac is
            instanced, but the argument is included if the user wishes to
            reexamine past plan or manually add plan to an existing set.
            Alternatively, plan can be a dictionary of settings if the user
            prefers not to subclass Almanac and/or use .csv file imports,
            and instead pass the needed settings in dictionary form (with the
            keys corresponding to the names of techniques used and the values
            including the parameters to be used).
        name: a string designating the name of the class which should be
            identical to the section of the menu configuration with relevant
            settings.
        auto_prepare: a boolean value that sets whether the prepare method is
            automatically called when the class is instanced.
        auto_start: sets whether to automatically call the 'start' method
            when the class is instanced.

    """
    ingredients : object = None
    steps : object = None
    plans : object = None
    name : str = 'almanac'
    auto_prepare : bool = True
    auto_start : bool = True

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

    def _define(self):
        self.index_column = 'index_universal'
        self.metadata_columns = []
        return self

    def _prepare_plan(self):
        """Initializes the step classes for use by the Almanac."""
        self.plans = []
        for step in self.steps:
            step_instance = self.plan_class(name = step,
                                            index_column = self.index_column)
            for technique in listify(getattr(self, step + '_techniques')):
                tool_instance = self.add_technique(
                        step = step,
                        technique = technique,
                        parameters = listify(getattr(self, technique)))
                step_instance.techniques.append(tool_instance)
            step_instance.prepare()
            self.plans.append(step_instance)
        return self

    def _set_columns(self, organizer):
        if not hasattr(self, 'columns'):
            self.columns = {self.index_column : int}
            if self.metadata_columns:
                self.columns.update(self.metadata_columns)
        self.columns.update(dict.fromkeys(self.columns, str))
        return self

    def _define(self):
        """ Declares default step names and classes in an Almanac."""
        super()._define()
        self.options = {'sow' : Sow,
                        'harvest' : Harvest,
                        'clean' : Clean,
                        'bundle' : Bundle,
                        'deliver' : Deliver}
        self.plan_class = Plan
        self.checks.extend(['plans', 'sections', 'defaults'])
        return self

    def prepare(self):
        """Creates a Almanac with all sequenced techniques applied at each
        step. Each set of methods is stored in a list within a Plan instance.
        """
        if self.verbose:
            print('Preparing Almanac')
        self._prepare_plan_class()
        self._prepare_steps()
        self._prepare_plan()
        if hasattr(self, '_set_folders'):
            self._set_folders()
        return self

    def start(self, ingredients = None):
        """Completes an iteration of an Almanac."""
        if not ingredients:
            ingredients = self.ingredients
        for plan in self.plans:
            self.step = plan.name
            # Adds initial columns dictionary to ingredients instance.
            if (self.step in ['harvest']
                    and 'organize' in self.harvest_techniques):
                self._set_columns(organizer = plan)
                ingredients.columns = self.columns
            self.conform(step = self.step)
            self.ingredients = plan.start(ingredients = self.ingredients)
            self.inventory.save(variable = self.ingredients,
                                file_name = self.step + '_ingredients')
        return self