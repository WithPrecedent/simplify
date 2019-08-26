
from dataclasses import dataclass
import os
import warnings

import pandas as pd

from .inventory import Inventory
from .menu import Menu
from .tools import listify
from ..ingredients import Ingredients

@dataclass
class Planner(object):
    """Parent class for Cookbook and Almanac to provide shared methods for
    creating data science workflows. Can also be subclassed to create other
    Planners.
    """

    def __post_init__(self):
        """Sets class instance defaults and calls prepare method if
        auto_prepare = True.
        """
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Creates menu attribute if not passed when class is instanced.
        self._check_menu()
        # Calls default setter method if it exists.
        if hasattr(self, '_set_defaults'):
            self._set_defaults()
        # Checks menu, inventory, and other attributes in self.checks list.
        self._checks()
        # Outputs Planner status to console if verbose option is selected.
        if self.verbose:
            print('Creating', self.name, 'planner')
        # Calls prepare method if subclass has an auto_prepare attribute that
        # is set to True.
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def __call__(self, *args, **kwargs):
        return self.start(*args, **kwargs)

    def __str__(self):
        """Returns lowercase name of class."""
        return self.__class__.__name__.lower()

    def _checks(self):
        """Checks attributes from self.checks and initializes them if they do
        not exist by calling the appropriate method.
        """
        for check in self.checks:
            getattr(self, '_check_' + check)()
        return self

    def _check_ingredients(self, ingredients = None):
        """Checks if ingredients attribute exists. If so, it determines if it
        contains a file folder, file path, or Ingredients instance. Depending
        upon its type, different actions are taken to actually create an
        Ingredients instance. If ingredients is None, then an Ingredients
        instance is created with no pandas DataFrames within it.

        Arguments:
            ingredients: an Ingredients instance, a file path containing a
                DataFrame or Series to add to an Ingredients instance, or
                a folder containing files to be used to compose Ingredients
                DataFrames and/or Series.
        """
        if ingredients:
            self.ingredients = ingredients
        if (isinstance(self.ingredients, pd.Series)
                or isinstance(self.ingredients, pd.DataFrame)):
            self.ingredients = Ingredients(menu = self.menu,
                                           inventory = self.inventory,
                                           df = self.ingredients)
        elif isinstance(self.ingredients, str):
            if os.path.isfile(self.ingredients):
                df = self.inventory.load(folder = self.inventory.data,
                                         file_name = self.ingredients)
                self.ingredients = Ingredients(menu = self.menu,
                                               inventory = self.inventory,
                                               df = df)
            elif os.path.isdir(self.ingredients):
                self.inventory.create_glob(folder = self.ingredients)
                self.ingredients = Ingredients(menu = self.menu,
                                               inventory = self.inventory,
                                               auto_load = False)
        elif not self.ingredients:
            print('yes')
            self.ingredients = Ingredients(menu = self.menu,
                                           inventory = self.inventory)
        return self

    def _check_inventory(self):
        """Adds a Inventory instance with default menu if one is not passed
        when a Planner subclass is instanced.
        """
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        if self.name in ['cookbook']:
            self.inventory.conform(step = 'cook')
        else:
            self.inventory.conform(step = self.step)
        return self

    def _check_menu(self):
        """Loads menu from an .ini file if a string is passed to menu instead
        of a menu instance. injects sections of menu to Planner instance.
        """
        if isinstance(self.menu, str):
            self.menu = Menu(file_path = self.menu)
        # Adds attributes to class from appropriate sections of the menu.
        sections = ['general', 'files']
        if hasattr(self, 'name') and self.name in self.menu.configuration:
            sections.append(self.name)
        self.menu.inject(instance = self, sections = sections)
        return self

    def _check_name(self):
        """Checks if name attribute exists. If not, a default name is used.
        """
        if not hasattr(self, 'name'):
            self.name = 'planner'
        if self.name in self.menu.configuration:
            self.menu.inject(instance = self, sections = [self.name])
        return self

    def _check_options(self):
        if not hasattr(self, 'options'):
            self.options = {}
        return self

    def _check_steps(self):
        if not self.steps:
            if hasattr(self, self.name + '_steps'):
                self.steps = listify(getattr(self, self.name + '_steps'))
            else:
                self.steps = []
        else:
            self.steps = listify(self.steps)
        if self.steps:
            self.step = self.steps[0]
        return self

    def _prepare_plan_class(self):
        """Adds menu and inventory instances to plan_class and injects
        general menu attributes.
        """
        self.plan_class.menu = self.menu
        self.menu.inject(instance = self.plan_class, sections = ['general'])
        self.plan_class.inventory = self.inventory
        return self

    def _prepare_steps(self):
        """Adds menu and inventory instances to step classes and injects
        general menu attributes.
        """
        for step in listify(self.steps):
            self.options[step].menu = self.menu
            self.menu.inject(instance = self.options[step],
                               sections = ['general'])
            self.options[step].inventory = self.inventory
        return self

    def _set_defaults(self):
        """ Declares defaults for Planner."""
        self.options = {}
        self.plan_class = None
        self.checks = ['steps', 'inventory', 'ingredients', 'name', 'options']
        self.state_attributes = ['inventory', 'ingredients']
        return self

    def add_options(self, step = None, techniques = None, algorithms = None):
        """Adds new technique name and corresponding algorithm to the options
        dictionary.
        """
        if step:
            self.options[step].add_options(techniques = techniques,
                                           algorithms = algorithms)
        else:
            options = dict(zip(listify(techniques), listify(algorithms)))
            self.options.update(options)
        return self

    def add_parameters(self, step, parameters):
        """Adds parameter sets to the parameters dictionary of a prescribed
        step. """
        self.options[step].add_parameters(parameters = parameters)
        return self

    def add_runtime_parameters(self, step, parameters):
        """Adds runtime_parameter sets to the parameters dictionary of a
        prescribed step."""
        self.options[step].add_runtime_parameters(parameters = parameters)
        return self

    def add_step_class(self, step_name, step_class):
        self.options.update({step_name, step_class})
        return self

    def add_technique(self, step, technique, parameters = None):
        tool_instance = self.options[step](technique = technique,
                                           parameters = parameters)
        return tool_instance

    def conform(self, step = None):
        if not step:
            step = self.step
        for attribute in self.state_attributes:
            getattr(self, attribute).conform(step = step)
        return self

    def save(self):
        """Exports the list of plans to disc as one object."""
        self.inventory.save(variable = self,
                            folder = self.inventory.experiment,
                            file_name = self.name,
                            file_format = 'pickle')
        return self

    def save_plan(self, plan, file_path = None):
        """Saves an instance of the plan class."""
        self.inventory.save(variable = plan,
                            file_path = file_path,
                            folder = getattr(self.inventory, plan.name),
                            file_name = 'recipe',
                            file_format = 'pickle')
        return self