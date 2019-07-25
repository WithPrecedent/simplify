
from dataclasses import dataclass
import warnings

from .plan import Plan
from ..implements import listify
from ..inventory import Inventory
from ..menu import Menu


@dataclass
class Planner(object):
    """Parent class for Cookbook and Harvest to provide shared methods for
    creating data science workflows.

    Attributes:
        menu: an instance of Menu or a string containing the path where a menu
            settings file exists.
        inventory: an instance of Inventory. If one is not passed when a
            Planner is instanced, one will be created with default options.
    """
    menu : object
    inventory : object = None

    def __post_init__(self):
        """Implements basic settings for Planner subclasses."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Checks menu, inventory, and other attributes in self.checks.
        self._checks()
        # Adds attributes to class from appropriate sections of the menu.
        sections = ['general', 'files']
        if hasattr(self, 'name') and self.name in self.menu.config:
            sections.append(self.name)
        self.menu.localize(instance = self, sections = sections)
        # Outputs Planner status to console if verbose option is selected.
        if self.verbose:
            print('Creating', self.name, 'planner')
        # Calls prepare method if subclass has an auto_prepare attribute that
        # is set to True.
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def _checks(self):
        """Checks attributes from self.checks and initializes if they do not
        exist.
        """
        for check in self.checks:
            getattr(self, '_check_' + check)()
        return self

    def _check_inventory(self):
        """Adds a Inventory instance with default menu if one is not passed
        when a Planner subclass is instanced.
        """
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        return self

    def _check_menu(self):
        """Loads menu from an .ini file if a string is passed to menu instead
        of a menu instance.
        """
        if isinstance(self.menu, str):
            self.menu = Menu(file_path = self.menu)
        return self

    def _check_name(self):
        """Checks if name attribute exists. If not, a default name is used.
        """
        if not hasattr(self, 'name'):
            self.name = 'planner'
        return self

    def _check_options(self):
        if not hasattr(self, 'options'):
            self.options = {}
        return self

    def _check_steps(self):
        if not self.steps:
            if hasattr(self, self.name + '_steps'):
                self.steps = getattr(self, self.name + '_steps')
            else:
                self.steps = []
        return self

    def _prepare_plan_class(self):
        self.plan_class.menu = self.menu
        self.menu.localize(instance = self.plan_class, sections = ['general'])
        self.plan_class.inventory = self.inventory
        return self

    def _prepare_steps(self):
        """Adds menu and inventory instances to step classes as needed."""
        for step in self.steps:
            self.options[step].menu = self.menu
            self.menu.localize(instance = self.options[step],
                               sections = ['general'])
            self.options[step].inventory = self.inventory
        return self

    def _set_defaults(self):
        """ Declares defaults for Planner."""
        self.options = {}
        self.plan_class = Plan
        self.checks = ['menu', 'inventory', 'name', 'options', 'steps']
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

    def save(self):
        """Exports the list of plans to disc as one object."""
        file_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = self.name + '.pkl')
        self.inventory.pickle_object(self, file_path = file_path)
        return self