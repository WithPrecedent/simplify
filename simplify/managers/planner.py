
from dataclasses import dataclass
from itertools import product
import warnings

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
        # Checks menu and inventory arguments passed when class is instanced.
        self._check_menu()
        self._check_inventory()
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

    def _prepare_compare(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.plans.
        """
        self._prepare_compare_plans()
        self.comparer = self.compare_class(menu = self.menu,
                                           inventory = self.inventory)
        self.plans = []
        if 'train_test_val' in self.data_to_use:
            self._prepare_compare_one_loop(data_to_use = 'train_test')
            self._prepare_compare_one_loop(data_to_use = 'train_val')
        else:
            self._prepare_compare_one_loop(data_to_use = self.data_to_use)
        return self

    def _prepare_compare_one_loop(self, data_to_use):
        for i, plan in enumerate(self.all_plans):
            plan_instance = self.plan_class(steps = self.steps)
            setattr(plan_instance, 'number', i)
            setattr(plan_instance, 'data_to_use', data_to_use)
            for j, step in enumerate(self.options.keys()):
                setattr(plan_instance, step, self.options[step](plan[j]))
            plan_instance.prepare()
            self.plans.append(plan_instance)
        return self

    def _prepare_compare_plans(self):
        """Initializes the step classes for use by the Cookbook."""
        self.step_lists = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, listify(getattr(self, step)))
            # Adds step to a list of all step lists
            self.step_lists.append(getattr(self, step))
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is a 'plan'
        self.all_plans = list(map(list, product(*self.step_lists)))
        return self

    def _prepare_sequence(self):
        self.option_list = []
        plan_instance = self.plan_class(steps = self.steps)
        for step in self.steps:
            for option, option_class in step.options.items():
                self.option_list.append(getattr(plan_instance, step))
                step_instance = step()
                step.prepare()
                setattr(plan_instance, step, self.options[step])

        return self

    def _prepare_plan_class(self):
        self.plan_class.menu = self.menu
        self.menu.localize(instance = self.plan_class, sections = ['general'])
        self.plan_class.inventory = self.inventory
        return self

    def _prepare_steps(self):
        """Adds menu and inventory instances to step classes as needed."""
        if not self.steps:
            self.steps = getattr(self, self.name + '_steps')
        for step in self.steps:
            self.options[step].menu = self.menu
            self.menu.localize(instance = self.options[step],
                               sections = ['general'])
            self.options[step].inventory = self.inventory
        return self

    def _start_compare(self, data_to_use = None):
        """Completes an iteration of a Planner."""
        for plan in self.plans:
            if self.verbose:
                print('Testing ' + plan.name + ' ' + str(plan.number))
            self.inventory._set_plan_folder(
                    plan = plan, steps_to_use = self.naming_classes)
            plan.start(ingredients = self.ingredients)
            self.comparer.start(plan)
            self._check_best(plan)
            self.save_plan_report()
            # To conserve memory, each recipe is deleted after being exported.
            del(plan)
        return self

    def _start_sequence(self):

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

    def prepare(self):
        self._prepare_plan_class()
        self._prepare_steps()
        getattr(self, '_prepare_' + self.plan_class.structure)()
        return self

    def save(self):
        """Exports the list of plans to disc as one object."""
        file_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = self.name + '.pkl')
        self.inventory.pickle_object(self, file_path = file_path)
        return self

    def save_plan_report(self, report = None):
        if not report:
            report = getattr(self.comparer.review,
                             self.model_type + '_report')
        file_path = self.inventory.create_path(
                    folder = self.inventory.plan,
                    file_name = self.model_type + '_report',
                    file_type = 'csv')
        self.inventory.save_df(report, file_path = file_path)
        return

    def set_plan_class(self, plan_class):
        self.plan_class = plan_class
        return self

    def start(self, ingredients = None):
        if ingredients:
            self.ingredients = ingredients
        getattr(self, '_start_' + self.plan_class.structure)()
        return self