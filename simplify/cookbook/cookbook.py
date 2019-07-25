"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
from itertools import product

from .recipe import Recipe
from .steps import Cleave, Encode, Mix, Model, Reduce, Sample, Scale, Split
from ..critic import Critic
from ..implements import listify
from ..managers import Planner


@dataclass
class Cookbook(Planner):
    """Dynamically creates recipes for final preprocessing, machine learning,
    and data analysis using a unified interface and architecture.

    Attributes:
        menu: an instance of Menu or a string containing the path where a menu
            settings file exists.
        inventory: an instance of Inventory. If one is not passed when Cookbook
            is instanced, one will be created with default options.
        steps: a dictionary of step names and corresponding classes. steps
            should only be passed if the user wants to override the options
            selected in the menu settings.
        ingredients: an instance of Ingredients (or a subclass).
        recipes: a list of instances of Recipe which Cookbook creates through
            the prepare method and applies through the start method.
            Ordinarily, recipes is not passed when Cookbook is instanced, but
            the argument is included if the user wishes to reexamine past
            recipes or manually create recipes.
        auto_prepare: sets whether to automatically call the prepare method
            when the class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            menu, this option should be set to True. If you plan to make such
            changes, prepare should be called when those changes are complete.
        name: a string designating the name of the class which should be
            identical to the section of the menu with relevant settings.
    """
    menu : object
    inventory : object = None
    steps : object = None
    ingredients : object = None
    recipes : object = None
    auto_prepare : bool = True
    name : str = 'cookbook'

    def __post_init__(self):
        """Sets up the core attributes of Cookbook."""
        self._set_defaults()
        super().__post_init__()
        return self

    @property
    def recipes(self):
        return self.plans

    @recipes.setter
    def recipes(self, plans):
        self.plans = plans

    def _check_best(self, recipe):
        """Checks if the current Recipe is better than the current best Recipe
        based upon the primary scoring metric.
        """
        if not self.best_recipe:
            self.best_recipe = recipe
            self.best_recipe_score = self.critic.review.report.loc[
                    self.critic.review.report.index[-1],
                    listify(self.metrics)[0]]
        elif (self.critic.review.report.loc[
                self.critic.review.report.index[-1],
                listify(self.metrics)[0]] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.critic.review.report.loc[
                    self.critic.review.report.index[-1],
                    listify(self.metrics)[0]]
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        self.ingredients.split_xy(label = self.label)
        Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                  ((self.ingredients.y == 1).sum())) - 1
        return self

    def _prepare_one_loop(self, data_to_use):
        for i, plan in enumerate(self.all_plans):
            plan_instance = self.plan_class(techniques = self.steps)
            setattr(plan_instance, 'number', i + 1)
            setattr(plan_instance, 'data_to_use', data_to_use)
            for j, step in enumerate(self.options.keys()):
                setattr(plan_instance, step, self.options[step](plan[j]))
            plan_instance.prepare()
            self.plans.append(plan_instance)
        return self

    def _prepare_plans(self):
        """Initializes the step classes for use by the Cookbook."""
        self.plans = []
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

    def _set_defaults(self):
        """ Declares default step names and classes in a Cookbook recipe."""
        super()._set_defaults()
        self.options = {'scaler' : Scale,
                        'splitter' : Split,
                        'encoder' : Encode,
                        'mixer' : Mix,
                        'cleaver' : Cleave,
                        'sampler' : Sample,
                        'reducer' : Reduce,
                        'model' : Model}
        self.plan_class = Recipe
        self.best_recipe = None
        return self

    def add_cleave(self, cleave_group, prefixes = [], columns = []):
        """Adds cleaves to the list of cleaves."""
        if not hasattr(self.cleaves) or not self.cleaves:
            self.cleaves = []
        columns = self.ingredients.create_column_list(prefixes = prefixes,
                                                      columns = columns)
        Cleave.add(cleave_group = cleave_group, columns = columns)
        self.cleaves.append(cleave_group)
        return self

    def add_recipe(self, recipe):
        if hasattr(self, 'recipes'):
            self.recipes.append(recipe)
        else:
            self.recipes = [recipe]
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc."""
        recipe = self.inventory.unpickle_object(file_path)
        self.add_recipe(recipe = recipe)
        return self

    def prepare(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.plans.
        """
        Model.search_parameters = self.menu['search_parameters']
        self._prepare_plan_class()
        self._prepare_steps()
        self._prepare_plans()
        self.critic = Critic(menu = self.menu, inventory = self.inventory)
        if 'train_test_val' in self.data_to_use:
            self._prepare_one_loop(data_to_use = 'train_test')
            self._prepare_one_loop(data_to_use = 'train_val')
        else:
            self._prepare_one_loop(data_to_use = self.data_to_use)
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  listify(self.metrics)[0], 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            for step in self.best_recipe.steps:
                print(step.capitalize(), ':',
                      getattr(self.best_recipe, step).technique)
        return

    def save_all_recipes(self, recipes = None):
        if not recipes:
            recipes = self.recipes
        for recipe in recipes:
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            recipe_path = self.inventory.create_path(
                    folder = self.inventory.recipe,
                    file_name = file_name,
                    file_type = 'pickle')
            self.save_recipe(recipe = recipe, file_path = recipe_path)
        return

    def save_best_recipe(self):
        if hasattr(self, 'best_recipe'):
            best_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'best_recipe.pkl')
            self.inventory.pickle_object(self.best_recipe,
                                         file_path = best_path)
        return self

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.save_review()
        self.save_best_recipe()
        self.ingredients.save_drops()
        return self

    def save_recipe(self, recipe, file_path):
        """Exports a recipe to disc."""
        self.inventory.pickle_object(recipe, file_path)
        return self

    def save_review(self, review = None):
        if not review:
            review = getattr(self.critic.review,
                             self.model_type + '_report')
        file_path = self.inventory.create_path(
                    folder = self.inventory.plan,
                    file_name = self.model_type + '_report',
                    file_type = 'csv')
        self.inventory.save_df(review, file_path = file_path)
        return

    def start(self, ingredients = None):
        """Completes an iteration of a Cookbook."""
        if ingredients:
            self.ingredients = ingredients
        for plan in self.plans:
            if self.verbose:
                print('Testing ' + plan.name + ' ' + str(plan.number))
            self.inventory._set_plan_folder(
                    plan = plan, steps_to_use = self.naming_classes)
            plan.start(ingredients = self.ingredients)
            self.critic.start(plan)
            self._check_best(plan)
            self.save_report()
            # To conserve memory, each recipe is deleted after being exported.
            del(plan)
        return self