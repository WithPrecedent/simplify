"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
import datetime
from itertools import product

from .critic import Critic
from .recipe import Recipe
from .steps import Cleave, Encode, Mix, Model, Reduce, Sample, Scale, Split
from ..tools import listify
from ..planner import Planner


@dataclass
class Cookbook(Planner):
    """Dynamically creates recipes for final preprocessing, machine learning,
    and data analysis using a unified interface and architecture.

    Attributes:
        menu: an instance of Menu or a string containing the path where a menu
            settings file exists.
        inventory: an instance of Inventory. If one is not passed when Cookbook
            is instanced, one will be created with default options.
        steps: an ordered list of step names to be completed. This argument
            should only be passed if the user whiches to override the Menu
            steps.
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
    menu : object = None
    inventory : object = None
    steps : object = None
    ingredients : object = None
    recipes : object = None
    auto_prepare : bool = True
    name : str = 'cookbook'
    step : str = 'cook'

    def __post_init__(self):
        """Sets up the core attributes of a Cookbook instance."""
        super().__post_init__()
        return self

    def _check_best(self, recipe):
        """Checks if the current recipe is better than the current best recipe
        based upon the primary scoring metric.

        Parameters:
            recipe: an instance of Recipe.
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
        """Computes hyperparameters that can be determined by the source data
        (without creating data leakage problems).
        """
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        self.ingredients.split_xy(label = self.label)
        # Model class is injected with scale_pos_weight for algorithms that
        # use that parameter.
        Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                  ((self.ingredients.y == 1).sum())) - 1
        return self

    def _prepare_one_loop(self, data_to_use):
        """Prepares one set of recipes from all_recipes as applied to a
        specific training/testing set.

        Parameters:
            data_to_use: a string corresponding to an Ingredients property
                which will return the appropriate training/testing set.
        """
        for i, plan in enumerate(self.all_recipes):
            plan_instance = self.plan_class(techniques = self.steps)
            setattr(plan_instance, 'number', i + 1)
            setattr(plan_instance, 'data_to_use', data_to_use)
            for j, step in enumerate(self.options.keys()):
                setattr(plan_instance, step, self.options[step](plan[j]))
            plan_instance.prepare()
            self.recipes.append(plan_instance)
        return self

    def _prepare_recipes(self):
        """Initializes the step classes for use by the Cookbook."""
        self.recipes = []
        self.step_lists = []
        for step in self.options.keys():
            # Stores each step attribute in a list
            setattr(self, step, listify(getattr(self, step)))
            # Adds step to a list of all step lists
            self.step_lists.append(getattr(self, step))
        # Creates a list of all possible permutations of step techniques
        # selected. Each item in the the list is a 'plan'
        self.all_recipes = list(map(list, product(*self.step_lists)))
        return self

    def _set_defaults(self):
        """ Declares default step names and classes in a Cookbook recipe."""
        # Initially sets defaults from parent class.
        super()._set_defaults()
        # Sets options for default steps of a Recipe.
        self.options = {'scaler' : Scale,
                        'splitter' : Split,
                        'encoder' : Encode,
                        'mixer' : Mix,
                        'cleaver' : Cleave,
                        'sampler' : Sample,
                        'reducer' : Reduce,
                        'model' : Model}
        # Assigns the particular plan_class to Recipe so that parent class
        # methods will point to the proper plan class.
        self.plan_class = Recipe
        # Initializes the best_recipe.
        self.best_recipe = None
        return self

    def _set_experiment_folder(self):
        """Sets the experiment folder and corresponding attributes in this
        class's Inventory instance based upon user settings.
        """
        if self.inventory.datetime_naming:
            subfolder = ('experiment_'
                         + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        else:
            subfolder = 'experiment'
        self.inventory.experiment = self.inventory.create_folder(
                folder = self.inventory.results, subfolder = subfolder)
        return self

    def _set_recipe_folder(self, recipe):
        """Creates file or folder path for plan-specific exports.

        Parameters:
            plan: an instance of Plan or Recipe for which files are to be
                saved.
            steps to use: a list of strings or single string containing names
                of steps from which the folder name should be created.
        """
        if hasattr(self, 'naming_classes') and self.naming_classes:
            subfolder = 'recipe_'
            for step in listify(self.naming_classes):
                subfolder += getattr(recipe, step).technique + '_'
            subfolder += str(recipe.number)
            self.inventory.recipe = self.inventory.create_folder(
                folder = self.inventory.experiment, subfolder = subfolder)
        return self

    def add_cleave(self, cleave_group, prefixes = None, columns = None):
        """Adds cleaves to the list of cleaves.

        Parameters:
            cleave_group: string naming the set of features in the group.
            prefixes: list or string of prefixes for columns to be included
                within the cleave.
            columns: list or string of columns to be included within the
                cleave."""
        if not hasattr(self.cleaves) or not self.cleaves:
            self.cleaves = []
        columns = self.ingredients.create_column_list(prefixes = prefixes,
                                                      columns = columns)
        Cleave.add(cleave_group = cleave_group, columns = columns)
        self.cleaves.append(cleave_group)
        return self

    def add_recipe(self, recipe):
        """Adds a single recipe to self.recipes.

        Parameters:
            recipe: an instance of Recipe.
        """
        if hasattr(self, 'recipes'):
            self.recipes.append(recipe)
        else:
            self.recipes = [recipe]
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc and adds it to self.recipes.

        Parameters:
            file_path: a path where the file to be loaded is located.
        """
        self.add_recipe(recipe = self.inventory.load(file_path = file_path,
                                                     file_format = 'pickle'))
        return self

    def prepare(self):
        """Creates a planner with all possible selected permutations of
        methods. Each set of methods is stored in a list of instances of the
        class stored in self.recipes.
        """
        Model.search_parameters = self.menu['search_parameters']
        # Unlike Almanac, Cookbook doesn't require state changes at each step.
        self.conform(step = 'cook')
        self._set_experiment_folder()
        self._prepare_plan_class()
        self._prepare_steps()
        self._prepare_recipes()
        self.critic = Critic(menu = self.menu, inventory = self.inventory)
        # Using training, test, validate sets creates two separate loops
        # through all recipes: one with the test set, one with the validation
        # set.
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
            for technique in self.best_recipe.techniques:
                print(technique.capitalize(), ':',
                      getattr(self.best_recipe, technique).technique)
        return

    def save_all_recipes(self):
        """Saves all recipes in self.recipes to disc as individual files."""
        for recipe in self.recipes:
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            self.save_recipe(recipe = recipe,
                             folder = self.inventory.recipe,
                             file_name = file_name,
                             file_format = 'pickle')
        return

    def save_best_recipe(self):
        """Saves the best recipe to disc."""
        if hasattr(self, 'best_recipe'):
            self.inventory.save(variable = self.best_recipe,
                                folder = self.inventory.experiment,
                                file_name = 'best_recipe',
                                file_format = 'pickle')
        return

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.save_review()
        self.save_best_recipe()
        self.ingredients.save_dropped()
        return

    def save_recipe(self, recipe, file_path = None):
        """Exports a recipe to disc.

        Parameters:
            recipe: an instance of Recipe.
            file_path: path of where file should be saved. If none, a default
                file_path will be created from self.inventory."""
        if self.verbose:
            print('Saving recipe', recipe.number)
        self._set_recipe_folder(recipe = recipe)
        self.save_plan(plan = recipe,
                       file_path = file_path)
        self.inventory.save(variable = recipe,
                            file_path = file_path,
                            folder = self.inventory.recipe,
                            file_name = 'recipe',
                            file_format = 'pickle')
        return

    def save_review(self, review = None):
        """Exports the Critic review to disc.

        Parameters:
            review: the attribute review from an instance of Critic. If none
                is provided, self.critic.review is saved.
        """
        if not review:
            review = getattr(self.critic.review,
                             self.model_type + '_report')
        self.inventory.save(variable = review,
                            folder = self.inventory.recipe,
                            file_name = self.model_type + '_report',
                            file_format = 'csv')
        return

    def start(self, ingredients = None):
        """Completes an iteration of a Cookbook.

        Parameters:
            ingredients: an Instance of Ingredients. If passsed, it will be
                assigned to self.ingredients. If not passed, self.ingredients
                will be used.
        """
        if ingredients:
            self.ingredients = ingredients
        for recipe in self.recipes:
            if self.verbose:
                print('Testing ' + recipe.name + ' ' + str(recipe.number))
            recipe.start(ingredients = self.ingredients)
            self.save_recipe(recipe = recipe)
            self.critic.start(recipe = recipe)
            self._check_best(recipe = recipe)
            self.save_review()
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
        return self