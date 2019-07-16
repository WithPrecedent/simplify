"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass

from .recipe import Recipe
from .steps import Cleave, Encode, Mix, Model, Reduce, Sample, Scale, Split
from ..critic import Critic
from ..implements import listify
from ..managers import Planner, Step


@dataclass
class Cookbook(Planner):
    """Dynamically creates recipes for final preprocessing, machine learning,
    and data analysis using a unified interface and architecture.

    Attributes:
        an instance of Menu or a string containing the path where a menu
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
    def critic(self):
        return self.comparer

    @critic.setter
    def critic(self, comparer):
        self.comparer = comparer

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


    def _set_defaults(self):
        """ Declares default step names and classes in a Cookbook recipe."""
        self.options = {'scaler' : Scale,
                        'splitter' : Split,
                        'encoder' : Encode,
                        'mixer' : Mix,
                        'cleaver' : Cleave,
                        'sampler' : Sample,
                        'reducer' : Reduce,
                        'model' : Model}
        self.plan_class = Recipe
        self.compare_class = Critic
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

    def add_parameters(self, step, parameters):
        """Adds parameter sets to the parameters dictionary of a prescribed
        step. """
        self.steps[step].add_parameters(parameters = parameters)
        return self

    def add_recipe(self, recipe):
        if hasattr(self, 'recipes'):
            self.recipes.append(recipe)
        else:
            self.recipes = [recipe]
        return self

    def add_runtime_parameters(self, step, parameters):
        """Adds runtime_parameter sets to the parameters dictionary of a
        prescribed step."""
        self.steps[step].add_runtime_parameters(parameters = parameters)
        return self

    def add_step(self, name, techniques, step_order = None, **kwargs):
        self.steps.update({name : Step(name = name,
                                       techniques = techniques,
                                       **kwargs)})
        if step_order:
            self.steps = step_order
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc."""
        recipe = self.inventory.unpickle_object(file_path)
        self.add_recipe(recipe = recipe)
        return self

    def prepare(self):
        Model.search_parameters = self.menu['search_parameters']
        super().prepare()
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

    def save_review(self):
        review_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'review.csv')
        self.inventory.save_df(self.critic.review.report,
                               file_path = review_path)
        return self