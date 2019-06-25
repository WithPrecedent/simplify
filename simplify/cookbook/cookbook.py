"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
from itertools import product
import warnings

from .countertop import Countertop
from .recipe import Recipe
from .steps import Cleave
from .steps import Custom
from .steps import Encode
from .steps import Mix
from .steps import Model
from .steps import Reduce
from .steps import Sample
from .steps import Scale
from .steps import Split
from ..critic import Presentation
from ..critic import Review


@dataclass
class Cookbook(Countertop):
    """Dynamically creates recipes for preprocessing, machine learning, and
        data analysis using a unified interface and architecture.

    Attributes:
        ingredients: an instance of Ingredients.
        menu: an instance of Menu.
        inventory: an instance of Inventory. If one is not passed when Cookbook
            is instanced, one will be created with default options.
        recipes: a list of instances of Recipe which Cookbook creates through
            the prepare method and applies through the create method.
        auto_prepare: sets whether to automatically call the prepare method
            when the class is instanced. If you do not plan to make any
            adjustments to the steps, techniques, or algorithms beyond the
            menu, this option should be set to True. If you plan to make such
            changes, prepare should be called when those changes are complete.
    """
    ingredients : object
    menu : object
    inventory : object = None
    recipes : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        """Sets up the core attributes of Cookbook."""
        # Local attributes are added from the Menu instance.
        self.menu.localize(instance = self, sections = ['general', 'recipes'])
        # Declares possible classes and steps in a cookbook recipe.
        self.steps = {'scaler' : Scale,
                      'splitter' : Split,
                      'encoder' : Encode,
                      'mixer' : Mix,
                      'cleaver' : Cleave,
                      'sampler' : Sample,
                      'reducer' : Reduce,
                      'model' : Model,
                      'custom1' : Custom,
                      'custom2' : Custom,
                      'custom3' : Custom,
                      'custom4' : Custom,
                      'custom5' : Custom,}
        # Calls method to set various default or user options.
        self._set_defaults()
        if self.auto_prepare:
            self.prepare()
        return self

    def _check_best(self, recipe):
        """Checks if the current Recipe is better than the current best Recipe
        based upon key_metric.
        """
        if not self.best_recipe:
            self.best_recipe = recipe
            self.best_recipe_score = self.review.report.loc[
                    self.review.report.index[-1], self.key_metric]
        elif (self.review.report.loc[self.review.report.index[-1],
                                    self.key_metric] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.review.report.loc[
                    self.review.report.index[-1], self.key_metric]
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                  ((self.ingredients.y == 1).sum())) - 1
        return self

    def _prepare_steps(self):
        """Initializes the step classes for use by the Cookbook."""
        for step, class_name in self.steps.items():
            setattr(self, step, self._listify(getattr(self, step)))
            if step != 'models':
                setattr(self, step + '_parameters',
                        self.menu[step + '_parameters'])
        return self

    def _set_critic(self):
        # Instances a Review class for storing review of each Recipe.create.
        self.review = Review()
        # Initializations graphing and other data visualizations.
        self.presentation = Presentation(inventory = self.inventory)
        return self

    def _set_defaults(self):
        """Sets default attributes depending upon arguments passed when the
        Cookbook is instanced.
        """
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Adds a Inventory instance with default menu if one is not passed when
        # the Cookbook class is instanced.
#        if not self.inventory:
#            self.inventory = Inventory(menu = self.menu)
        # Creates lists for custom step classes. A Recipe can have up to 5.
        self.custom1 = ['none']
        self.custom2 = ['none']
        self.custom3 = ['none']
        self.custom4 = ['none']
        self.custom5 = ['none']
        # Sets key scoring metric for methods that require a single scoring
        # metric.
        self.key_metric = self._listify(self.metrics)[0]
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        if self.compute_hyperparameters:
            self.ingredients.split_xy(label = self.label)
            self._compute_hyperparameters()
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

    def add_custom_step(self, name, techniques, parameters,
                        runtime_parameters = None, data_to_use = 'train'):
        custom = Custom(name = name,
                        techniques = techniques,
                        parameters = parameters,
                        runtime_parameters = runtime_parameters,
                        data_to_use = data_to_use)
        custom_name = 'custom' + str(len(self.custom_steps) + 1)
        setattr(self, custom_name, list(self.techniques.keys()))
        self.steps.update({custom_name : custom})
        self.custom_steps.append(name)
        return self

    def add_parameters(self, step, parameters):
        """Adds parameters to recipe step."""
        self.steps[step].add_parameters(parameters)
        return self

    def add_recipe(self):

        return self

    def add_runtime_parameters(self, step, runtime_parameters):
        """Adds parameters to recipe step."""
        self.steps[step].add_runtime_parameters(runtime_parameters)
        return self

    def add_techniques(self, step, techniques, algorithms):
        """Adds techniques and algorithms to recipe step."""
        self.steps[step].add_techniques(techniques, algorithms)
        return self

    def create(self):
        """Iterates through each of the possible recipes. The best overall
        recipe is stored in self.best_recipe.
        """
        if self.verbose:
            print('Testing recipes')
        # Calls methods to set critic options.
        self._set_critic()
        self.best_recipe = None
        if self.data_to_use == 'train_test_val':
            self.create_recipes(data_to_use = 'train_test')
            self.create_recipes(data_to_use = 'train_val')
        else:
            self.create_recipes(data_to_use = self.data_to_use)
        return self

    def create_recipes(self, recipes = None, data_to_use = 'train_test'):
        """Completes one iteration of a Cookbook, storing the review in the
        review report dataframe. Plots and the recipe are exported to the
        recipe folder.
        """
        if not recipes:
            recipes = self.recipes
        for recipe in recipes:
            if self.verbose:
                print('Testing recipe ' + str(recipe.number))
            self.inventory._set_recipe_path(recipe)
            self.ingredients.split_xy(label = self.label)
            recipe.create(ingredients = self.ingredients,
                          data_to_use = data_to_use)
            self.review.evaluate(recipe)
            self.presentation.create(recipe = recipe, review = self.review)
            self._check_best(recipe)
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            if self.export_all_recipes:
                recipe_path = self.inventory.create_path(
                        folder = self.inventory.recipe,
                        file_name = file_name,
                        file_type = 'pickle')
                self.save_recipe(recipe = recipe, file_path = recipe_path)
            cr_path = self.inventory.create_path(
                    folder = self.inventory.recipe,
                    file_name = 'class_report',
                    file_type = 'csv')
            self.inventory.save_df(self.review.class_report_df,
                                   file_path = cr_path)
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
        return self

    def load_recipe(self, file_path):
        """Imports a single recipe from disc."""
        recipe = self.inventory.unpickle_object(file_path)
        return recipe

    def save_recipe(self, recipe, file_path):
        """Exports a recipe to disc."""
        self.inventory.pickle_object(recipe, file_path)
        return self

    def new_order(self, order_list):
        self.order = order_list
        return self

    def prepare(self):
        """Creates the cookbook with all possible selected preprocessing,
        modeling, and testing methods. Each set of methods is stored in a list
        of instances of the Recipe class (self.recipes).
        """
        if self.verbose:
            print('Creating preprocessing, modeling, and testing recipes')
        self._prepare_steps()
        self.recipes = []
        all_perms = product(self.scaler, self.splitter, self.encoder,
                            self.mixer, self.cleaver, self.sampler,
                            self.reducer, self.model, self.custom1,
                            self.custom2, self.custom3, self.custom4,
                            self.custom5)
        for i, (scale, split, encode, mix, cleave, sample,
                reduce, model, custom1, custom2, custom3, custom4,
                custom5) in enumerate(all_perms):
            recipe = Recipe(number = i + 1,
                            order = self.order,
                            scaler = Scale(scale),
                            splitter = Split(split),
                            encoder = Encode(encode),
                            mixer = Mix(mix),
                            cleaver = Cleave(cleave),
                            sampler = Sample(sample),
                            reducer = Reduce(reduce),
                            model = Model(model),
                            custom1 = Custom(custom1),
                            custom2 = Custom(custom2),
                            custom3 = Custom(custom3),
                            custom4 = Custom(custom4),
                            custom5 = Custom(custom5))
            self.recipes.append(recipe)
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.key_metric, 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            print('Scaler:', self.best_recipe.scaler.technique)
            print('Splitter:', self.best_recipe.splitter.technique)
            print('Encoder:', self.best_recipe.encoder.technique)
            print('Mixer:', self.best_recipe.mixer.technique)
            print('Cleaver:', self.best_recipe.cleaver.technique)
            print('Sampler:', self.best_recipe.sampler.technique)
            print('Reducer:', self.best_recipe.reducer.technique)
            print('Custom:', self.best_recipe.custom1.technique)
            print('Model:', self.best_recipe.model.technique)
        return

    def save(self):
        """Exports the list of recipes to disc as one object."""
        cookbook_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'cookbook.pkl')
        self.inventory.pickle_object(self.recipes, file_path = cookbook_path)
        return self

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.save_review()
        self.ingredients.save_drops()
        if hasattr(self, 'best_recipe'):
            best_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'best_recipe.pkl')
            self.inventory.pickle_object(self.best_recipe, best_path)
        return self

    def save_review(self):
        review_path = self.inventory.create_path(
                folder = self.inventory.experiment,
                file_name = 'review.csv')
        self.inventory.save_df(self.review.report, review_path)
        return self