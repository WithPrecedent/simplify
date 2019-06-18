"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass

from itertools import product
import warnings

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
#from ..menu import Menu
#from ..implements.implement import Implement
from ..inventory import Inventory
from ..critic import Presentation
from ..critic import Review


@dataclass
class Cookbook(object):
    """Dynamically creates recipes for preprocessing, machine learning, and
        data analysis using a unified interface and architecture.

    Attributes:
        ingredients: an instance of Ingredients.
        menu: an instance of Menu or a string containing the file path for a
            file containing the information needed for a Menu instance to be
            created.
        inventory: an instance of Inventory. If one is not passed when Cookbook
            is instanced, one will be created with default options.
        recipes: a list of instances of Recipe which Cookbook creates through
            the prepare method and applies through the create method.
    """
    ingredients : object
    menu : object
    inventory : object = None
    recipes : object = None

    def __post_init__(self):
        """Sets up the core attributes of Cookbook."""
        # Local attributes are added from the Menu instance.
        self.menu.localize(instance = self, sections = ['general', 'recipes'])
        # Declares possible classes and steps in a cookbook recipe.
        self.steps = {'scalers' : Scale,
                      'splitter' : Split,
                      'encoders' : Encode,
                      'mixers' : Mix,
                      'cleavers' : Cleave,
                      'samplers' : Sample,
                      'reducers' : Reduce,
                      'models' : Model}
        # Calls method to set various default or user options.
        self._set_defaults()
        # Calls methods to set critic options.
        self._set_critic()
        return self

#    def __delitem__(self, value):
#        """Deletes techniques by passing [steps, techniques]."""
#        steps, techniques = value
#        del_techniques = zip(self._listify(steps), self._listify(techniques))
#        for step, technique in del_techniques.items():
#            if technique in self.steps[step].options:
#                self.steps[step].options.pop(technique)
#            else:
#                error = technique + ' is not in ' + step
#                raise KeyError(error)
#        return self
#
#    def __getitem__(self, value):
#        """Gets particular techniques by passing [step, technique]."""
#        step, technique = value
#        if step in self.steps:
#            return self.steps[step].options[technique]
#        else:
#            error = step + ' or ' + technique + ' not found'
#            raise KeyError(error)
#            return
#
#    def __setitem__(self, value):
#        """Sets new techniques by passing either strings or lists of strings
#        containing the steps, techniques, algorithms in the form of
#        [steps, techniques, algorithms].
#        """
#        steps, techniques, algorithms = value
#        set_techniques = zip(self._listify(steps),
#                             self._listify(techniques),
#                             self._listify(algorithms))
#        for step, technique, algorithm in set_techniques.items():
#            self.steps[step].options.update({technique : algorithm})
#        return self

    def _check_best(self, recipe):
        """Checks if the current Recipe is better than the current best Recipe
        based upon key_metric.
        """
        if not self.best_recipe:
            self.best_recipe = recipe
            self.best_recipe_score = self.review.table.loc[
                    self.review.table.index[-1], self.key_metric]
        elif (self.review.table.loc[self.review.table.index[-1],
                                    self.key_metric] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.review.table.loc[
                    self.review.table.index[-1], self.key_metric]
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        Model.scale_pos_weight = (len(self.ingredients.y.index) /
                                  ((self.ingredients.y == 1).sum())) - 1
        return self

#    def _inject(self):
#        """Injects inventory and menu methods into Implement class and/or data
#        instance.
#        """
#        Implement.inventory = self.inventory
#        Implement.menu = self.menu
#        if not self.ingredients.inventory:
#            self.ingredients.inventory = self.inventory
#        if not self.ingredients.menu:
#            self.ingredients.menu = self.menu
#        return self

    def _listify(self, variable):
        """Checks to see if variable is a list. If not, it is converted to a
        list or a list of 'none' is created.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

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
        # Sets key scoring metric for methods that require a single scoring
        # metric.
        self.key_metric = self._listify(self.metrics)[0]
        # Initializations graphing and other data visualizations.
        self.presentation = Presentation()
        return self

    def _set_defaults(self):
        """Sets default attributes depending upon arguments passed when the
        Cookbook is instanced.
        """
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Loads menu from an .ini file if not passed when class is
        # instanced.
        if self.menu:
            if isinstance(self.menu, str):
                self.menu = Menu(file_path = self.menu)
        else:
            error = 'Menu or string containing menu path needed.'
            raise AttributeError(error)
        # Adds a Inventory instance with default menu if one is not passed when
        # the Cookbook class is instanced.
        if not self.inventory:
            self.inventory = Inventory(menu = self.menu)
        # Injects dependencies with appropriate attributes.
#        self._inject()
        # Creates empty lists and dictionary for custom methods and parameters
        # to be added by user.
        self.customs = ['none']
        self.customs_parameters = {}
        self.customs_runtime_parameters = {}
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

#    def add_custom(self, name, parameters, func, runtime_parameters = None):
#        if not hasattr(self.customs) or not self.cleaves:
#            self.cleaves = []
#        setattr(self, name, Custom(technique = name,
#                                   parameters = parameters,
#                                   method = func,
#                                   runtime_parameters = runtime_parameters))
#        setattr(self, '_' + step, self._customs)
#        return self

    def add_technique(self, steps, techniques, algorithms):
        """Adds techniques and algorithms to recipe steps."""
        new_techniques = zip(self._listify(steps),
                             self._listify(techniques),
                             self._listify(algorithms))
        for step, technique, algorithm in new_techniques.items():
            self.recipe_classes[step].add(technique, algorithm)
        return self

    def create(self):
        """Iterates through each of the possible recipes. The best overall
        recipe is stored in self.best_recipe.
        """
        if self.verbose:
            print('Testing recipes')
        self.best_recipe = None
        if self.data_to_use == 'train_test_val':
            self.create_recipes(data_to_use = 'train_test')
            self.create_recipes(data_to_use = 'train_val')
        else:
            self.create_recipes(data_to_use = self.data_to_use)
        return self

    def create_recipes(self, recipes = None, data_to_use = 'train_test'):
        """Completes one iteration of a Cookbook, storing the review in the
        review table dataframe. Plots and the recipe are exported to the
        recipe folder.
        """
        if not recipes:
            recipes = self.recipes
        for recipe in recipes:
            if self.verbose:
                print('Testing recipe ' + str(recipe.number))
            self.ingredients.split_xy(label = self.label)
            recipe.create(ingredients = self.ingredients,
                          data_to_use = self.data_to_use)
            self.review.table.loc[len(self.review.table)] = (
                    recipe)
            self._check_best(recipe)
            file_name = (
                'recipe' + str(recipe.number) + '_' + recipe.model.technique)
            if self.export_all_recipes:
                recipe_path = self.inventory._iter_path(
                        model = recipe.model,
                        recipe_number = recipe.number,
                        cleave = recipe.cleave,
                        file_name = file_name,
                        file_type = 'pickle')
                recipe.save(recipe, export_path = recipe_path)
            cr_path = self.inventory._iter_path(model = recipe.model,
                                            recipe_number = recipe.number,
                                            cleave = recipe.cleave,
                                            file_name = 'class_report',
                                            file_type = 'csv')
            recipe.evaluator.save_classification_report(export_path = cr_path)
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
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
        print(self.customs)
        all_steps = product(self.scalers, self.splitter, self.encoders,
                            self.mixers, self.cleavers, self.samplers,
                            self.customs, self.reducers, self.models)
        for i, (scale, split, encode, mix, cleave, sample, custom, reduce,
                model) in enumerate(all_steps):
            print(self.menu[model + '_parameters'])
            recipe = Recipe(number = i + 1,
                            order = self.order,
                            scaler = Scale(scale, self.scalers_parameters),
                            splitter = Split(split, self.splitter_parameters),
                            encoder = Encode(encode, self.encoders_parameters),
                            mixer = Mix(mix, self.mixers_parameters),
                            cleaver = Cleave(cleave, self.cleavers_parameters),
                            sampler = Sample(sample, self.samplers_parameters),
                            customs = Custom(custom, self.customs_parameters),
                            reducer = Reduce(reduce, self.reducers_parameters),
                            model = Model(model,
                                          self.menu[model + '_parameters']))
            self.recipes.append(recipe)
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.key_metric, 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            print('Scaler:', self.best_recipe.scale.technique)
            print('Splitter:', self.best_recipe.split.technique)
            print('Encoder:', self.best_recipe.encode.technique)
            print('Mixer:', self.best_recipe.mix.technique)
            print('Cleaver:', self.best_recipe.cleave.technique)
            print('Sampler:', self.best_recipe.sample.technique)
            print('Reducer:', self.best_recipe.reduce.technique)
            print('Custom:', self.best_recipe.custom.technique)
            print('Model:', self.best_recipe.model.technique)
        return

    def save(self, file_path = None):
        """Exports the list of recipes to disc as one object."""
        self.inventory.save(self.recipes,
                            folder = self.inventory.results,
                            file_path = file_path,
                            file_name = 'cookbook.pkl')
        return self

    def save_everything(self):
        """Automatically saves the recipes, results, dropped columns from
        ingredients, and the best recipe (if one has been stored)."""
        self.save()
        self.review.save()
        self.ingredients.save_drops()
        if hasattr(self, 'best_recipe'):
            self.inventory.save(self.best_recipe,
                                file_name = 'best_recipe.pkl')
        return self