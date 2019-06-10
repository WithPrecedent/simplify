"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
import datetime
from itertools import product
import os
import pickle
import warnings

from .tools.settings import Settings
from .tools.filer import Filer
from .ingredients.custom import Custom
from .ingredients.encoder import Encoder
from .ingredients.evaluator import Evaluator
from .ingredients.ingredient import Ingredient
from .ingredients.interactor import Interactor
from .ingredients.model import Model
from .ingredients.plotter import Plotter
from .ingredients.sampler import Sampler
from .ingredients.scaler import Scaler
from .ingredients.selector import Selector
from .ingredients.splicer import Splicer
from .ingredients.splitter import Splitter
from .recipe import Recipe
from .results import Results


@dataclass
class Cookbook(object):
    """Creates dynamic recipes for preprocessing, machine learning, and data
    analysis using a unified interface and architecture.

    Attributes:
        codex: an instance of Codex.
        settings: an instance of Settings or a string containing the file
            path for a file containing the settings needed for a Settings
            instance to be created.
        filer: an instance of Filer.
        recipes: a list of instances of Recipe which Cookbook creates through
            the create method and applies through the bake method.
        new_techniques: a nested dictionary of techniques and matching
            algorithms if the user wants to provide more algorithms to the
            Cookbook when instanced. Alternatively, after the class is
            instanced, the user can use the include method to add algorithms
            to any of the recipe ingredients.
        best_recipe: the best recipe tested based upon the key metric set in
            the instance of Settings (the first metric in metrics)
    """
    codex : object
    settings : object
    filer : object = None
    recipes : object = None
    new_techniques : object = None
    best_recipe : object = None

    def __post_init__(self):
        """Sets up the core attributes of Cookbook."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Loads settings from an .ini file if not passed when class is
        # instanced.
        if self.settings:
            if isinstance(self.settings, str):
                self.settings = Settings(file_path = self.settings)
        else:
            error = 'Settings or string containing settings path needed.'
            raise AttributeError(error)
        # Local attributes are added from the Settings instance.
        self.settings.localize(instance = self,
                               sections = ['general', 'files', 'recipes'])

        # Adds a Filer instance with default settings if one is not passed when
        # the Cookbook class is instanced.
        if not self.filer:
            self.filer = Filer(settings = self.settings)
        # Injects dependencies with appropriate attributes.
        self._inject()
        # Instances a Results class for storing results of each Recipe.bake.
        self.results = Results()
        # Sets key scoring metric for methods that require a single scoring
        # metric.
        self.key_metric = self._listify(self.metrics)[0]
        # Creates empty lists and dictionary for custom methods and parameters
        # to be added by user.
        self.customs = []
        self.customs_params = {}
        self.customs_runtime_params = {}
        # Declares possible classes and ingredients in a cookbook recipe.
        self.ingredients = {'scalers' : Scaler,
                            'splitter' : Splitter,
                            'encoders' : Encoder,
                            'interactors' : Interactor,
                            'splicers' : Splicer,
                            'samplers' : Sampler,
                            'selectors' : Selector,
                            'customs' : Custom,
                            'models' : Model,
                            'evaluator' : Evaluator,
                            'plotter' : Plotter}
        # Adds any new techniques passed in Cookbook instance.
        if self.new_techniques:
            for ingredient, nested_dict in self.new_techniques.items():
                for key, value in nested_dict.items():
                    self.ingredients[ingredient].options.update({key, value})
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        if self.compute_hyperparameters:
            self.codex.split_xy(label = self.label)
            self._compute_hyperparameters()
        return self

    def __delitem__(self, value):
        """Deletes techniques by passing [ingredients, techniques]."""
        ingredients, techniques = value
        del_ingredients = zip(self._listify(ingredients),
                              self._listify(techniques))
        for ingredient, technique in del_ingredients.items():
            if technique in self.ingredients[ingredient].options:
                self.ingredients[ingredient].options.pop(technique)
            else:
                error_message = (
                        technique + ' is not in ' + ingredient + ' method')
                raise KeyError(error_message)
        return self

    def __getitem__(self, value):
        """Gets particular techniques by passing [ingredient, technique]."""
        ingredient, technique = value
        if ingredient in self.ingredients:
            return self.ingredients[ingredient].options[technique]
        else:
            error_message = ingredient + ' or ' + technique + ' not found'
            raise KeyError(error_message)
            return

    def __setitem__(self, value):
        """Sets new techniques by passing either strings or lists of strings
        containing the ingredients, techniques, algorithms in the form of
        [ingredients, techniques, algorithms].
        """
        ingredients, techniques, algorithms = value
        set_ingredients = zip(self._listify(ingredients),
                              self._listify(techniques),
                              self._listify(algorithms))
        for ingredient, technique, algorithm in set_ingredients.items():
            self.ingredients[ingredient].options.update(
                    {technique : algorithm})
        return self

    def _check_best(self, recipe):
        """Checks if the current Recipe is better than the current best Recipe.
        """
        if not self.best_recipe:
            self.best_recipe = recipe
            self.best_recipe_score = self.results.table.loc[
                    self.results.table.index[-1], self.key_metric]
        elif (self.results.table.loc[self.results.table.index[-1],
                                    self.key_metric] > self.best_recipe_score):
            self.best_recipe = recipe
            self.best_recipe_score = self.results.table.loc[
                    self.results.table.index[-1], self.key_metric]
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        Model.scale_pos_weight = (len(self.codex.y.index) /
                                  ((self.codex.y == 1).sum())) - 1
        return self

    def _inject(self):
        """Injects filer, settings, and _listify method into Ingredient class,
        Recipe class, and/or data instance.
        """
        Ingredient.filer = self.filer
        Ingredient.settings = self.settings
#        Ingredient._listify = self._listify
        Recipe.filer = self.filer
        Recipe.settings = self.settings
        if not self.codex.filer:
            self.codex.filer = self.filer
        return self

    def _listify(self, variable):
        """Checks to see if the methods are stored in a list. If not, the
        methods are converted to a list or a list of 'none' is created.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def _prepare_ingredients(self):
        """Initializes the ingredient classes for use by the Cookbook."""
        for ingredient in self.ingredients:
            if ingredient in ['evaluator', 'explainer', 'plotter']:
                setattr(self, ingredient, getattr(self, ingredient))
            else:
                setattr(self, ingredient,
                        self._listify(getattr(self, ingredient)))
            if not ingredient in ['models']:
                param_var = ingredient + '_params'
                setattr(self, param_var, self.settings[param_var])
        Evaluator.options = self.results.options
        Evaluator.columns = self.results.columns
        Evaluator.prob_options = self.results.prob_options
        Evaluator.score_options = self.results.score_options
        Evaluator.spec_metrics = self.results.spec_metrics
        Evaluator.neg_metrics = self.results.neg_metrics
        return self

    def _set_folders(self):
        """Sets and creates folder paths for recipe results to be stored."""
        if self.recipe_folder in ['dynamic']:
            subfolder = ('recipe_'
                         + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        else:
            subfolder = self.recipe_folder
        self.filer.results = os.path.join(self.filer.results, subfolder)
        self.filer.recipes = self.filer.make_path(folder = self.filer.results,
                                                  subfolder = 'recipes')
        self.filer._make_folder(self.filer.results)
        self.filer._make_folder(self.filer.recipes)
        return self

    def add_technique(self, ingredients, techniques, algorithms):
        """Adds techniques and algorithms to recipe ingredients."""
        new_techniques = zip(self._listify(ingredients),
                             self._listify(techniques),
                             self._listify(algorithms))
        for ingredient, technique, algorithm in new_techniques.items():
            self.recipe_classes[ingredient].add(technique, algorithm)
        return self

    def add_splice(self, splice_label, prefixes = [], columns = []):
        """Adds splices to the list of splicers."""
        self.splicers.add_splice(splice_label = splice_label,
                                 prefixes = prefixes,
                                 columns = columns)
        self.splicers.append(splice_label)
        return self

    def bake(self):
        """Iterates through each of the possible recipes. The best overall
        recipe is stored in self.best_recipe.
        """
        if self.verbose:
            print('Testing recipes')
        self._set_folders()
        self.best_recipe = None
        if self.data_to_use == 'train_test_val':
            self.bake_cookbook(data_to_use = 'train_test')
            self.bake_cookbook(data_to_use = 'train_val')
        else:
            self.bake_cookbook(data_to_use = self.data_to_use)
        return self

    def bake_cookbook(self, data_to_use = 'train_test'):
        """Completes one iteration of a Cookbook, storing the results in the
        results table dataframe. Plots and the recipe are exported to the
        recipe folder.
        """
        for recipe in self.recipes:
            if self.verbose:
                print('Testing recipe ' + str(recipe.number))
            self.codex.split_xy(label = self.label)
            recipe.bake(codex = self.codex,
                        data_to_use = self.data_to_use)
            self.results.table.loc[len(self.results.table)] = (
                    recipe.evaluator.result)
            self._check_best(recipe)
            file_name = 'recipe' + str(recipe.number) + '_' + recipe.model.technique
            if self.export_all_recipes:
                recipe_path = self.filer._iter_path(
                        model = recipe.model,
                        recipe_number = recipe.number,
                        splicer = recipe.splicer,
                        file_name = file_name,
                        file_type = 'pickle')
                recipe.save(recipe, export_path = recipe_path)
            cr_path = self.filer._iter_path(model = recipe.model,
                                            recipe_number = recipe.number,
                                            splicer = recipe.splicer,
                                            file_name = 'class_report',
                                            file_type = 'csv')
            recipe.evaluator.save_classification_report(export_path = cr_path)
            # To conserve memory, each recipe is deleted after being exported.
            del(recipe)
        return self

    def load(self, import_path = None, return_cookbook = False):
        """Imports a single pickled cookbook from disc."""
        if not import_path:
            import_path = self.filer.import_folder
        recipes = pickle.load(open(import_path, 'rb'))
        if return_cookbook:
            return recipes
        else:
            self.recipes = recipes
            return self

    def prepare(self):
        """Creates the cookbook with all possible selected preprocessing,
        modeling, and testing methods. Each set of methods is stored in a list
        of instances of the Recipe class (self.recipes).
        """
        if self.verbose:
            print('Creating preprocessing, modeling, and testing recipes')
        self._prepare_ingredients()
        self.recipes = []
        all_ingredients = product(self.scalers, self.splitter, self.encoders,
                            self.interactors, self.splicers, self.samplers,
                            self.customs, self.selectors, self.models)
        for i, (scaler, splitter, encoder, interactor, splicer, sampler,
                custom, selector, model) in enumerate(all_ingredients):
            recipe = Recipe(number = i + 1,
                            order = self.order,
                            scaler = Scaler(scaler,
                                            self.scalers_params),
                            splitter = Splitter(splitter,
                                                self.splitter_params),
                            encoder = Encoder(encoder,
                                              self.encoders_params),
                            interactor = Interactor(interactor,
                                                    self.interactors_params),
                            splicer = Splicer(splicer,
                                              self.splicers_params),
                            sampler = Sampler(sampler,
                                              self.samplers_params),
                            custom = Custom(custom,
                                            self.customs_params),
                            selector = Selector(selector,
                                                self.selectors_params),
                            model = Model(model,
                                          self.settings[model + '_params']),
                            evaluator = Evaluator(self.evaluator,
                                                  self.evaluator_params),
                            plotter = Plotter(self.plotter,
                                              self.plotter_params),
                            settings = self.settings,
                            filer = self.filer)
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
            print('Interactor:', self.best_recipe.interactor.technique)
            print('Splicer:', self.best_recipe.splicer.technique)
            print('Sampler:', self.best_recipe.sampler.technique)
            print('Selector:', self.best_recipe.selector.technique)
            print('Custom:', self.best_recipe.custom.technique)
            print('Model:', self.best_recipe.model.technique)
        return

    def save(self, export_path = None):
        """Exports a cookbook to disc."""
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(self.recipes, open(export_path, 'wb'))
        return self

    def save_everything(self):
        """Automatically saves the cookbook, scores, and best recipe."""
        self.save(export_path = os.path.join(self.filer.results,
                                             'cookbook.pkl'))
        self.results.save(export_path = os.path.join(self.filer.results,
                                                     'results_table.csv'))
        self.codex.save_drops()
        if self.best_recipe:
            self.best_recipe.save(recipe = self.best_recipe,
                                  export_path = os.path.join(
                                          self.filer.results,
                                          'best_recipe.pkl'))
        return self