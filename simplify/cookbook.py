"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and utilization.
"""
from dataclasses import dataclass
import datetime
from itertools import product
import os
from pathlib import Path
import pickle
import warnings

from .steps.custom import Custom
from .steps.encoder import Encoder
from .steps.evaluator import Evaluator
from .filer import Filer
from .steps.interactor import Interactor
from .steps.model import Model
from .steps.plotter import Plotter
from .recipe import Recipe
from .results import Results
from .steps.sampler import Sampler
from .steps.scaler import Scaler
from .steps.selector import Selector
from .settings import Settings
from .steps.splicer import Splicer
from .steps.splitter import Splitter
from .steps.step import Step


@dataclass
class Cookbook(object):
    """Class for creating dynamic recipes for preprocessing, machine learning,
    and data analysis using a unified interface and architecture.

    Attributes:
        data: an instance of Data.
        settings: an instance of Settings.
        settings_path: if settings is not passed, settings_path should be
            passed so that an instance of Settings can be loaded.
        filer: an instance of Filer.
        recipes: a list of instances of Recipe which Cookbook creates through
            the create method and applies through the bake method.
        data_folder: the path for the folder where source data is stored.
        results_folder: the path where results of the analysis should be
            stored.
        splicers: list of groups of predictors for testing in comparison to
            each other.
        new_algorithms: a nested dictionary of ingredients and matching
            algorithms if the user wants to provide more algorithms to the
            cookbook when instanced. Alternatively, after the class is
            instanced, the user can use the include method to add algorithms
            to any of the recipe steps.
        best_recipe: the best recipe tested based upon the key metric set in
            the instance of Settings (the first metric in metrics)
    """
    data : object
    settings : object = None
    settings_path : str = ''
    filer : object = None
    recipes : object = None
    data_folder : str = ''
    results_folder : str = ''
    splicers : object = None
    new_algorithms : object = None
    best_recipe : object = None

    def __post_init__(self):
        """Sets up the core attributes of cookbook."""
        # Loads settings from an .ini file if not passed when class is
        # instanced. Local attributes are added from the settings instance.
        if not self.settings:
            if not self.settings_path:
                self.settings_path = Path(os.path.join('settings_files',
                                          'simplify_settings.ini'))
            if self.settings_path.is_file():
                self.settings = Settings(file_path = self.settings_path)
            else:
                error = self.settings_path + ' does not exist'
                raise OSError(error)
        self.settings.localize(instance = self,
                               sections = ['general', 'files', 'recipes'])
        # Removes the numerous python warnings from console output if option
        # selected by user.
        if not self.warnings:
            warnings.filterwarnings('ignore')
        # Adds a Filer instance if one is not passed when the class is instanced.
        if not self.filer:
            self.filer = Filer(root = self.data_folder,
                               results = self.results_folder,
                               recipes = self.recipe_folder,
                               settings = self.settings)
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
        # Declares possible classes and steps in a cookbook recipe.
        self.recipe_steps = {'scalers' : Scaler,
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
        # Adds any new algorithms passed in Cookbook instance.
        if self.new_algorithms:
            for step, nested_dict in self.new_algorithms.items():
                for key, value in nested_dict.items():
                    self.step_classes[step].options.update({key, value})
        # Data is split in oder for certain values to be computed that require
        # features and the label to be split.
        if self.compute_hyperparameters:
            self.data.split_xy(label = self.label)
            self._compute_hyperparameters()
        return self

    def __getitem__(self, value):
        """Gets particular algorithms by passing [step, name].
        """
        step, name = value
        if step in self.step_classes:
            return self.recipe_steps[step].options[name]
        else:
            error_message = step + ' or ' + name + ' not found'
            raise KeyError(error_message)
            return

    def __setitem__(self, value):
        """Sets new algorithms by passing either strings or lists of strings
        containing the steps, names, and algorithms in the form of
        [steps, names, algorithms].
        """
        steps, names, algorithms = value
        if isinstance(steps, str) or isinstance(steps, list):
            if isinstance(names, str) or isinstance(names, list):
                if (isinstance(algorithms, object)
                    or isinstance(algorithms, list)):
                    steps = self._listify(steps)
                    names = self._listify(names)
                    steps = self._listify(steps)
                    new_algorithms = zip(steps, names, algorithms)
                    for step, name, algorithm in new_algorithms.items():
                        self.recipe_steps[step][name][algorithm]
                else:
                    error_message = (
                            name + ' must be an object of list of objects')
                    raise TypeError(error_message)
            else:
                error_message = name + ' must be a string or list of strings'
                raise TypeError(error_message)
        else:
            error_message = step + ' must be a string or list of strings'
            raise TypeError(error_message)
        return self

    def __delitem__(self, value):
        """Deletes algorithms by passing [steps, algorithm_names]."""
        steps, names = value
        steps = self._listify(steps)
        names = self._listify(names)
        del_steps = zip(steps, names)
        for step, name in del_steps.items():
            if name in self.recipe_steps[step].options:
                self.recipe_steps[step][name]
            else:
                error_message = name + ' is not in ' + step + ' method'
                raise KeyError(error_message)
        return self

    def _inject(self):
        """Injects filer, settings, and _listify method into Step class, Recipe
        class, and/or data instance.
        """
        Step.filer = self.filer
        Step.settings = self.settings
        Step._listify = self._listify
        Recipe.filer = self.filer
        Recipe.settings = self.settings
        if not self.data.filer:
            self.data.filer = self.filer
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

    def _prepare_steps(self):
        """Initializes the step classes for use by the Cookbook."""
        for step in self.recipe_steps:
            if step in ['evaluator', 'explainer', 'plotter']:
                setattr(self, step, getattr(self, step))
            else:
                setattr(self, step, self._listify(getattr(self, step)))
            if not step in ['models']:
                param_var = step + '_params'
                setattr(self, param_var, self.settings[param_var])
        Evaluator.options = self.results.options
        Evaluator.columns = self.results.columns
        Evaluator.prob_options = self.results.prob_options
        Evaluator.score_options = self.results.score_options
        Evaluator.spec_metrics = self.results.spec_metrics
        Evaluator.neg_metrics = self.results.neg_metrics
        return self

    def _compute_hyperparameters(self):
        """Computes hyperparameters that can be determined by the source data.
        """
        Model.scale_pos_weight = (len(self.data.y.index) /
                                  ((self.data.y == 1).sum())) - 1
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

    def _stringify(self, variable):
        """Checks to see if the variables is a string. If not, a string is
        taken the first item from the list.
        """
        if not variable:
            return 'none'
        elif isinstance(variable, str):
            return variable
        else:
            return variable[0]


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

    def include(self, steps, ingredients, algorithms, **kwargs):
        """Adds algorithms to recipe steps."""
        steps = self._listify(steps)
        ingredients = self._listify(ingredients)
        algorithms = self._listify(algorithms)
        new_algorithms = zip(steps, ingredients, algorithms)
        for step, ingredient, algorithm in new_algorithms.items():
            self.recipe_classes[step].include(ingredient, algorithm, **kwargs)
        return self

    def add_splice(self, splice_label, prefixes = [], columns = []):
        """Adds splices to the list of splicers."""
        self.splicers.add_splice(splice_label = splice_label,
                                 prefixes = prefixes,
                                 columns = columns)
        self.splicers.append(splice_label)
        return self

    def create(self):
        """Creates the cookbook with all possible selected preprocessing,
        modeling, and testing methods. Each set of methods is stored in a list
        of instances of the Recipe class (self.recipes).
        """
        if self.verbose:
            print('Creating preprocessing, modeling, and testing recipes')
        self._prepare_steps()
        self.recipes = []
        all_steps = product(self.scalers, self.splitter, self.encoders,
                            self.interactors, self.splicers, self.samplers,
                            self.customs, self.selectors, self.models)
        for i, (scaler, splitter, encoder, interactor, splicer, sampler,
                custom, selector, model) in enumerate(all_steps):
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
                                              self.plotter_params))
            self.recipes.append(recipe)
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
            self.data.split_xy(label = self.label)
            recipe.bake(data = self.data,
                        data_to_use = self.data_to_use,
                        runtime_params = self.customs_runtime_params)
            self.results.table.loc[len(self.results.table)] = (
                    recipe.evaluator.result)
            self._check_best(recipe)
            file_name = 'recipe' + str(recipe.number) + '_' + recipe.model.name
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

    def apply(self):
        """Implements bake method for those who prefer non-cooking method
        names.
        """
        self.bake()
        return self

    def print_best(self):
        """Prints output to the console about the best recipe."""
        if self.verbose:
            print('The best test recipe, based upon the',
                  self.key_metric, 'metric with a score of',
                  f'{self.best_recipe_score : 4.4f}', 'is:')
            print('Scaler:', self.best_recipe.scaler.name)
            print('Splitter:', self.best_recipe.splitter.name)
            print('Encoder:', self.best_recipe.encoder.name)
            print('Interactor:', self.best_recipe.interactor.name)
            print('Splicer:', self.best_recipe.splicer.name)
            print('Sampler:', self.best_recipe.sampler.name)
            print('Selector:', self.best_recipe.selector.name)
            print('Custom:', self.best_recipe.custom.name)
            print('Model:', self.best_recipe.model.name)
        return

    def save_everything(self):
        """Automatically saves the cookbook, scores, and best recipe."""
        self.save(export_path = os.path.join(self.filer.results,
                                             'cookbook.pkl'))
        self.results.save(export_path = os.path.join(self.filer.results,
                                                     'results_table.csv'))
        self.data.save_drops()
        if self.best_recipe:
            self.best_recipe.save(recipe = self.best_recipe,
                                  export_path = os.path.join(
                                          self.filer.results,
                                          'best_recipe.pkl'))
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

    def save(self, export_path = None):
        """Exports a cookbook to disc."""
        if not export_path:
            export_path = self.filer.results_folder
#        pickle.dump(self.recipes, open(export_path, 'wb'))
        return self