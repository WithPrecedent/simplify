"""
The siMpLify package allows users to create a cookbook of dynamic recipes that
mix-and-match feature engineering and modeling ingredients based upon a common,
simple interface. It then analyzes the results using selected, appropriate
metrics and exports tables, charts, and graphs compatible with the models and
data types.

siMpLify divides the feature engineering and modeling process into eleven
major steps that can be sequenced in different orders:

    Scaler: converts numerical features into a common scale, using scikit-learn
        methods.
    Splitter: divides data into train, test, and/or validation sets once or
        iteratively through k-folds cross-validation.
    Encoder: converts categorical features into numerical ones, using
        category-encoders methods.
    Interactor: converts selected categorical features into new polynomial
        features, using PolynomialEncoder from category-encoders or other
        mathmatical combinations.
    Splicer: creates different subgroups of features to allow for easy
        comparison between them.
    Sampler: synthetically resamples training data for imbalanced data,
        using imblearn methods, for use with models that struggle with
        imbalanced data.
    Selector: selects features recursively or as one-shot based upon user
        criteria, using scikit-learn methods.
    Custom: allows users to add any scikit-learn or siMpLify compatible
        method to be added into a recipe.
    Model: implements machine learning algorithms, currently includes
        xgboost and scikit-learn methods. The user can opt to either test
        different hyperparameters for the models selected or a single set
        of hyperparameters. Hyperparameter earch methods currently include
        RandomizedSearchCV, GridSearchCV, and bayesian optimization through
        skopt.
    Evaluator: tests the models using user-selected or default metrics and
        explainers from sklearn, shap, eli5, and lime.
    Plotter: produces helpful graphical representations based upon the model
        selected and evaluator and explainers used, utilizing seaborn and
        matplotlib methods.

Together, these steps form a recipe. Each recipe will be tested iteratively
using Cookbook.bake (or Cookbook.apply if preferred) or individually using
Recipe.bake (or Recipe.apply if preferred). If users choose to apply any of the
three steps in the recipe, results will be exported automatically.

siMpLify contains the following accessible classes:
    Cookbook: containing the methods needed to create dynamic recipes and
        stores them in Cookbook.recipes. For that reason, the Recipe class does
        not ordinarily need to be instanced directly.
    Recipe: if the user wants to manually create a single recipe, the Recipe
        class is made public for this purpose.
    Data: includes methods for creating and modifying pandas dataframes used
        by Cookbook. As the data is split into features, labels, test, train,
        etc. dataframes, they are all created as attributes to an instance of
        the Data class.
    Scaler, Splitter, Encoder, Interactor, Splicer, Sampler, Selector, and
        Custom: contain the different ingredient options for each step in a
        recipe.
    Model: contains different machine learning algorithms divided into three
        major model_type: classifier, regressor, and unsupervised.
    Results: contains the metrics used by Evaluator and stores a dataframe
        (Results.table) applying those metrics. Each row of the table stores
        each of the steps used, the folder in which the relevant files are
        stored, and all of the metrics used for that recipe.
    Evaluator: applies user-selected or default metrics for each recipe and
        passes those results for storage in Results.table.
    Plotter: prepares and exports plots and other visualizations based upon
        the model type and Evaluator and Estimator methods.
    Filer: creates and contains the path structure for loading data and
        settings as well as saving results, data, and plots.
    Library: provides methods for importing and parsing recipes, results,
        cookbooks, and ingredients for reuse.
    Settings: contains the methods for parsing the settings file to create
        a nested dictionary used by the other classes.

If the user opts to use the settings.ini file, the only classes that absolutely
need to be used are Cookbook and Data. Nonetheless, the rest of the classes and
attributes are still available for use. All of the ten step classes are stored
in a list of recipes (Cookbook.recipes). Cookbook.evaluator collects the
results from the recipes being tested.

If a Filer instance is not passed to Cookbook when it instanced, an
import_folder and export_folder must be passed. Then the Cookbook will create
an instance of Filer as an attribute of the Cookbook (Cookbook.filer).
If an instance of Settings is not passed when the Cookbook is instanced, a
settings file will be loaded automatically.
"""

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

from custom import Custom
from encoder import Encoder
from evaluator import Evaluator
from interactor import Interactor
from model import Model
from plotter import Plotter
from recipe import Recipe
from results import Results
from sampler import Sampler
from scaler import Scaler
from selector import Selector
from simple_filer.filer import Filer
from simple_settings.settings import Settings
from splicer import Splicer
from splitter import Splitter
from step import Step


@dataclass
class Cookbook(object):
    """
    Class for creating dynamic recipes for preprocessing, machine learning,
    and data analysis using a unified interface and architecture.
    """
    data : object
    settings : object = None
    settings_path : str = ''
    recipes : object = None
    tester : object = None
    filer : object = None
    data_folder : str = ''
    results_folder : str = ''
    splicers : object = None
    new_algorithms : object = None
    best_recipe : object = None

    def __post_init__(self):
        """
        Loads settings from an .ini file if not passed when class is instanced.
        Local attributes are added from the settings instance.
        """
        if not self.settings:
            if not self.settings_path:
                self.settings_path = Path(os.path.join('..', 'settings',
                                          'simplify_settings.ini'))
            if self.settings_path.is_file():
                self.settings = Settings(file_path = self.settings_path)
            else:
                error = self.settings_path + ' does not exist'
                raise OSError(error)
        self.settings.localize(instance = self,
                               sections = ['general', 'files', 'recipes'])
        """
        Removes the numerous python warnings from console output if option
        selected by user.
        """
        if not self.warnings:
            warnings.filterwarnings('ignore')
        """
        Adds a Filer instance if one is not passed when the class is instanced.
        """
        if not self.filer:
            self.filer = Filer(root = self.data_folder,
                               results = self.results_folder,
                               recipes = self.recipe_folder,
                               settings = self.settings)
        """
        Injects dependencies with appropriate attributes.
        """
        self._inject()
        """
        Instances a Evaluator class for storing results of each Recipe.bake.
        """
        self.results = Results()
        """
        Sets key scoring metric for methods that require a single scoring
        metric.
        """
        self.key_metric = self._listify(self.metrics)[0]
        """
        Creates empty lists and dictionary for custom methods and parameters
        to be added by user.
        """
        self.customs = []
        self.customs_params = {}
        self.customs_runtime_params = {}
        """
        Declares possible classes and steps in a cookbook recipe.
        """
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
        """
        Adds any new algorithms passed in Cookbook instance.
        """
        if self.new_algorithms:
            for step, nested_dict in self.new_algorithms.items():
                for key, value in nested_dict.items():
                    self.step_classes[step].options.update({key, value})
        """
        Data is split in oder for certain values to be computed that require
        features and the label to be split.
        """
        if self.compute_hyperparameters:
            self.data.split_xy(label = self.label)
            self._compute_hyperparameters()
        return self

    def __getitem__(self, value):
        """
        Allows users to access particular algorithms by passing [step, name].
        """
        step, name = value
        if step in self.step_classes:
            return self.recipe_steps[step].options[name]
        else:
            error_message = step + ' or ' + name + ' not found'
            raise KeyError(error_message)
            return

    def __setitem__(self, value):
        """
        Allows users to add new algorithms by passing either strings or lists
        of strings containing the steps, names, and algorithms in the form of
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
        """
        Allows user to delete algorithms by passing [steps, algorithm_names].
        """
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
        """
        Injects filer, settings, and _listify method into Step class, Recipe
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
        """
        Sets and creates folder paths for recipe results to be stored.
        """
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
        """
        Initializes the step classes for use by the Cookbook.
        """
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
        """
        This method contains any hyperparameters that are computed based
        upon the source data.
        """
        Model.scale_pos_weight = (len(self.data.y.index) /
                                  ((self.data.y == 1).sum())) - 1
        return self

    def _listify(self, variable):
        """
        Checks to see if the methods are stored in a list. If not, the
        methods are converted to a list or a list of 'none' is created.
        """
        if not variable:
            return ['none']
        elif isinstance(variable, list):
            return variable
        else:
            return [variable]

    def _stringify(self, variable):
        if not variable:
            return 'none'
        elif isinstance(variable, str):
            return variable
        else:
            return variable[0]


    def _check_best(self, recipe):
        """
        Checks if the current Recipe is better than the current best Recipe.
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
        """
        Allows user to manually add algorithms to recipe steps.
        """
        steps = self._listify(steps)
        ingredients = self._listify(ingredients)
        algorithms = self._listify(algorithms)
        new_algorithms = zip(steps, ingredients, algorithms)
        for step, ingredient, algorithm in new_algorithms.items():
            self.recipe_classes[step].include(ingredient, algorithm, **kwargs)
        return self

    def add_splice(self, splice_label, prefixes = [], columns = []):
        """
        Adds splices to the list of splicers.
        """
        self.splicers.add_splice(splice_label = splice_label,
                                 prefixes = prefixes,
                                 columns = columns)
        self.splicers.append(splice_label)
        return self

    def create(self):
        """
        This method creates the cookbook with all possible selected
        preprocessing, modeling, and testing methods. Each set of methods is
        stored in a list of instances of the Recipe class (self.recipes).
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
        """
        This method iterates through each of the possible recipes. The
        best overall recipe is stored in self.best_recipe.
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
        """
        Completes one iteration of a Cookbook, storing the results in the
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
            """
            To conserve memory, each recipe is deleted after being exported.
            """
            del(recipe)
        return self

    def apply(self):
        """
        Implements bake method for those who prefer non-cooking method names.
        """
        self.bake()
        return self

    def print_best(self):
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
        """
        Automatically saves the cookbook, scores, and best recipe.
        """
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
        """
        Imports a single pickled cookbook from disc.
        """
        if not import_path:
            import_path = self.filer.import_folder
        recipes = pickle.load(open(import_path, 'rb'))
        if return_cookbook:
            return recipes
        else:
            self.recipes = recipes
            return self

    def save(self, export_path = None):
        """
        Exports a cookbook to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
#        pickle.dump(self.recipes, open(export_path, 'wb'))
        return self