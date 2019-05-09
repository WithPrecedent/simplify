"""
The siMpLify package allows users to create a cookbook of dynamic recipes that
mix-and-match feature engineering and modeling ingredients based upon a common,
simple interface.

siMpLify divides the feature engineering and modeling process into ten major
steps that can be sequenced in different orders:

    Scaler: converts numerical features into a common scale, using scikit-learn
        methods.
    Splitter: divides data into train, test, and/or validation sets once or
        through k-folds cross-validation.
    Encoder: converts categorical features into numerical ones, using
        category-encoders methods.
    Interactor: converts selected categorical features into new polynomial
        features, using PolynomialEncoder from category-encoders.
    Splicer: creates different subgroups of features to allow for easy
        comparison between them.
    Sampler: synthetically resamples training data for imbalanced data,
        using imblearn methods, for use with models that struggle with
        imbalanced data.
    Selector: selects features recursively or as one-shot based upon user
        criteria, using scikit-learn methods.
    Custom: allows users to add any scikit-learn or siMpLify compatible
        method to be added into an recipe.
    Model: implements machine learning algorithms, currently includes
        xgboost and scikit-learn methods. The user can opt to either test
        different hyperparameters for the models selected or a single set
        of hyperparameters. Search methods currently include
        RandomizedSearchCV and GridSearchCV - Bayesian methods coming soon.
    Plotter: produces helpful graphical representations based upon the model
        selected, includes shap, seaborn, and matplotlib methods.

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
        Custom: child classes of Step which contain the different ingredient
        options for each step in a recipe.
    Model: contains different machine learning algorithms divided into four
        major model_type: classifier, regressor, and unsupervised. It is
        also a child class of Step.
    Plotter: another Step child class that prepares and exports plots based
        upon the model_type.
    Results: another Step child class that applies user-selected or default
        metrics for each recipe and stores the results in a dataframe
        (Results.table). Each row of the results table stores each of the steps
        used, the folder in which the relevant files are stored, and all of
        the metrics for that recipe.
    Filer: creates and contains the path structure for loading data and
        settings as well as saving results, data, and plots.
    Settings: contains the methods for parsing the settings file to create
        a nested dictionary used by the other classes.

If the user opts to use the settings.ini file, the only classes that absolutely
need to be used are Cookbook and Data. Nonetheless, the rest of the classes and
attributes are still available for use. All of the ten step classes are stored
in a list of recipes (Cookbook.recipes). Cookbook.results collects the results
from the recipes.

If a Filer instance is not passed to Cookbook when it instanced, an import_folder
and export_folder must be passed. Then the Cookbook will create an instance of
Filer as an attribute of the Cookbook (Cookbook.filer).
If an instance of settings is not passed when the Cookbook is instanced, a
settings file will be loaded automatically.
"""

"""
cookbook.py is the primary control file for the siMpLify package. It contains
the Cookbook class, which handles the cookbook construction and application.
"""
from dataclasses import dataclass
import datetime
from itertools import product
import os
import pickle
import warnings

from simplify.custom import Custom
from simplify.encoder import Encoder
from simplify.filer import Filer
from simplify.interactor import Interactor
from simplify.model import Model
from simplify.plotter import Plotter
from simplify.recipe import Recipe
from simplify.results import Results
from simplify.sampler import Sampler
from simplify.scaler import Scaler
from simplify.selector import Selector
from simplify.splicer import Splicer
from simplify.splitter import Splitter
from simplify.step import Step
from simplify.settings import Settings


@dataclass
class Cookbook(object):
    """
    Class for creating dynamic recipes for preprocessing, machine learning,
    and data analysis using a unified interface and architecture.
    """
    data : object
    filer : object = None
    data_folder : str = ''
    results_folder : str = ''
    settings : object = None
    settings_path : str = ''
    splicers : object = None
    new_algorithms : object = None
    best_recipe : object = None

    def __post_init__(self):
        """
        Loads settings from an .ini file if not passed when class is instanced.
        Otherwise an empty
        """
        if not self.settings:
            if not self.settings_path:
                self.settings_path = os.path.join('..', 'settings',
                                                  'simplify_settings.ini')
            self.settings = Settings(file_path = self.settings_path)
        self.settings.localize(class_instance = self,
                               sections = ['general', 'files', 'steps'])
        """
        Removes the numerous pandas warnings from console output if option
        selected by user.
        """
        if not self.pandas_warnings:
            warnings.filterwarnings('ignore')
        """
        Adds a Filer instance if one is not passed when class is instanced.
        """
        if not self.filer:
            self.filer = Filer(root_import = self.data_folder,
                               root_export = self.results_folder,
                               recipe_folder = self.recipe_folder,
                               settings = self.settings)
        """
        Instances a Results class for storing results of each Recipe.bake.
        """
        self.results = Results(settings = self.settings)
        """
        Sets key scoring metric for tools that require a single scoring metric.
        """
        self.key_metric = self._listify(self.settings['results']['metrics'])[0]
        """
        Creates empty lists and dictionary for custom methods and splicers
        to be added by user.
        """
        self.customs = []
        self.customs_params = {}
        """
        Declares possible methods classes and steps in cookbook.
        """
        self.step_classes = {'scalers' : Scaler,
                             'splitter' : Splitter,
                             'encoders' : Encoder,
                             'interactors' : Interactor,
                             'splicers' : Splicer,
                             'samplers' : Sampler,
                             'selectors' : Selector,
                             'customs' : Custom,
                             'models' : Model,
                             'plotters' : Plotter}
        self.steps = list(self.step_classes.keys())
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
            return self.class_steps[step].options[name]
        else:
            error_message = step + ' or ' + name + ' not found'
            raise KeyError(error_message)
            return

    def __setitem__(self, value):
        """
        Allows users to add new algorithms by passing either strings or lists
        of strings containing the steps, names, and algorithms in the form of
        [steps, names, values].
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
                        self.class_steps[step][name][algorithm]
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
        Allows user to delete algorithms by passing [steps, names].
        """
        steps, names = value
        steps = self._listify(steps)
        names = self._listify(names)
        del_steps = zip(steps, names)
        for step, name in del_steps.items():
            if name in self.class_steps[step].options:
                self.class_steps[step][name]
            else:
                error_message = name + ' is not in ' + step + ' method'
                raise KeyError(error_message)
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
        for step in self.steps:
            setattr(self, step, self._listify(getattr(self, step)))
            if not step in ['models']:
                param_var = step + '_params'
                setattr(self, param_var, self.settings[param_var])
        """
        Injects filer, random seed, settings, and _listify method into Step
        class.
        """
        Step.filer = self.filer
        Step.seed = self.seed
        Step.settings = self.settings
        Step._listify = self._listify
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

    def _one_loop(self, use_val_set = False):
        """
        Completes one iteration of a Recipe, storing the results in the results
        table dataframe. Plots and the recipe are exported to the recipe
        folder.
        """
        self._set_folders()
        for recipe in self.recipes:
            if self.verbose:
                print('Testing recipe ' + str(recipe.number))
            self.data.split_xy(label = self.label)
            recipe.bake(data = self.data,
                        use_val_set = use_val_set)
            self.results.add_result(recipe = recipe,
                                    use_val_set = use_val_set)
            self._check_best(recipe)
            file_name = 'recipe' + recipe.number + '_' + recipe.model.name
            export_path = self.filer._iter_path(model = recipe.model,
                                                recipe_num = recipe.number,
                                                splicer = recipe.splicer,
                                                file_name = file_name,
                                                file_type = 'pickle')
            self.save_recipe(recipe, export_path = export_path)
            """
            To conserve memory, each recipe is deleted after being exported.
            """
            del(recipe)
        return self

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


    def _visualize_recipe(self, recipe):
        """
        Iterates through all selected plots for a single recipe.
        """
        if self.visuals == 'default':
            plots = list(self.plotter.options.keys())
        else:
            plots = self._listify(self.visuals)
        self.plotter._one_cycle(plots = plots,
                                model = recipe.model)
        return self

    def add_steps(self, steps, names, methods):
        """
        Allows user to manually add an algorithm to the varios methods
        dictionaries.
        """
        steps = self._listify(steps)
        names = self._listify(names)
        methods = self._listify(methods)
        new_algorithms = zip(steps, names, methods)
        for step, name, method in new_algorithms.items():
            self.step_classes[step].options.update({name, method})
        return self

    def add_splice(self, splice, prefixes = [], columns = []):
        """
        Adds splices to the list of splicers.
        """
        self.splicers.add_splice(splice = splice, prefixes = prefixes,
                                 columns = columns)
        self.splicers.append(splice)
        return self

    def create(self):
        """
        This method creates the cookbook with all possible selected preprocessing
        and modelling methods. Each set of methods is stored in a list of
        instances of Recipe (self.recipes).
        """
        if self.verbose:
            print('Creating all possible preprocessing and modeling recipes')
        self._prepare_steps()
        self.recipes = []
        all_steps = product(self.scalers, self.splitter, self.encoders,
                            self.interactors, self.splicers, self.samplers,
                            self.customs, self.selectors, self.models,
                            self.plotter)
        for i, (scaler, splitter, encoder, interactor, splicer, sampler,
                custom, selector, algorithm, plotter) in enumerate(all_steps):
            model_params = (self.settings[algorithm + '_params'])
            recipe = Recipe(i + 1,
                            self.order,
                            Scaler(scaler, self.scalers_params),
                            Splitter(splitter, self.splitter_params),
                            Encoder(encoder, self.encoders_params,
                                    self.data.cat_cols),
                            Interactor(interactor, self.interactors_params,
                                       self.data.interact_cols),
                            Splicer(splicer, self.splicers_params),
                            Sampler(sampler, self.samplers_params),
                            Custom(custom, self.customs_params),
                            Selector(selector, self.selectors_params),
                            Model(algorithm, self.model_type, model_params),
                            Plotter(self.plotters_params),
                            self.settings)
            self.recipes.append(recipe)
        return self

    def iterate(self):
        """
        This method iterates through each of the possible recipes. The
        best overall recipe is stored in self.best_recipe.
        """
        if self.verbose:
            print('Testing recipes')
        self.best_recipe = None
        self._one_loop()
        if self.splitter_params['val_size'] > 0:
            self._one_loop(use_val_set = True)
        return self

    def visualize(self, recipe = None, cookbook = None):
        """
        Allows user to manually create plots for a single recipe or entire
        cookbook.
        """
        if recipe:
            self._visualize_recipe(recipe)
        else:
            if not cookbook:
                cookbook = self
            for recipe in cookbook.recipes:
                self._visualize_recipe(recipe)
        return self

    def save_everything(self):
        """
        Automatically saves the cookbook, results table, and best recipe.
        """
        self.save_cookbook(export_path = os.path.join(self.filer.results,
                                                      'cookbook.pkl'))
        self.save_results(export_path = os.path.join(self.filer.results,
                                                     'results_table.csv'))
        if self.best_recipe:
            self.save_recipe(export_path = os.path.join(
                    self.filer.results, 'best_recipe.pkl'), recipe = self.best_recipe)
        return self

    def load_cookbook(self, import_path = None, return_cookbook = False):
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

    def save_cookbook(self, export_path = None):
        """
        Exports a cookbook to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(self.recipes, open(export_path, 'wb'))
        return self

    def load_recipe(self, import_path = None):
        """
        Imports a single recipe from disc.
        """
        if not import_path:
            import_path = self.filer.import_folder
        recipe = pickle.load(open(import_path, 'rb'))
        return recipe

    def save_recipe(self, recipe, export_path = None):
        """
        Exports a recipe to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(recipe, open(export_path, 'wb'))
        return self

    def load_results(self, import_path = None, file_name = 'results_table',
                     file_format = 'csv', encoding = 'windows-1252',
                     float_format = '%.4f', message = 'Importing results',
                     return_results = False):
        """
        Imports results table file from disc. This method can be used if
        the user wants to reconstruct parts of recipes with loading the entire
        cookbook or individual recipes.
        """
        if not import_path:
            import_path = self.filer.import_folder
        results_path = self.filer.path_join(folder = import_path,
                                            file_name = file_name,
                                            file_type = file_format)
        results = self.data.load(import_path = results_path,
                                 encoding = encoding,
                                 float_format = float_format,
                                 message = message)
        if return_results:
            return results
        else:
            self.results.table = results
            return self

    def save_results(self, export_path = None, file_name = 'results_table',
                     file_format = 'csv', encoding = 'windows-1252',
                     float_format = '%.4f', message = 'Exporting results'):
        """
        Exports results table to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
            export_path = self.filer.make_path(folder = export_path,
                                               name = file_name,
                                               file_type = file_format)
        self.data.save(df = self.results.table,
                       export_path = export_path,
                       encoding = encoding,
                       float_format = float_format,
                       message = message)
        return self