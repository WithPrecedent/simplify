"""
The ml_funnel package allows users to create dynamic experiments that mix
feature engineering and modeling methods based upon easy-to-use selections.

ml_funnel divides the feature engineering and modeling process into ten major
method groups:

    Scalers: convert numerical features into a common scale, using scikit-learn
        methods.
    Splitter: divides data into train, test, and/or validation sets once or
        through k-folds cross-validation.
    Encoders: convert categorical features into numerical ones, using
        category-encoders methods.
    Interactors: convert selected categorical features into new polynomial
        features, using PolynomialEncoder from category-encoders.
    Splicers: create different subgroups of features to allow for easy
        comparison between them.
    Samplers: synthetically resample training data for imbalanced data,
        using imblearn methods, for use with models that struggle with
        imbalanced data.
    Selectors: select features recursively or as one-shot based upon user
        criteria, using scikit-learn methods.
    Customs: allow users to add any scikit-learn or ml_funnel compatible
        method to be added into an experiment.
    Models: implement machine learning algorithms, currently includes
        xgboost and scikit-learn methods. The user can opt to either test
        different hyperparameters for the models selected or a single set
        of hyperparameters. Search methods currently include
        RandomizedSearchCV and GridSearchCV - Bayesian methods coming soon.
    Plots: produce helpful graphical representations based upon the model
        selected, includes shap, seaborn, and matplotlib methods.

ml_funnel contains the following accessible classes:
    Funnel: containing the methods needed to create dynamic experiments and
        stores the different test tubes in Funnel.tubes. For that reason,
        the Tube class does not need to be instanced directly.
    Tube: if the user wants to manually create a single test tube, the Tube
        class is made public for this purpose. This Tube can then be passed
        to various methods of other classes.
    Data: includes methods for creating and modifying pandas dataframes used
        by the funnel. As the data is split into features, labels, test,
        train, etc. dataframes, they are all created as attributes to an
        instance of the Data class.
    Scaler, Splitter, Encoder, Interactor, Splicer, Sampler, and Selector:
        child classes of Methods which contain the different algorithms for
        each step in a test tube.
    Model: contains different machine learning algorithms divided into three
        major model_type: classifier, regressor, and grouper. It is also
        a child class of Methods.
    Grid: contains the method and parameters for different hyperparameter
        search methods as a Methods child class. If the user includes two
        values for any hyperparameter, the Grid method is automatically
        implemented. In such cases, the best hyperparameter set is stored
        in Funnel.best.
    Plotter: another Methods child class that prepares and exports plots based
        upon the model model_type. Users can directly access the three
        major graphing method groups: ClassifierPlotter, GrouperPlotter,
        LinearPlotter.
    Results: another Methods child class that applies user-selected or default
        metrics for each test tube and stores the results in a dataframe
        (.table). Each row of the results table stores each of the methods
        used, the folder in which the relevant files are stored, and all of
        the metrics for that test tube.
    Filer: creates and contains the path structure for loading data and
        settings as well as saving results, data, and plots.
    Settings: contains the methods for parsing the settings file to create
        a nested dictionary used by the other classes.

If the user opts to use the settings.ini file, the only classes that absolutely
needs to be used is Funnel. Nonetheless, the rest of the classes and attributes
are still available even when only using the Funnel directly. All other
classes will be created as lower-case named attributes of the funnel instance
(e.g. Funnel.scaler, Funnel.data, Funnel.plotter, etc.).

If a Filer instance is not passed to Funnel when it instanced, an import_folder
and export_folder must be passed. Then the Funnel will create an instance of
Filer as an attribute of the Funnel (Funnel.filer).
If an instance of settings is not passed when the Funnel is instanced, a
settings file will be loaded automatically.
"""

"""
funnel.py is the primary control file for the ml_funnel package. It contains
the Funnel class, which handles the funnel construction and application.
"""
from dataclasses import dataclass
import datetime
from itertools import product
import os
import pickle
import warnings

from ml_funnel.filer import Filer
from ml_funnel.methods import Custom, Encoder, Interactor, Methods, Model
from ml_funnel.methods import Sampler, Scaler, Selector, Splicer, Splitter
from ml_funnel.plotter import ClassifierPlotter, GrouperPlotter, LinearPlotter
from ml_funnel.plotter import Plotter
from ml_funnel.results import Results
from ml_funnel.settings import Settings
from ml_funnel.test_tube import Tube

@dataclass
class Funnel(object):
    """
    Class for creating dynamic test tubes for preprocessing, machine learning,
    and data analysis using a unified interface and architecture.
    """
    data : object
    filer : object = None
    data_folder : str = ''
    results_folder : str = ''
    settings : object = None
    settings_path : str = ''
    splicers : object = None
    new_methods : object = None
    best : object = None

    def __post_init__(self):
        """
        Loads settings from an .ini file if not passed when class is instanced.
        Otherwise an empty
        """
        if not self.settings:
            if not self.settings_path:
                self.settings_path = os.path.join('..', 'settings',
                                                  'ml_funnel_settings.ini')
            self.settings = Settings(file_path = self.settings_path)
        self.settings.simplify(class_instance = self,
                               sections = ['general', 'files', 'methods'])
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
                               experiment_folder = self.experiment_folder,
                               settings = self.settings)
        """
        Instances a Results class for storing results of each Tube.apply.
        """
        self.results = Results(settings = self.settings)
        """
        Sets key scoring metric for tools that require a single scoring metric.
        """
        self.key_metric = self._listify(self.settings['results']['metrics'])[0]
        """
        Assigns appropriate plotter class based upon model_type.
        """
        Plotter.params = self.settings['plotters_params']
        self.plotter = self._set_plotter(plotter_type = self.model_type)
        """
        Creates empty lists and dictionary for custom methods and splicers
        to be added by user.
        """
        self.customs = []
        self.customs_params = {}
        self.splicers = []
        """
        Declares possible methods classes and steps in funnel.
        """
        self.method_classes = {'scalers' : Scaler,
                               'splitter' : Splitter,
                               'encoders' : Encoder,
                               'interactors' : Interactor,
                               'splicers' : Splicer,
                               'samplers' : Sampler,
                               'selectors' : Selector,
                               'customs' : Custom,
                               'models' : Model,
                               'plotters' : self.plotter}
        self.steps = list(self.method_classes.keys())
        """
        Adds any new methods passed in Funnel instance.
        """
        if self.new_methods:
            for step, nested_dict in self.new_methods.items():
                for key, value in nested_dict.items():
                    self.method_classes[step].options.update({key, value})
        """
        Data is split in oder for certain values to be computed that require
        features and the label to be split.
        """
        if self.compute_hyperparameters:
            self.data.split_xy(label = self.label)
            self._compute_hyperparameters()
        return self

    def __getitem__(self, value):
        step, name = value
        if step in self.method_classes:
            return self.class_methods[step][name]
        else:
            error_message = step + ' or ' + name + ' not found'
            raise KeyError(error_message)
            return

    def __setitem__(self, value):
        steps, names, methods = value
        if isinstance(steps, str) or isinstance(steps, list):
            if isinstance(names, str) or isinstance(names, list):
                if isinstance(methods, object) or isinstance(methods, list):
                    steps = self._listify(steps)
                    names = self._listify(names)
                    steps = self._listify(steps)
                    new_methods = zip(steps, names, methods)
                    for step, name, method in new_methods.items():
                        self.class_methods[step][name][method]
                else:
                    error_message = name + ' must be an object of list of objects'
                    raise TypeError(error_message)
            else:
                error_message = name + ' must be a string or list of strings'
                raise TypeError(error_message)
        else:
            error_message = step + ' must be a string or list of strings'
            raise TypeError(error_message)
        return self

    def __delitem__(self, value):
        steps, names = value
        steps = self._listify(steps)
        names = self._listify(names)
        del_methods = zip(steps, names)
        for step, name in del_methods.items():
            if name in self.class_methods[step].options:
                self.class_methods[step][name]
            else:
                error_message = name + ' is not in ' + step + ' method'
                raise KeyError(error_message)
        return self

    def _set_folders(self):
        """
        Sets and creates folder paths for experimental results to be stored.
        """
        if self.experiment_folder == 'dynamic':
            subfolder = ('experiment_'
                         + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        else:
            subfolder = self.experiment_folder
        self.filer.results = os.path.join(self.filer.results, subfolder)
        self.filer.test_tubes = self.filer.make_path(
                folder = self.filer.results,
                subfolder = 'test_tubes')
        self.filer._make_folder(self.filer.results)
        self.filer._make_folder(self.filer.test_tubes)
        return self

    def _prepare_methods(self):
        for step in self.steps:
            setattr(self, step, self._listify(getattr(self, step)))
            if not step in ['models']:
                param_var = step + '_params'
                setattr(self, param_var, self.settings[param_var])
        """
        Injects filer, random seed, settings, and column lists into Methods
        classes.
        """
        Methods.filer = self.filer
        Methods.seed = self.seed
        Methods.settings = self.settings
        Methods._listify = self._listify
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

    def _select_params(self, params_to_use = []):
        new_params = {}
        if self.params:
            for key, value in self.params.items():
                if key in params_to_use:
                    new_params.update({key : value})
            self.params = new_params
        return self

    def _compute_values(self):
        """
        This method contains any hyperparameters that are computed based
        upon the source data.
        """
        if self.compute_scale_pos_weight:
            Model.scale_pos_weight = (len(self.data.y.index) /
                                      ((self.data.y == 1).sum())) - 1
        return self

    def _set_plotter(self, plotter_type):
        options = {'classifier' : ClassifierPlotter,
                   'regressor' : LinearPlotter,
                   'grouper' : GrouperPlotter}
        return options[plotter_type]

    def _one_loop(self, use_val_set = False):
        """
        Completes one iteration of a Tube, storing the results in the results
        table dataframe. Plots and the tube are exported to the experiment
        folder.
        """
        self._set_folders()
        for i, tube in enumerate(self.tubes):
            if self.verbose:
                print('Testing tube ' + str(i + 1))
            self.data.split_xy(label = self.label)
            tube.apply(tube_num = str(i + 1),
                       data = self.data,
                       use_val_set = use_val_set)
            self.results.add_result(tube = tube,
                                    use_val_set = use_val_set)
            self._check_best(tube)
            file_name = 'tube' + tube.tube_num + '_' + tube.model.name
            export_path = self.filer._iter_path(model = tube.model,
                                                tube_num = tube.tube_num,
                                                splicer = tube.splicer,
                                                file_name = file_name,
                                                file_type = 'pickle')
            self.save_tube(tube, export_path = export_path)
            del(tube)
        return self

    def _check_best(self, tube):
        """
        Checks if the current Tube is better than the current best Tube.
        """
        if not self.best:
            self.best = tube
            self.best_score = self.results.table.loc[
                    self.results.table.index[-1], self.key_metric]
        elif (self.results.table.loc[self.results.table.index[-1],
                                     self.key_metric] > self.best_score):
            self.best = tube
            self.best_score = self.results.table.loc[
                    self.results.table.index[-1], self.key_metric]
        return self


    def _visualize_tube(self, tube):
        """
        Iterates through all selected plots for a single test tube.
        """
        if self.visuals == 'default':
            plots = list(self.plotter.options.keys())
        else:
            plots = self._listify(self.visuals)
        self.plotter._one_cycle(plots = plots,
                                model = tube.model)
        return self

    def add_methods(self, steps, names, methods):
        """
        Allows user to manually add an algorithm to the varios methods
        dictionaries.
        """
        steps = self._listify(steps)
        names = self._listify(names)
        methods = self._listify(methods)
        new_methods = zip(steps, names, methods)
        for step, name, method in new_methods.items():
            self.method_classes[step].options.update({name, method})
        return self

    def add_splice(self, splice, prefixes = [], columns = []):
        self.splicer.add_splice(splice = splice, prefixes = prefixes,
                                columns = columns)
        self.splicers.append(splice)
        return self

    def create(self):
        """
        This method creates the funnel with all possible selected preprocessing
        and modelling methods. Each set of methods is stored in a list of
        instances of Tube (self.tubes).
        """
        if self.verbose:
            print('Creating all possible preprocessing test tubes')
        self._prepare_methods()
        self.tubes = []
        all_methods = product(self.scalers, self.splitter, self.encoders,
                              self.interactors, self.splicers, self.samplers,
                              self.customs, self.selectors, self.models)
        for (scaler, splitter, encoder, interactor, splicer, sampler, custom,
             selector, algorithm) in all_methods:
            model_params = (self.settings[algorithm + '_params'])
            model = Model(algorithm, self.model_type, model_params, self.gpu)
            tube = Tube(self.steps,
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
                        model,
                        self.plotter(self.plotters, self.plotters_params))
            self.tubes.append(tube)
        return self

    def iterate(self):
        """
        This method iterates through each of the possible test tubes. The
        best overall test tube is stored in self.best.
        """
        if self.verbose:
            print('Testing tubes')
        self.best = None
        self._one_loop()
        if self.splitter_params['val_size'] > 0:
            self._one_loop(use_val_set = True)
        return self

    def add_plot(self, name, method):
        """
        Allows user to manually add a plot option.
        """
        self.plotter.options.update({name : method})
        return self

    def del_plot(self, name):
        self.plotter.options.pop(name)
        return self

    def visualize(self, tube = None, funnel = None):
        """
        Allows user to manually create plots for a single test tube or funnel.
        """
        if tube:
            self._visualize_tube(tube)
        else:
            if not funnel:
                funnel = self
            for tube in funnel.pipes:
                self._visualize_tube(tube)
        return self

    def save_everything(self):
        """
        Automatically saves the funnel, results table, and best tube.
        """
        self.save_funnel(export_path = os.path.join(self.filer.results,
                                                    'funnel.pkl'))
        self.save_results(export_path = os.path.join(self.filer.results,
                                                     'results_table.csv'))
        if self.best:
            self.save_tube(export_path = os.path.join(
                    self.filer.results, 'best_tube.pkl'), tube = self.best)
        return self

    def load_funnel(self, import_path = None, return_funnel = False):
        """
        Imports a single pickled funnel from disc.
        """
        if not import_path:
            import_path = self.filer.import_folder
        tubes = pickle.load(open(import_path, 'rb'))
        if return_funnel:
            return tubes
        else:
            self.tubes = tubes
            return self

    def save_funnel(self, export_path = None):
        """
        Exports a funnel to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(self.tubes, open(export_path, 'wb'))
        return self

    def load_tube(self, import_path = None):
        """
        Imports a single tube from disc.
        """
        if not import_path:
            import_path = self.filer.import_folder
        tube = pickle.load(open(import_path, 'rb'))
        return tube

    def save_tube(self, tube, export_path = None):
        """
        Exports a tube to disc.
        """
        if not export_path:
            export_path = self.filer.results_folder
        pickle.dump(tube, open(export_path, 'wb'))
        return self

    def load_results(self, import_path = None, file_name = 'results_table',
                     file_format = 'csv', encoding = 'windows-1252',
                     float_format = '%.4f', message = 'Importing results',
                     return_results = False):
        """
        Imports results table file from disc. This method can be used if
        the user wants to reconstruct parts of tubes with loading the entire
        funnel or individual tubes.
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