"""
Step is the parent class containing shared methods for all steps in siMpLify
recipes.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle


@dataclass
class Step(object):
    """
    Parent class for preprocessing and modeling algorithms in siMpLify.
    The Step class allows for shared initialization, loading, and saving
    methods to be accessed by all machine learning and preprocessing steps.
    """
    def __post_init__(self):
        self.settings.localize(instance = self,
                               sections = ['general', 'recipes'])
        return self

    def __getitem__(self, ingredient):
        if ingredient in self.options:
            return self.options[ingredient]
        else:
            error = ingredient + ' is not in ' + self.name + ' algorithm'
            raise KeyError(error)
            return

    def __setitem__(self, ingredient, algorithm):
        if isinstance(ingredient, str):
            if isinstance(algorithm, object):
                self.options.update({ingredient : algorithm})
            else:
                error = ingredient + ' must be a Step object'
                raise TypeError(error)
        else:
            error = algorithm + ' must be a string type'
            raise TypeError(error)
        return self

    def __delitem__(self, ingredient):
        if ingredient in self.options:
            self.options.pop(ingredient)
        else:
            error_message = (
                    ingredient + ' is not in ' + self.name + ' algorithm')
            raise KeyError(error_message)
        return self

    def __contains__(self, ingredient):
        if ingredient in self.options:
            return True
        else:
            return False

    def _check_params(self):
        if not self.params:
            self.params = self.defaults
        return self

    def _get_indices(self, df, columns):
        col_indices = [df.columns.get_loc(col) for col in columns]
        return col_indices

    def _list_type(self, test_list, data_type):
        return any(isinstance(i, data_type) for i in test_list)

    def _store_feature_names(self, x, y = None):
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def _get_feature_names(self, x, y = None):
        x = pd.DataFrame(x, columns = self.x_cols)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name = self.y_col)
            return x, y
        else:
            return x

    def initialize(self, select_params = False):
        """
        Adds parameters to algorithm selected in a recipe step.
        """
        self._check_params()
        if select_params:
            self.select_params(params_to_use = list(self.defaults.keys()))
        if self.runtime_params:
            self.params.update(self.runtime_params)
        if self.name != 'none':
            self.algorithm = self.options[self.name]
            if self.params:
                self.algorithm = self.algorithm(**self.params)
            else:
                self.algorithm = self.algorithm()
        return self

    def select_params(self, params_to_use = []):
        """
        For subclasses that only need a subset of the parameters stored in the
        settings file, this function selects that subset.
        """
        new_params = {}
        if self.params:
            for key, value in self.params.items():
                if key in params_to_use:
                    new_params.update({key : value})
            self.params = new_params
        return self

    def include(self, ingredients, algorithms, **kwargs):
        """
        Adds new ingredient name and corresponding algorithm to a step options
        dictionary.
        """
        ingredients = self._listify(ingredients)
        algorithms = self._listify(algorithms)
        new_algorithms = zip(ingredients, algorithms)
        for ingredient, algorithm in new_algorithms.items():
            self.options.update({ingredient, algorithm})
        return self

    def mix(self, x, y = None):
        if self.name != 'none':
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return x

    def fit(self, x, y):
        self.initialize()
        return self.algorithm.fit(x, y)

    def transform(self, x):
        return self.algorithm.transform(x)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def load(self, file_name, import_folder = '', prefix = '', suffix = ''):
        import_path = self.filer.path_join(folder = import_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        if self.verbose:
            print('Importing', file_name)
        self.algorithm = pickle.load(open(import_path, 'rb'))
        return self

    def save(self, file_name, export_folder = '', prefix = '', suffix = ''):
        if self.verbose:
            print('Exporting', file_name)
        export_path = self.filer.path_join(folder = export_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        pickle.dump(self.algorithm, open(export_path, 'wb'))
        return self