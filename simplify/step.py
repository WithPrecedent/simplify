"""
Step is the parent class containing shared methods for all steps in siMpLify
recipes.
"""
from dataclasses import dataclass
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
                               sections = ['general', 'steps'])
        return self

    def __getitem__(self, ingredient):
        if ingredient in self.options:
            return self.options[ingredient]
        else:
            error_message = (
                    ingredient + ' is not in ' + self.name + ' algorithm')
            raise KeyError(error_message)
            return

    def __setitem__(self, step, ingredient):
        if isinstance(step, str):
            if isinstance(ingredient, object):
                self.options.update({step : ingredient})
            else:
                error_message = step + ' must be a Step object'
                raise TypeError(error_message)
        else:
            error_message = ingredient + ' must be a string type'
            raise TypeError(error_message)
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

    def _select_params(self, params_to_use = []):
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

    def _list_type(self, test_list, data_type):
        return any(isinstance(i, data_type) for i in test_list)

    def initialize(self):
        self._check_params()
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
        new_params = {}
        if self.params:
            for key, value in self.params.items():
                if key in params_to_use:
                    new_params.update({key : value})
            self.params = new_params
        return self

    def mix(self, x, y = None):
        if self.name != 'none':
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return x

    def fit(self, x, y):
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