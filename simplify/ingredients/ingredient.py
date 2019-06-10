
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle


@dataclass
class Ingredient(object):
    """Parent class for preprocessing and modeling techniques in siMpLify.
    The Ingredient class allows for shared initialization, loading, and saving
    methods to be accessed by all machine learning and preprocessing steps.
    """
    def __post_init__(self):
        """Adds local attributes from settings."""
        self.settings.localize(instance = self,
                               sections = ['general', 'recipes'])
        return self

    def __contains__(self, technique):
        """Checks whether technique name is lists in options."""
        if technique in self.options:
            return True
        else:
            return False

    def __delitem__(self, technique):
        """Deletes technique and algorithm if technique in options."""
        if technique in self.options:
            self.options.pop(technique)
        else:
            error_message = (
                    technique + ' is not in ' + self.__class__.__name__)
            raise KeyError(error_message)
        return self

    def __getitem__(self, technique):
        """Gets algorithm from options if technique in options."""
        if technique in self.options:
            return self.options[technique]
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
            return

    def __setitem__(self, technique, algorithm):
        """Adds technique and corresponding algorithm to options."""
        if isinstance(technique, str):
            if isinstance(algorithm, object):
                self.options.update({technique : algorithm})
            else:
                error = technique + ' must be an Ingredient object'
                raise TypeError(error)
        else:
            error = algorithm + ' must be a string type'
            raise TypeError(error)
        return self

    def _add_param(self, param):
        """Adds a param set to params dictionary."""
        return self.params.update(param)

    def _check_params(self):
        """Checks if params exists. If not, defaults are used."""
        if not self.params:
            self.params = self.defaults
        return self

    def _get_feature_names(self, x, y = None):
        """Gets feature names."""
        x = pd.DataFrame(x, columns = self.x_cols)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name = self.y_col)
            return x, y
        else:
            return x

    def _get_indices(self, df, columns):
        """Gets column indices for a list of column names."""
        col_indices = [df.columns.get_loc(col) for col in columns]
        return col_indices

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

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

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def add(self, techniques, algorithms, **kwargs):
        """Adds new technique name and corresponding algorithm to the subclass
        options dictionary.
        """
        new_algorithms = zip(self._listify(techniques),
                             self._listify(algorithms))
        for technique, algorithm in new_algorithms.items():
            self.options.update({technique, algorithm})
        return self

    def initialize(self, select_params = False):
        """Adds parameters to ingredient algorithm."""
        self._check_params()
        if select_params:
            self.select_params(params_to_use = list(self.defaults.keys()))
        if self.runtime_params:
            self.params.update(self.runtime_params)
        if self.technique != 'none':
            self.algorithm = self.options[self.technique]
            if self.params:
                self.algorithm = self.algorithm(**self.params)
            else:
                self.algorithm = self.algorithm()
        return self

    def select_params(self, params_to_use = []):
        """For subclasses that only need a subset of the parameters stored in
        settings, this function selects that subset.
        """
        new_params = {}
        if self.params:
            for key, value in self.params.items():
                if key in params_to_use:
                    new_params.update({key : value})
            self.params = new_params
        return self

    def mix(self, x, y = None):
        """Generic mix method for adding ingredients into recipe and applying
        the appropriate algorithm.
        """
        if self.algorithm != 'none':
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return x

    def fit(self, x, y):
        """Generic fit method for partial compatibility to sklearn."""
        self.initialize()
        return self.algorithm.fit(x, y)

    def transform(self, x):
        """Generic transform method for partial compatibility to sklearn."""
        return self.algorithm.transform(x)

    def fit_transform(self, x, y):
        """Generic fit_transform method for partial compatibility to sklearn."""
        self.fit(x, y)
        return self.transform(x)

    def load(self, file_name, import_folder = '', prefix = '', suffix = ''):
        """Loads stored ingredient from disc."""
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
        """Saves ingredient to disc."""
        if self.verbose:
            print('Exporting', file_name)
        export_path = self.filer.path_join(folder = export_folder,
                                           prefix = prefix,
                                           file_name = file_name,
                                           suffix = suffix,
                                           file_type = 'pickle')
        pickle.dump(self.algorithm, open(export_path, 'wb'))
        return self