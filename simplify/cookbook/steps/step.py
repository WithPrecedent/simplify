
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle

from ...implements.implement import Implement

@dataclass
class Step(Implement):
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
        """Checks whether technique is listed in techniques dictionary."""
        if technique in self.techniques:
            return True
        else:
            return False

    def __delitem__(self, technique):
        """Deletes technique and algorithm if technique is in techniques
        dictionary.
        """
        if technique in self.techniques:
            self.techniques.pop(technique)
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getitem__(self, technique):
        """Gets algorithm if technique is in techniques dictionary."""
        if technique in self.techniques:
            return self.techniques[technique]
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
            return

    def __setitem__(self, technique, algorithm):
        """Adds technique and algorithm to techniques dictionary."""
        if isinstance(technique, str):
            if isinstance(algorithm, object):
                self.techniques.update({technique : algorithm})
            else:
                error = technique + ' must be an algorithm of object type'
                raise TypeError(error)
        else:
            error = technique + ' must be a string type'
            raise TypeError(error)
        return self

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used."""
        if not self.parameters:
            self.parameters = self.defaults
        return self

    def _get_feature_names(self, x, y = None):
        """Gets feature names if previously stored by _store_feature_names."""
        x = pd.DataFrame(x, columns = self.x_cols)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name = self.y_col)
            return x, y
        else:
            return x

    def _get_indices(self, df, columns):
        """Gets column indices for a list of column names."""
        return [df.columns.get_loc(col) for col in columns]

    def _initialize(self, select_parameters = False):
        """Adds parameters to algorithm."""
        self._check_parameters()
        if select_parameters:
            self._select_parameters(
                    parameters_to_use = list(self.defaults.keys()))
        if self.runtime_parameters:
            self.parameters.update(self.runtime_parameters)
        if self.technique != 'none':
            self.algorithm = self.techniques[self.technique]
            if self.parameters:
                self.algorithm = self.algorithm(**self.parameters)
            else:
                self.algorithm = self.algorithm()
        return self

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _select_parameters(self, parameters_to_use = []):
        """For subclasses that only need a subset of the parameters stored in
        settings, this function selects that subset.
        """
        new_parameters = {}
        if self.parameters:
            for key, value in self.parameters.items():
                if key in parameters_to_use:
                    new_parameters.update({key : value})
            self.parameters = new_parameters
        return self

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def add_parameters(self, parameters):
        """Adds a param set to parameters dictionary."""
        if isinstance(parameters, dict):
            return self.parameters.update(parameters)
        else:
            error = 'parameters must be a dictionary type'
            raise TypeError(error)
            return self

    def add_techiques(self, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        new_algorithms = zip(self._listify(techniques),
                             self._listify(algorithms))
        for technique, algorithm in new_algorithms.items():
            self.techniques.update({technique : algorithm})
        return self

    def blend(self, x, y = None):
        """Generic blend method for adding ingredients into recipe and applying
        the appropriate algorithm.
        """
        self.initialize()
        if self.algorithm != 'none':
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return x

    def fit(self, x, y):
        """Generic fit method for partial compatibility to sklearn."""
        self.initialize()
        return self.algorithm.fit(x, y)

    def fit_transform(self, x, y):
        """Generic fit_transform method for partial compatibility to sklearn.
        """
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

    def transform(self, x):
        """Generic transform method for partial compatibility to sklearn."""
        return self.algorithm.transform(x)