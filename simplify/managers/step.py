
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd

from ..implements import listify


@dataclass
class Step(object):
    """Parent class for preprocessing and modeling techniques in the siMpLify
    package."""

    def __post_init__(self):
        if hasattr(self, '_set_defaults'):
            self._set_defaults()
        if hasattr(self, 'auto_prepare') and self.auto_prepare:
            self.prepare()
        return self

    def __contains__(self, technique):
        """Checks whether technique is listed in techniques dictionary."""
        if technique in self.options:
            return True
        else:
            return False

    def __delitem__(self, technique):
        """Deletes technique and algorithm if technique is in options
        dictionary.
        """
        if technique in self.options:
            self.options.pop(technique)
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
        return self

    def __getitem__(self, technique):
        """Gets algorithm if technique is in options dictionary."""
        if technique in self.options:
            return self.options[technique]
        else:
            error = technique + ' is not in ' + self.__class__.__name__
            raise KeyError(error)
            return

    def __setitem__(self, technique, algorithm):
        """Adds technique and algorithm to options dictionary."""
        if isinstance(technique, str):
            if isinstance(algorithm, object):
                self.options.update({technique : algorithm})
            else:
                error = technique + ' must be an algorithm of object type'
                raise TypeError(error)
        else:
            error = technique + ' must be a string type'
            raise TypeError(error)
        return self

    def _check_lengths(self, variable1, variable2):
        """Checks lists to ensure they are of equal length."""
        if len(listify(variable1) != listify(variable1)):
            error = 'Lists must be of equal length'
            raise RuntimeError(error)
            return self
        else:
            return True

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used. If there
        are no defaults, an empty dict is created for parameters.
        """
        if not hasattr(self, 'parameters') or self.parameters == None:
            if hasattr(self, 'menu') and self.name in self.menu.config:
                self.parameters = self.menu.config[self.name]
            elif hasattr(self, 'default_parameters'):
                self.parameters = self.default_parameters
            else:
                self.parameters = {}
        return self

    def _check_runtime_parameters(self):
        """Checks if class has runtime_parameters and, if so, adds them to
        the parameters attribute.
        """
        if hasattr(self, 'runtime_parameters') and self.runtime_parameters:
            self.parameters.update(self.runtime_parameters)
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

    def _list_type(self, test_list, data_type):
        """Tests whether any item in a list is of the passed data type."""
        return any(isinstance(i, data_type) for i in test_list)

    def _prepare_generic(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def _select_parameters(self, parameters_to_use = None):
        """For subclasses that only need a subset of the parameters stored in
        menu, this function selects that subset.
        """
        if hasattr(self, 'selected_parameters') and self.selected_parameters:
            if not parameters_to_use:
                parameters_to_use = list(self.default_parameters.keys())
            new_parameters = {}
            if self.parameters:
                for key, value in self.parameters.items():
                    if key in self.default_parameters:
                        new_parameters.update({key : value})
                self.parameters = new_parameters
        return self

    def _start_generic(self, ingredients):
        ingredients = self.algorithm.start(ingredients)
        return ingredients

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def add_options(self, techniques, algorithms):
        """Adds new technique name and corresponding algorithm to the
        techniques dictionary.
        """
        if self._check_lengths(techniques, algorithms):
            if getattr(self, 'options') == None:
                self.options = dict(zip(listify(techniques),
                                        listify(algorithms)))
            else:
                self.options.update(dict(zip(listify(techniques),
                                             listify(algorithms))))
            return self

    def add_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters == None:
                self.parameters = parameters
            else:
                self.parameters.update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)
            return self

    def add_runtime_parameters(self, parameters):
        """Adds a parameter set to runtime_parameters dictionary."""
        if isinstance(parameters, dict):
            if (not hasattr(self, 'runtime_parameters')
                    or self.runtime_parameters == None):
                self.runtime_parameters = parameters
            else:
                self.runtime_parameters.update(parameters)
            return self
        else:
            error = 'runtime_parameters must be a dict type'
            raise TypeError(error)
            return self

    def fit(self, x, y = None):
        """Generic fit method for partial compatibility to sklearn."""
        self.prepare()
        if isinstance(y, pd.Series):
            return self.algorithm.fit(x, y)
        else:
            return self.algorithm.fit(x)

    def fit_transform(self, x, y = None):
        """Generic fit_transform method for partial compatibility to sklearn.
        """
        self.fit(x, y)
        x = self.transform(x)
        return x

    def load(self, file_name, folder = '', prefix = '', suffix = ''):
        """Loads stored ingredient from disc."""
        if self.verbose:
            print('Importing', file_name)
        file_path = self.inventory.create_path(folder = folder,
                                               prefix = prefix,
                                               file_name = file_name,
                                               suffix = suffix,
                                               file_type = 'pickle')
        self.algorithm = pickle.load(open(file_path, 'rb'))
        return self

    def prepare(self):
        """Adds parameters to algorithm."""
        self._check_parameters()
        self._select_parameters()
        self._check_runtime_parameters()
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def save(self, file_name, folder = '', prefix = '', suffix = ''):
        """Saves step to disc."""
        if self.verbose:
            print('Exporting', file_name)
        file_path = self.inventory.create_path(folder = folder,
                                               prefix = prefix,
                                               file_name = file_name,
                                               suffix = suffix,
                                               file_type = 'pickle')
        pickle.dump(self.algorithm, open(file_path, 'wb'))
        return self

    def start(self, ingredients, recipe):
        """Generic implement method for adding ingredients into recipe and
        applying the appropriate algorithm.
        """
        if self.technique != 'none':
            self.algorithm.fit(ingredients.x, ingredients.y)
            ingredients.x = self.algorithm.transform(ingredients.x)
        return ingredients

    def transform(self, x):
        """Generic transform method for partial compatibility to sklearn."""
        x = self.algorithm.transform(x)
        return x