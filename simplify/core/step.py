
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simplify.core.base import SimpleClass


@dataclass
class Step(SimpleClass):
    """Parent class for Almanac, Cookbook, and Review steps in the siMpLify
    package. The class can also be subclassed in the creation of other
    Planner classes."""

    def __post_init__(self):
        super().__post_init__()
        return self

    def _check_parameters(self):
        """Checks if parameters exists. If not, defaults are used. If there
        are no defaults, an empty dict is created for parameters.
        """
        if not hasattr(self, 'parameters') or self.parameters is None:
            if hasattr(self, 'menu') and self.name in self.menu.configuration:
                self.parameters = self.menu.configuration[self.name]
            elif hasattr(self, 'default_parameters'):
                self.parameters = self.default_parameters
            else:
                self.parameters = {}
        if hasattr(self, 'runtime_parameters') and self.runtime_parameters:
            self.parameters.update(self.runtime_parameters)
        return self

    def _define(self):
        if not hasattr(self, 'options'):
            self.options = {}
        if not hasattr(self, 'parameters'):
            self.parameters = {}
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

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
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

    def prepare(self):
        """Adds parameters to algorithm and sets import/export folders."""
        self._check_parameters()
        self._select_parameters()
        if self.technique != 'none':
            self.algorithm = self.options[self.technique](**self.parameters)
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

    def update_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = parameters
            else:
                self.parameters.update(parameters)
            return self
        else:
            error = 'parameters must be a dict type'
            raise TypeError(error)

    def update_runtime_parameters(self, parameters):
        """Adds a runtime parameter set to runtime_parameters dictionary."""
        if isinstance(parameters, dict):
            if (not hasattr(self, 'runtime_parameters')
                    or self.runtime_parameters is None):
                self.runtime_parameters = parameters
            else:
                self.runtime_parameters.update(parameters)
            return self
        else:
            error = 'runtime_parameters must be a dict type'
            raise TypeError(error)


    # def conform(self):
    #     self.inventory.step = self.__class__.__name__.lower()
    #     return self

    # def _prepare_generic_dict(self):
    #     self.algorithms.append(self.options[self.technique](**self.parameters))
    #     return self

    # def _prepare_generic_list(self):
    #     self.algorithms.append(self.options[self.technique](*self.parameters))
    #     return self

    # def _start_generic(self, ingredients, algorithm):
    #     ingredients.df = algorithm.start(df = ingredients.df,
    #                                      source = ingredients.source)
    #     return self

    # def prepare(self):
    #     if isinstance(self.parameters, list):
    #         for key in self.parameters:
    #             if hasattr(self, '_prepare_' + self.technique):
    #                 getattr(self, '_prepare_' + self.technique)(key = key)
    #             else:
    #                 getattr(self, '_prepare_generic_list')(key = key)
    #             self.algorithms.append(
    #                     self.options[self.technique](**self.parameters))
    #     elif isinstance(self.parameters, dict):
    #         for key, value in self.parameters.items():
    #             if hasattr(self, '_prepare_' + self.technique):
    #                 getattr(self, '_prepare_' + self.technique)(key = key,
    #                                                             value = value)
    #             else:
    #                 getattr(self, '_prepare_generic_dict')(key = key,
    #                                                        value = value)
    #             self.algorithms.append(
    #                     self.options[self.technique](**self.parameters))
    #     return self

    # def start(self, ingredients):
    #     for algorithm in self.algorithms:
    #         if hasattr(self, '_start_' + self.technique):
    #             ingredients = getattr(
    #                     self, '_start_' + self.technique)(
    #                             ingredients = ingredients,
    #                             algorithm = algorithm)
    #         else:
    #             getattr(self, '_start_generic')(
    #                             ingredients = ingredients,
    #                             algorithm = algorithm)
    #     return ingredients
