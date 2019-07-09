
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...implements import listify
from ...managers import Step


@dataclass
class CookbookStep(Step):
    """Parent class for modeling techniques in the siMpLify package."""

    def _check_lengths(self, variable1, variable2):
        """Checks lists to ensure they are of equal length."""
        if len(listify(variable1) != listify(variable1)):
            error = 'Lists must be of equal length'
            raise RuntimeError(error)
            return self
        else:
            return True

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

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def add_runtime_parameters(self, parameters):
        """Adds a parameter set to parameters dictionary."""
        if isinstance(parameters, dict):
            if not hasattr(self.runtime_parameters):
                self.runtime_parameters = {}
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

    def prepare(self):
        """Adds parameters to algorithm."""
        self._check_parameters()
        self._select_parameters()
        self._check_runtime_parameters()
        if self.technique != 'none':
            self.algorithm = self.techniques[self.technique](**self.parameters)
        return self

    def start(self, ingredients, recipe):
        """Generic implement method for adding ingredients into recipe and
        applying the appropriate algorithm.
        """
        if self.technique != 'none':
            x, y = ingredients[recipe.data_to_use]
            self.algorithm.fit(x, y)
            x = self.algorithm.transform(x)
        return ingredients

    def transform(self, x):
        """Generic transform method for partial compatibility to sklearn."""
        x = self.algorithm.transform(x)
        return x