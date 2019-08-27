
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..step import Step


@dataclass
class CookbookStep(Step):
    """Parent class for preprocessing and modeling steps in the siMpLify
    package."""

    def __post_init__(self):
        super().__post_init__()
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

    def _store_feature_names(self, x, y = None):
        """Stores feature names."""
        self.x_cols = list(x.columns.values)
        if isinstance(y, pd.Series):
            self.y_col = self.label
        return self

    def conform(self):
        self.inventory.step = 'cook'
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
        self._set_folders()
        self.conform()
        self._check_parameters()
        self._select_parameters()
        self._check_runtime_parameters()
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