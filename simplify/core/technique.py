
from dataclasses import dataclass

import pandas as pd

from simplify.base.ingredients import Ingredients
from simplify.base.manager import SimpleManager


@dataclass
class Technique(SimpleManager):

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {}
        return self
    
    def finalize(self):
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients

    """ Scikit-Learn Compatibility Methods """
    
    def fit(self, x, y = None):
        if hasattr(self.algorithm, 'fit'):
            if isinstance(x, pd.DataFrame):
                if y is None:
                    self.algorithm.fit(x)
                else:
                    self.algorithm.fit(x, y)
            elif isinstance(x, Ingredients):
                self.algorithm.fit(Ingredients.x_train)
        else:
            self.finalize()
        return self

    def fit_transform(self, x, y = None):
        self.fit(x = x, y = y)
        self.transform(x = x, y = y)
        return x

    def transform(self, x, y = None):
        if hasattr(self.algorithm, 'transform'):
            if isinstance(x, pd.DataFrame):           
                if y:
                    x = self.algorithm.transform(x, y)
                else:
                    x = self.algorithm.transform(x)
            elif isinstance(x, Ingredients):
                x = self.produce(ingredients = x)
        else:
            x = self.produce(ingredients = x)
        return x