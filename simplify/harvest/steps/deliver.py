
from dataclasses import dataclass

import pandas as pd

from .harvest_step import HarvestStep
from ...managers import Technique


@dataclass
class Deliver(HarvestStep):

    options : object = None
    almanac : object = None
    auto_prepare : bool = True
    name : str = 'delivery'

    def __post_init__(self):
        self.default_options = {'shapers' : Shaper,
                                'streamliners' : Streamliner}
        super().__post_init__()
        return self

    def _prepare_shapers(self):
        return self

    def _prepare_streamliners(self):
        return self


@dataclass
class Shaper(Technique):

    shape_type : str = ''
    stubs : object = None
    id_column : str = ''
    values : object = None
    separator : str = ''

    def __post_init__(self):
        return self

    def _long(self, df):
        """A simple wrapper method for pandas wide_to_long method using more
        intuitive parameter names than 'i' and 'j'.
        """
        df = (pd.wide_to_long(df,
                              stubnames = self.stubs,
                              i = self.id_column,
                              j = self.values,
                              sep = self.separator).reset_index())
        return df

    def _wide(self, df):
        """A simple wrapper method for pandas pivot method named as
        corresponding method to reshape_long.
        """
        df = (df.pivot(index = self.id_column,
                       columns = self.stubs,
                       values = self.values).reset_index())
        return df


    def start(self, ingredients):
        ingredients.df = getattr(self, '_' + self.shape_type)(ingredients.df)
        return ingredients

@dataclass
class Streamliner(Technique):

    algorithm : object = None

    def __post_init__(self):
        return self

    def start(self, ingredients):
        ingredients = self.algorithm(ingredients)
        return ingredients
