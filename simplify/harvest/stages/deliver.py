
from dataclasses import dataclass

import pandas as pd

from .stage import Stage, Technique


@dataclass
class Deliver(Stage):

    technique : str = ''
    parameters : object = None
    name : str = 'clean'
    auto_prepare : bool = True

    def __post_init__(self):
        self.techniques = {'shapers' : Shaper,
                           'streamliners' : Streamliner}
        super().__post_init__()
        return self

@dataclass
class Shaper(Technique):

    shape_type : str = ''
    stubs : object = None
    id_column : str = ''
    values : object = None
    separator : str = ''
    auto_shape : bool = False
    df : object = None

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

    def start(self, df):
        return df

@dataclass
class Streamliner(Technique):

    technique : str = ''

    def __post_init__(self):
        return self

    def start(self, ingredients):
        return ingredients
