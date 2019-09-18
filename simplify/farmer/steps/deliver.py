
from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleStep


@dataclass
class Deliver(SimpleStep):

    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _finalize_shapers(self, harvest):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def _finalize_streamliners(self, harvest):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def draft(self):
        self.options = {'shapers' : Shaper,
                        'streamliners' : Streamliner}
        self.needed_parameters = {'shapers' : ['shape_type', 'stubs',
                                               'id_column', 'values',
                                               'separator'],
                                  'streamliners' : ['method']}
        return self

    def produce(self, ingredients):
        ingredients = self.algorithm.produce(ingredients)
        return ingredients

@dataclass
class Shaper(object):

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


    def produce(self, ingredients):
        ingredients.df = getattr(self, '_' + self.shape_type)(ingredients.df)
        return ingredients

@dataclass
class Streamliner(object):

    method : object = None

    def __post_init__(self):
        return self

    def produce(self, ingredients):
        ingredients = self.method(ingredients)
        return ingredients
