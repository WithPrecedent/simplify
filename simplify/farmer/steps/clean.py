
from dataclasses import dataclass

import numpy as np

from ...retool import ReTool
from ..harvest_step import HarvestStep


@dataclass
class Clean(HarvestStep):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def plan(self):
        self.options = {'keyword' : ReTool,
                        'combiner' : Combine}
        return self

    def _perform_combiner(self, ingredients):
        ingredients = self.algorithm.perform(ingredients)
        return ingredients

    def _perform_keyword(self, ingredients):
        ingredients.df = self.algorithm.perform(ingredients.df)
        return ingredients

    def perform(self, ingredients):
        ingredients = getattr(self, '_perform_' + self.technique)(ingredients)
        return ingredients

@dataclass
class Combine(object):

    method : object
    parameters : object

    def __post_init__(self):
        super().__post_init()
        return self

    def _combine_all(self, ingredients):
        ingredients.df[self.parameters['out_column']] = np.where(
                np.all(ingredients.df[self.parameters['in_columns']]),
                True, False)
        return ingredients

    def _combine_any(self, ingredients):
        ingredients.df[self.parameters['out_column']] = np.where(
                np.any(ingredients.df[self.parameters['in_columns']]),
                True, False)
        return ingredients

    def _dict(self, ingredients):
        ingredients.df[self.parameters['out_column']] = (
                ingredients.df[self.parameters['in_columns']].map(
                        self.method))
        return ingredients

    def plan(self):
        self.options = {'all' : self._combine_all,
                        'any' : self._combine_any,
                        'dict' : self._dict}
        if isinstance(self.method, str):
            self.algorithm = self.options[self.method]
        else:
            self.algorithm = self._dict
        return self

    def perform(self, ingredients):
        self.ingredients = self.algorithm(ingredients)
        return ingredients