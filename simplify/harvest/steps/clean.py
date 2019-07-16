
from dataclasses import dataclass
import os

import numpy as np

from ...implements import ReFrame
from ...managers import Step, Technique


@dataclass
class Clean(Step):

    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'cleaner'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options = {'keyword' : ReFrame,
                        'combiner' : Combine}
        return self

    def _prepare_combiner(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def _prepare_keyword(self):
        file_path = os.path.join(self.inventory.keywords,
                                 self.parameters['section'] + '.csv')
        self.update.parameters(
                {'file_path' : file_path},
                {'out_prefix' : self.parameters['section'] + '_'})
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def _start_combiner(self, ingredients):
        ingredients = self.algorithm.start(ingredients)
        return ingredients

    def _start_keyword(self, ingredients):
        ingredients.df = self.algorithm.match(ingredients.df)
        return ingredients

    def prepare(self):
        getattr(self, '_prepare_' + self.technique)()
        return self

    def start(self, ingredients):
        ingredients = getattr(self, '_start_' + self.technique)(ingredients)
        return ingredients

@dataclass
class Combine(Technique):

    in_columns : object = None
    out_column : str = ''
    algorithm : object = None

    def __post_init__(self):
        return self

    def _combine_all(self, ingredients):
        ingredients.df[self.out_column] = np.where(
                np.all(ingredients.df[self.in_columns]), True, False)
        return self

    def _combine_any(self, ingredients):
        ingredients.df[self.out_column] = np.where(
                np.any(ingredients.df[self.in_columns]), True, False)
        return self

    def prepare(self):
        if isinstance(self.algorithm, str):
            self.algorithm = getattr(self, '_combine_' + self.algorithm)
        return self

    def start(self, ingredients):
        self.ingredients = self.algorithm(ingredients)
        return ingredients