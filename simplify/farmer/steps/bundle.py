
from dataclasses import dataclass

from ..harvest_step import HarvestStep


@dataclass
class Bundle(HarvestStep):
    """Class for combining different datasets."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def plan(self):
        self.options = {'merger' : Merger}
        self.needed_parameters = {'merger' : ['index_columns', 'merge_type']}
        return self

    def prepare(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def perform(self, ingredients):
        ingredients = self.algorithm.perform(ingredients)
        return ingredients

@dataclass
class Merger(object):

    index_columns : object = None
    merge_type : str = ''

    def __post_init__(self):
        return self

    def perform(self, ingredients, sources):
        return ingredients