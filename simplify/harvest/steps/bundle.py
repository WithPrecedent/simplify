
from dataclasses import dataclass

from ...managers import Step, Technique


@dataclass
class Bundle(Step):
    """Class for combining different datasets."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'bundler'

    def __post_init__(self):
        super().__post_init__()
        return self

    def _set_defaults(self):
        self.options = {'mergers' : Merger}
        return self

    def prepare(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.algorithm.start(ingredients)
        return ingredients

@dataclass
class Merger(Technique):

    index_columns : object = None
    merge_type : str = ''

    def __post_init__(self):
        return self

    def start(self, ingredients, sources):
        return ingredients