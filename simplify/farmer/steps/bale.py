
from dataclasses import dataclass

from simplify.core.step import Step
from simplify.core.technique import Technique


@dataclass
class Bale(Step):
    """Class for combining different datasets."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {'merger' : Merger}
        self.needed_parameters = {'merger' : ['index_columns', 'merge_type']}
        return self

    def finalize(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients):
        ingredients = self.algorithm.produce(ingredients)
        return ingredients

@dataclass
class Merger(Technique):

    index_columns : object = None
    merge_type : str = ''

    def __post_init__(self):
        return self

    def produce(self, ingredients, sources):
        return ingredients