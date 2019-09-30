
from dataclasses import dataclass

from simplify.core.base import SimplePlan


@dataclass
class Bale(SimplePlan):
    """Class for combining different datasets."""
    technique: str = ''
    parameters: object = None
    auto_finalize: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
                'merge': ['simplify.farmer.steps.merge', 'Merge'],
                'supplement': ['simplify.farmer.steps.supplement',
                               'Supplement']}
        self.needed_parameters = {'merger': ['index_columns', 'merge_type']}
        return self

    def finalize(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients):
        ingredients = self.algorithm.produce(ingredients)
        return ingredients
