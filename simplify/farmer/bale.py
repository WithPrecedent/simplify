"""
.. module:: bale
:synopsis: merges and joins datasets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.iterable import SimpleIterable


@dataclass
class Bale(SimpleIterable):
    """Class for combining different datasets."""
    technique: object = None
    parameters: object = None
    auto_publish: bool = True

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

    def publish(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def implement(self, ingredients):
        ingredients = self.algorithm.implement(ingredients)
        return ingredients
