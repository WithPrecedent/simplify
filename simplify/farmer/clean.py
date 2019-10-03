"""
.. module:: almanac
:synopsis: munges and cleans pandas DataFrames using vectorized methods
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.plan import SimplePlan


@dataclass
class Clean(SimplePlan):
    """Cleans, munges, and parsers data using fast, vectorized methods.

    Args:
        steps(dict): dictionary containing keys of SimpleStep names (strings)
            and values of SimpleStep class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'cleaner'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
                'keyword': ['simplify.core.retool', 'ReTool'],
                'combine': ['simplify.farmer.steps.combine', 'Combine']}
        return self

    def _implement_combiner(self, ingredients):
        ingredients = self.algorithm.implement(ingredients)
        return ingredients

    def _implement_keyword(self, ingredients):
        ingredients.df = self.algorithm.implement(ingredients.df)
        return ingredients

    def implement(self, ingredients):
        ingredients = getattr(self, '_implement_' + self.technique)(ingredients)
        return ingredients
