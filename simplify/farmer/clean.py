
from dataclasses import dataclass

from simplify.core.base import SimplePlan


@dataclass
class Clean(SimplePlan):
    """Cleans, munges, and parsers data using fast, vectorized methods.

    Args:
        steps(dict): dictionary containing keys of SimpleStep names (strings)
            and values of SimpleStep class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_finalize(bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'cleaner'
    auto_finalize: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        self.options = {
                'keyword': ['simplify.core.retool', 'ReTool'],
                'combine': ['simplify.farmer.steps.combine', 'Combine']}
        return self

    def _produce_combiner(self, ingredients):
        ingredients = self.algorithm.produce(ingredients)
        return ingredients

    def _produce_keyword(self, ingredients):
        ingredients.df = self.algorithm.produce(ingredients.df)
        return ingredients

    def produce(self, ingredients):
        ingredients = getattr(self, '_produce_' + self.technique)(ingredients)
        return ingredients
