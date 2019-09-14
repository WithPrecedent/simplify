
from dataclasses import dataclass

from .base import SimpleClass


@dataclass
class Technique(SimpleClass):

    def __post_init__(self):
        super().__post_init__()
        return self

    def fit(self):
        self.finalize()
        return self

    def fit_transform(self, ingredients):
        self.fit()
        ingredients = self.transform(ingredients = ingredients)
        return ingredients

    def finalize(self):
        self.tool = self.options[self.technique](**self.parameters)
        return self

    def produce(self, ingredients):
        ingredients = self.tool(ingredients)
        return ingredients

    def transform(self, ingredients):
        ingredients = self.produce(ingredients = ingredients)
        return ingredients