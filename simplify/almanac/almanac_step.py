
from dataclasses import dataclass

from ..implements.step import Step


@dataclass
class AlmanacStep(Step):
    """Parent class for preprocessing steps in the siMpLify package."""

    def __post_init__(self):
        super().__post_init()
        return self

    def conform(self):
        self.inventory.step = self.__class__.__name__.lower()
        return self

    def prepare(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def start(self, ingredients):
        ingredients = self.algorithm.start(ingredients)
        return ingredients