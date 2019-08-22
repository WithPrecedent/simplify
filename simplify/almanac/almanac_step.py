
from dataclasses import dataclass

from ..implements.step import Step


@dataclass
class AlmanacStep(Step):
    """Parent class for preprocessing steps in the siMpLify package."""

    def __post_init__(self):
        self.algorithms = []
        super().__post_init__()
        return self

    def conform(self):
        self.inventory.step = self.__class__.__name__.lower()
        return self

    def _prepare_generic_dict(self):
        self.algorithms.append(self.options[self.technique](**self.parameters))
        return self

    def _prepare_generic_list(self):
        self.algorithms.append(self.options[self.technique](*self.parameters))
        return self

    def _start_generic(self, ingredients, algorithm):
        ingredients.df = algorithm.start(df = ingredients.df,
                                         source = ingredients.source)
        return self

    def prepare(self):
        if isinstance(self.parameters, list):
            for key in self.parameters:
                if hasattr(self, '_prepare_' + self.technique):
                    getattr(self, '_prepare_' + self.technique)(key = key)
                else:
                    getattr(self, '_prepare_generic_list')(key = key)
                self.algorithms.append(
                        self.options[self.technique](**self.parameters))
        elif isinstance(self.parameters, dict):
            for key, value in self.parameters.items():
                if hasattr(self, '_prepare_' + self.technique):
                    getattr(self, '_prepare_' + self.technique)(key = key,
                                                                value = value)
                else:
                    getattr(self, '_prepare_generic_dict')(key = key,
                                                           value = value)
                self.algorithms.append(
                        self.options[self.technique](**self.parameters))
        return self

    def start(self, ingredients):
        for algorithm in self.algorithms:
            if hasattr(self, '_start_' + self.technique):
                ingredients = getattr(
                        self, '_start_' + self.technique)(
                                ingredients = ingredients,
                                algorithm = algorithm)
            else:
                getattr(self, '_start_generic')(
                                ingredients = ingredients,
                                algorithm = algorithm)
        return ingredients