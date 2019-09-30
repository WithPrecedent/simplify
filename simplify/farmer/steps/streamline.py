
from dataclasses import dataclass

import pandas as pd

from simplify.core.base import SimpleStep


@dataclass
class Streamline(SimpleStep):
    """Combines, divides, and otherwise prepares features for analysis.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: str = ''
    parameters: object = None
    name: str = 'scaler'
    auto_finalize: bool = True

    def __post_init__(self):
        return self

    def produce(self, ingredients):
        ingredients = self.method(ingredients)
        return ingredients