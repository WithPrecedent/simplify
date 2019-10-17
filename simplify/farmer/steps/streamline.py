"""
.. module:: streamline
:synopsis: last-stage data processing before analysis
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.technique import FarmerTechnique


@dataclass
class Streamline(FarmerTechnique):
    """Combines, divides, and otherwise prepares features for analysis.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'scaler'
    auto_publish: bool = True

    def __post_init__(self):
        return self

    def implement(self, ingredients):
        ingredients = self.method(ingredients)
        return ingredients