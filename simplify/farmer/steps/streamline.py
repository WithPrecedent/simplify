"""
.. module:: streamline
:synopsis: last-stage data processing before analysis
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.typesetter import FarmerTechnique


@dataclass
class Streamline(FarmerTechnique):
    """Combines, divides, and otherwise prepares features for analysis.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance and
            other methods and classes in the siMpLify package..
        auto_draft (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'scaler'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def publish(self, ingredients):
        data = self.method(ingredients)
        return ingredients