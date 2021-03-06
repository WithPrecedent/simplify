"""
.. module:: streamline
:synopsis: last-worker data processing before analysis
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass

import pandas as pd

from simplify.core.definitionsetter import WranglerTechnique


@dataclasses.dataclass
class Streamline(WranglerTechnique):
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

    def publish(self, dataset):
        data = self.method(dataset)
        return dataset