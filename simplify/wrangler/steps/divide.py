"""
.. module:: divide
:synopsis: divides data source files into manageable sizes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass
import os

from simplify.core.definitionsetter import WranglerTechnique


@dataclasses.dataclass
class Divide(WranglerTechnique):
    """Divides data source files so that they can be loaded in memory.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'converter'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def publish(self, dataset):
        return self
