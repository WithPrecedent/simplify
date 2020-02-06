"""
.. module:: supplement
:synopsis: adds new data to existing DataFrame
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.definitionsetter import WranglerTechnique


@dataclass
class Supplement(WranglerTechnique):
    """Adds new data to similarly structured DataFrame.

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
    name: str = 'supplementer'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        return self

    def draft(self) -> None:
        self._options = Repository(options = {}
        return self

    def publish(self, dataset, sources):
        return dataset