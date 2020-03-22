"""
.. module:: bale
:synopsis: merges and joins datasets
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses.dataclasses import dataclasses.dataclass

from simplify.core.definitionsetter import WranglerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleDirector
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleDirector subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'merge': ['simplify.wrangler.steps.merge', 'Merge'],
    'supplement': ['simplify.wrangler.steps.supplement', 'Supplement']}


@dataclasses.dataclass
class Bale(SimpleIterable):
    """Class for combining different datasets."""
    step: object = None
    parameters: object = None
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self.needed_parameters = {'merger': ['index_columns', 'merge_type']}
        return self

    def publish(self):
        self.algorithm = self.workers[self.step](**self.parameters)
        return self

    def publish(self, dataset):
        data = self.algorithm.implement(dataset)
        return dataset
