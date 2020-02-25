"""
.. module:: manual
:synopsis: munges and cleans pandas DataFrames using vectorized methods
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses import dataclass

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
    'keyword': ['simplify.core.retool', 'ReTool'],
    'combine': ['simplify.wrangler.steps.combine', 'Combine']}


@dataclass
class Clean(SimpleIterable):
    """Cleans, munges, and parsers data using fast, vectorized methods.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'cleaner'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        return self

    def _implement_combiner(self, dataset):
        data = self.algorithm.implement(dataset)
        return dataset

    def _implement_keyword(self, dataset):
        dataset.df = self.algorithm.implement(dataset.df)
        return dataset

    def publish(self, dataset):
        data = getattr(self, '_implement_' + self.step)(dataset)
        return dataset
