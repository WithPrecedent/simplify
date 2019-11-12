"""
.. module:: bale
:synopsis: merges and joins datasets
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import FarmerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleComposite
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleComposite subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'merge': ['simplify.farmer.techniques.merge', 'Merge'],
    'supplement': ['simplify.farmer.techniques.supplement', 'Supplement']}


@dataclass
class Bale(SimpleIterable):
    """Class for combining different datasets."""
    technique: object = None
    parameters: object = None
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self.needed_parameters = {'merger': ['index_columns', 'merge_type']}
        return self

    def publish(self):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def publish(self, ingredients):
        ingredients = self.algorithm.implement(ingredients)
        return ingredients
