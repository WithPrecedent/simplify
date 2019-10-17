"""
.. module:: encode
:synopsis: converts categorical features to numeric ones
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import ChefTechnique
from simplify.core.decorators import numpy_shield


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'backward': ['category_encoders', 'BackwardDifferenceEncoder'],
    'basen': ['category_encoders', 'BaseNEncoder'],
    'binary': ['category_encoders', 'BinaryEncoder'],
    'dummy': ['category_encoders', 'OneHotEncoder'],
    'hashing': ['category_encoders', 'HashingEncoder'],
    'helmert': ['category_encoders', 'HelmertEncoder'],
    'loo': ['category_encoders', 'LeaveOneOutEncoder'],
    'ordinal': ['category_encoders', 'OrdinalEncoder'],
    'sum': ['category_encoders', 'SumEncoder'],
    'target': ['category_encoders', 'TargetEncoder']}


@dataclass
class Encode(ChefTechnique):
    """Encodes categorical variables according to a selected algorithm.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish (bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'encode'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        return self

    def publish(self):
        pass

    @numpy_shield
    def implement(self, ingredients, plan = None, columns = None):
        if columns is None:
            columns = ingredients.encoders
        if columns:
            self.runtime_parameters.update({'cols': columns})
        super().publish()
        self.algorithm.fit(ingredients.x, ingredients.y)
        self.algorithm.transform(
                ingredients.x_train).reset_index(drop = True)
        self.algorithm.transform(
                ingredients.x_test).reset_index(drop = True)
        return ingredients