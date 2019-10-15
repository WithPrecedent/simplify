"""
.. module:: scale
:synopsis: scales or bins numerical features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass, field
from typing import Dict

from simplify.core.technique import SimpleTechnique
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
    'bins': ['sklearn.preprocessing', 'KBinsDiscretizer'],
    'gauss': ['simplify.chef.steps.techniques.gaussify', 'Gaussify'],
    'maxabs': ['sklearn.preprocessing', 'MaxAbsScaler'],
    'minmax': ['sklearn.preprocessing', 'MinMaxScaler'],
    'normalize': ['sklearn.preprocessing', 'Normalizer'],
    'quantile': ['sklearn.preprocessing', 'QuantileTransformer'],
    'robust': ['sklearn.preprocessing', 'RobustScaler'],
    'standard': ['sklearn.preprocessing', 'StandardScaler']}


@dataclass
class Scale(SimpleTechnique):
    """Scales numerical data according to selected algorithm.

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
    name: str = 'scale'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    """ Core siMpLify Public Methods """

    def draft(self):
        super().draft()
        self.default_parameters = {
            'bins': {'encode': 'ordinal',
                        'strategy': 'uniform',
                        'n_bins': 5},
            'gauss': {'standardize': False,
                        'copy': False},
            'maxabs': {'copy': False},
            'minmax': {'copy': False},
            'normalize': {'copy': False},
            'quantile': {'copy': False},
            'robust': {'copy': False},
            'standard': {'copy': False}}
        self.selected_parameters = True
        self.custom_options = ['gauss']
        return self

    def publish(self):
        super().publish()
        return self

    @numpy_shield
    def implement(self, ingredients, plan = None, columns = None):
        if columns is None:
            columns = ingredients.scalers
        if self.technique in self.custom_options:
            ingredients = self.algorithm.implement(ingredients = ingredients,
                                                 columns = columns)
        else:
            ingredients.x[columns] = self.algorithm.fit_transform(
                    ingredients.x[columns], ingredients.y)
        return ingredients
