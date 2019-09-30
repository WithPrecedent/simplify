"""
.. module:: illustrate
:synopsis: visualizations for data summary
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from simplify.core.base import SimplePlan
from simplify.core.decorators import localize


@dataclass
class Illustrate(SimplePlan):
    """Creates data summary visualizations.

    Args:
        steps(dict(str: SimpleStep)): names and related SimpleStep classes for
            creating data visualizations.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_finalize (bool): whether to call the 'finalize' method when the
            class is instanced.
        auto_produce (bool): whether to call the 'produce' method when the class
            is instanced.
    """
    steps: object = None
    name: str = 'illustrator'
    auto_finalize: bool = True
    auto_produce: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    """ Public Tool Methods """
    
    def histogram(self, features = None, file_name = 'histogram.png',
                  **kwargs):
        for feature in self.listify(features):
            self.options['histogram'](self.x[feature], feature, **kwargs)
            self.save(feature + '_' + file_name)
        return self

    """ Public Input/Output Methods """
    
    def save(self, file_name):
        self.depot.save(
            variable = plt,
            folder = self.depot.recipe,
            file_name = file_name,
            file_format = 'png')
        return self

    """ Core siMpLify Methods """
    
    def draft(self):
        self.options = {
            'histogram' : ['seaborn', 'distplot']}
        return self
    
    def finalize(self):
        return self

    @localize
    def produce(self, recipes = None, reviews = None):
        for step in self.steps:
            getattr(self, step)()
        return self
