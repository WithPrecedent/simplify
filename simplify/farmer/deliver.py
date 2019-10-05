"""
.. module:: deliver
:synopsis: completes data gathering and preparation process
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import pandas as pd

from simplify.core.iterables import SimplePlan


@dataclass
class Deliver(SimplePlan):
    """Makes final structural changes to data before analysis.

    Args:
        steps(dict): dictionary containing keys of SimpleStep names (strings)
            and values of SimpleStep class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_publish(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'delivery'
    auto_publish: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _publish_shapers(self, harvest):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def _publish_streamliners(self, harvest):
        self.algorithm = self.options[self.technique](**self.parameters)
        return self

    def draft(self):
        self.options = {
                'reshape': ['simplify.farmer.steps.reshape', 'Reshape'],
                'streamline': ['simplify.farmer.steps.streamline',
                               'Streamline']}
        self.needed_parameters = {'shapers': ['shape_type', 'stubs',
                                               'id_column', 'values',
                                               'separator'],
                                  'streamliners': ['method']}
        return self

    def implement(self, ingredients):
        ingredients = self.algorithm.implement(ingredients)
        return ingredients
