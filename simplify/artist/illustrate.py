"""
.. module:: illustrate
:synopsis: visualizations for data summary
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

#from simplify.core.decorators import localize
from simplify.core.technique import ArtistTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'histogram': ['seaborn', 'distplot']}


@dataclass
class Illustrate(SimpleIterable):
    """Creates data summary visualizations.

    Args:
        techniques(dict(str: ArtistTechnique)): names and related ArtistTechnique
            classes for creating data visualizations.
        name(str): designates the name of the class which should be identical
            to the section of the idea configuration with relevant settings.
        auto_draft (bool): whether to call the 'publish' method when the
            class is instanced.
        auto_publish (bool): whether to call the 'publish' method when the
            class is instanced.
    """
    techniques: object = None
    name: str = 'illustrator'
    auto_draft: bool = True
    auto_publish: bool = True

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
        self.options =
        return self

    def publish(self):
        return self

    #@localize
    def implement(self, recipes = None, reviews = None):
        for step in self.techniques:
            getattr(self, step)()
        return self
