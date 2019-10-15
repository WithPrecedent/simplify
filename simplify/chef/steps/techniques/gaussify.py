"""
.. module:: gaussify
:synopsis: adaptive method for fitting data to gaussian distribution
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.technique import SimpleTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'box-cox': ['sklearn.preprocessing', 'PowerTransformer'],
    'yeo-johnson': ['sklearn.preprocessing', 'PowerTransformer']}

@dataclass
class Gaussify(SimpleTechnique):
    """Transforms data columns to more gaussian distribution.

    The particular method applied is chosen between 'box-cox' and 'yeo-johnson'
    based on whether the particular data column has values below zero.

    Args:
        technique(str): name of technique used.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_publish(bool): whether 'fina
        lize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: str = 'box-cox and yeo-johnson'
    parameters: object = None
    name: str = 'gaussifier'
    auto_publish: bool = True

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def draft(self):
        return self

    def publish(self):
        self.rescaler = self.parameters['rescaler'](
                copy = self.parameters['copy'])
        del self.parameters['rescaler']
        self._publish_parameters()
        self.positive_tool = self.options['box_cox'](
                method = 'box_cox', **self.parameters)
        self.negative_tool = self.options['yeo_johnson'](
                method = 'yeo_johnson', **self.parameters)
        return self

    def implement(self, ingredients, columns = None):
        if not columns:
            columns = ingredients.numerics
        for column in columns:
            if ingredients.x[column].min() >= 0:
                ingredients.x[column] = self.positive_tool.fit_transform(
                        ingredients.x[column])
            else:
                ingredients.x[column] = self.negative_tool.fit_transform(
                        ingredients.x[column])
            ingredients.x[column] = self.rescaler.fit_transform(
                    ingredients.x[column])
        return ingredients