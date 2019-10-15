"""
.. module:: cleave
:synopsis: divides features into groups for comparison and combination
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
DEFAULT_OPTIONS = {}


@dataclass
class Cleave(SimpleTechnique):
    """Stores different groups of features (to allow comparison among those
    groups).

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
    name: str = 'cleave'
    auto_publish: bool = True
    options: Dict = field(default_factory = lambda: DEFAULT_OPTIONS)

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    def _cleave(self, ingredients):
        if self.technique != 'all':
            cleave = self.options[self.technique]
            drop_list = [i for i in self.test_columns if i not in cleave]
            for col in drop_list:
                if col in ingredients.x_train.columns:
                    ingredients.x_train.drop(col, axis = 'columns',
                                             inplace = True)
                    ingredients.x_test.drop(col, axis = 'columns',
                                            inplace = True)
        return ingredients

    def _publish_cleaves(self):
        for group, columns in self.options.items():
            self.test_columns.extend(columns)
        if self.parameters['include_all']:
            self.options.update({'all': self.test_columns})
        return self

    def add(self, cleave_group, columns):
        """For the cleavers in siMpLify, this step alows users to manually
        add a new cleave group to the cleaver dictionary.
        """
        self.options.update({cleave_group: columns})
        return self

    def draft(self):
        self.options = {
                'compare': ['simplify.chef.steps.techniques.cleavers',
                            'CompareCleaves'],

                'combine': ['simplify.chef.steps.techniques.cleavers',
                            'CombineCleaves']}
        return self

    def publish(self):
        super().publish()
        self.algorithm = self._cleave
        return self

    @numpy_shield
    def implement(self, ingredients, plan = None):
        self._publish_cleaves()
        ingredients = self.algorithm(ingredients)
        return ingredients