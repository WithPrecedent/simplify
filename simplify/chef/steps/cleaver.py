"""
.. module:: cleaver
:synopsis: divides features into groups for comparison and combination
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.chef.composer import ChefAlgorithm as Algorithm
from simplify.chef.composer import ChefComposer as Composer
from simplify.chef.composer import ChefTechnique as Technique


@dataclass
class Cleaver(Composer):
    """Divides features for comparison or recombination."""

    name: str = 'cleaver'

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    # def _cleave(self, ingredients):
    #     if self.technique != 'all':
    #         cleave = self.options[self.technique]
    #         drop_list = [i for i in self.test_columns if i not in cleave]
    #         for col in drop_list:
    #             if col in ingredients.x_train.columns:
    #                 ingredients.x_train.drop(col, axis = 'columns',
    #                                          inplace = True)
    #                 ingredients.x_test.drop(col, axis = 'columns',
    #                                         inplace = True)
    #     return ingredients

    # def _publish_cleaves(self):
    #     for group, columns in self.options.items():
    #         self.test_columns.extend(columns)
    #     if self.parameters['include_all']:
    #         self.options.update({'all': self.test_columns})
    #     return self

    # def add(self, cleave_group, columns):
    #     """For the cleavers in siMpLify, this step alows users to manually
    #     add a new cleave group to the cleaver dictionary.
    #     """
    #     self.options.update({cleave_group: columns})
    #     return self

    def draft(self):
        self.compare = Technique(
            name = 'compare',
            module = None,
            algorithm = 'CompareCleaves')
        self.combine = Technique(
            name = 'combine',
            module = None,
            algorithm = 'CombineCleaves')
        super().draft()
        return self


@dataclass
class CompareCleaves(Algorithm):
    """[summary]

    Args:
        technique (str):
        parameters (dict):
        space (dict):
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        self.idea_sections = ['chef']
        super().__post_init__()
        return self


@dataclass
class CombineCleaves(Algorithm):
    """[summary]

    Args:
        technique (str):
        parameters (dict):
        space (dict):
    """
    technique: str
    parameters: object
    space: object

    def __post_init__(self):
        super().__post_init__()
        return self
