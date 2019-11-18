"""
.. module:: cleaver
:synopsis: divides features into groups for comparison and combination
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.contributor import Algorithm
from simplify.core.contributor import Outline
from simplify.core.contributor import SimpleContributor


@dataclass
class Cleaver(SimpleContributor):
    """Divides features for comparison or recombination.

    Args:
        name (Optional[str]): designates the name of the class used for internal
            referencing throughout siMpLify. If the class needs settings from
            the shared Idea instance, 'name' should match the appropriate
            section name in Idea. When subclassing, it is a good idea to use
            the same 'name' attribute as the base class for effective
            coordination between siMpLify classes. 'name' is used instead of
            __class__.__name__ to make such subclassing easier. If 'name' is not
            provided, __class__.__name__.lower() is used instead.

    """

    name: str = 'cleaver'

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self

    # def _cleave(self, ingredients):
    #     if self.step != 'all':
    #         cleave = self.options[self.step]
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

    def draft(self) -> None:
        super().draft()
        self.options = {
        'compare': Outline(
            name = 'compare',
            module = None,
            algorithm = 'CompareCleaves'),
        'combine': Outline(
            name = 'combine',
            module = None,
            algorithm = 'CombineCleaves')}

        return self


@dataclass
class CompareCleaves(Algorithm):
    """[summary]

    Args:
        step (str):
        parameters (dict):
        space (dict):
    """
    step: str
    parameters: object
    space: object

    def __post_init__(self) -> None:
        self.idea_sections = ['chef']
        super().__post_init__()
        return self


@dataclass
class CombineCleaves(Algorithm):
    """[summary]

    Args:
        step (str):
        parameters (dict):
        space (dict):
    """
    step: str
    parameters: object
    space: object

    def __post_init__(self) -> None:
        super().__post_init__()
        return self
