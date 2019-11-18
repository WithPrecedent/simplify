"""
.. module:: mixer
:synopsis: combines features into new features
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from simplify.core.contributor import SimpleContributor
from simplify.core.contributor import Outline


@dataclass
class Mixer(SimpleContributor):
    """Computes new features by combining existing ones.

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

    name: str = 'mixer'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
        'polynomial': Outline(
            name = 'polynomial_mixer',
            module = 'sklearn.preprocessing',
            algorithm = 'PolynomialFeatures',
            default = {
                'degree': 2,
                'interaction_only': True,
                'include_bias': True}),
        'quotient': Outline(
            name = 'quotient',
            module = None,
            algorithm = 'QuotientFeatures'),
        'sum': Outline(
            name = 'sum',
            module = None,
            algorithm = 'SumFeatures'),
        'difference': Outline(
            name = 'difference',
            module = None,
            algorithm = 'DifferenceFeatures')}
        return self


# @dataclass
# class DifferenceFeatures(Algorithm):
#     """[summary]

#     Args:
#         step (str):
#         parameters (dict):
#         space (dict):
#     """
#     step: str
#     parameters: object
#     space: object

#     def __post_init__(self) -> None:
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self


# @dataclass
# class QuotientFeatures(Algorithm):
#     """[summary]

#     Args:
#         step (str):
#         parameters (dict):
#         space (dict):
#     """
#     step: str
#     parameters: object
#     space: object

#     def __post_init__(self) -> None:
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self


# @dataclass
# class SumFeatures(Algorithm):
#     """[summary]

#     Args:
#         step (str):
#         parameters (dict):
#         space (dict):
#     """
#     step: str
#     parameters: object
#     space: object

#     def __post_init__(self) -> None:
#         self.idea_sections = ['chef']
#         super().__post_init__()
#         return self