"""
.. module:: encoder
:synopsis: converts categorical features to numeric ones
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from simplify.core.contributor import SimpleContributor
from simplify.core.contributor import Outline


@dataclass
class Encoder(SimpleContributor):
    """Transforms categorical data to numerical data.

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

    name: str = 'encoder'

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        super().draft()
        self.options = {
            'backward': Outline(
                name = 'backward',
                module = 'category_encoders',
                algorithm = 'BackwardDifferenceEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'basen': Outline(
                name = 'basen',
                module = 'category_encoders',
                algorithm = 'BaseNEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'binary': Outline(
                name = 'binary',
                module = 'category_encoders',
                algorithm = 'BinaryEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'dummy': Outline(
                name = 'dummy',
                module = 'category_encoders',
                algorithm = 'OneHotEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'hashing': Outline(
                name = 'hashing',
                module = 'category_encoders',
                algorithm = 'HashingEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'helmert': Outline(
                name = 'helmert',
                module = 'category_encoders',
                algorithm = 'HelmertEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'james_stein': Outline(
                name = 'james_stein',
                module = 'category_encoders',
                algorithm = 'JamesSteinEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'loo': Outline(
                name = 'loo',
                module = 'category_encoders',
                algorithm = 'LeaveOneOutEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'm_estimate': Outline(
                name = 'm_estimate',
                module = 'category_encoders',
                algorithm = 'MEstimateEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'ordinal': Outline(
                name = 'ordinal',
                module = 'category_encoders',
                algorithm = 'OrdinalEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'polynomial': Outline(
                name = 'polynomial_encoder',
                module = 'category_encoders',
                algorithm = 'PolynomialEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'sum': Outline(
                name = 'sum',
                module = 'category_encoders',
                algorithm = 'SumEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'target': Outline(
                name = 'target',
                module = 'category_encoders',
                algorithm = 'TargetEncoder',
                data_dependent = {'cols': 'categoricals'}),
            'woe': Outline(
                name = 'weight_of_evidence',
                module = 'category_encoders',
                algorithm = 'WOEEncoder',
                data_dependent = {'cols': 'categoricals'})}
        return self
